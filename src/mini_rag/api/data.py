from pydantic import functional_serializers
from fastapi import FastAPI , APIRouter , Depends , UploadFile ,status , Request
from fastapi.responses import JSONResponse
from mini_rag.helpers import get_settings , settings
from mini_rag.controllers import DataController , ProjectController , ProcessController
import aiofiles
from mini_rag.models import ResponseEnum 
from mini_rag.models.ProjectModel import ProjectModel
from mini_rag.models.DataChunkModel import DataChunkModel
import os
import logging
from .schemas.data import ProcessRequest 
from mini_rag.models.db_schemas.data_chunk import DataChunk

logger = logging.getLogger("uvicorn_error")



data_router = APIRouter(
    prefix="/api/v1/data",
    tags = ['api_v1','data']
)

@data_router.post("/upload/{project_id}")
async def upload_data(request : Request ,project_id : str , file : UploadFile ,app_settings = Depends(get_settings)):
    
    project_model = ProjectModel(request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)


    data_controller = DataController()
    is_valid , signal = data_controller.validate_uploaded_file(file=file)

    if (not is_valid):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":signal}
        )

        
    project_dir_path = ProjectController().get_project_path(project_id=project_id)
    file_path , file_id = data_controller.generate_uniuqe_filepath(orig_file_name=file.filename , project_id=project_id)


    try:
        async with aiofiles.open(file_path,"wb") as f :
            while chunk := await file.read(app_settings.FILE_DEFAULT_CHUNK_SIZE):
                await f.write(chunk)
    except Exception as e :
        logger.error(f"while uploading file {e}")
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":ResponseEnum.FILE_UPLOAD_FAILED.value}
        )



    return JSONResponse(
            content={"signal":ResponseEnum.FILE_UPLOADED_SUCCESS.value,
                    "file_id":file_id}
        )


@data_router.post("/process/{project_id}")
async def process_endpoint(request : Request , project_id:str,process_request : ProcessRequest):
    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap = process_request.overlap
    do_reset =process_request.do_reset 

    project_model = ProjectModel(request.app.db_client)
    project = await project_model.get_project_or_create_one(project_id=project_id)




    processcontroller = ProcessController(project_id=project_id)
    file_content = processcontroller.get_file_content(file_id=file_id)
    file_chunks = processcontroller.process_file_content(file_content=file_content,
                        file_id=file_id , chunk_size=chunk_size,overlap=overlap )
    
    if file_chunks is None or len(file_chunks)==0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":ResponseEnum.FILE_CHUNKS_FAILED.value}
        )
    file_chunks_records = [
        DataChunk( 
                chunk_text = chunk.page_content,
                chunk_metadata = chunk.metadata,
                chunk_order = i+1,
                chunk_project_id = project._id,

        )
        for i , chunk in enumerate(file_chunks)
    ]


    chunk_model = DataChunkModel(
        db_client=request.app.db_client
    )


    if do_reset:
        result = await chunk_model.delete_chunk_by_project_id(project_id=project._id)



    no_rec = await chunk_model.insert_many_chunks(chunks=file_chunks_records,batch_size=100)
    return JSONResponse(
        content={
            "signal": ResponseEnum.PROCESSING_SUCCESS.value ,
            "chunks_inserted": no_rec
        }
        )



