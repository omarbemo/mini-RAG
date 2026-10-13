from pydantic import functional_serializers
from fastapi import FastAPI , APIRouter , Depends , UploadFile ,status
from fastapi.responses import JSONResponse
from mini_rag.helpers import get_settings , settings
from mini_rag.controllers import DataController , ProjectController , ProcessController
import aiofiles
from mini_rag.models import ResponseEnum
import os
import logging
from .schemas.data import ProcessRequest 

logger = logging.getLogger("uvicorn_error")



data_router = APIRouter(
    prefix="/api/v1/data",
    tags = ['api_v1','data']
)

@data_router.post("/upload/{project_id}")
async def upload_data(project_id : str , file : UploadFile ,app_settings = Depends(get_settings)):
    
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
async def process_endpoint(project_id:str,process_request : ProcessRequest):
    file_id = process_request.file_id
    chunk_size = process_request.chunk_size
    overlap = process_request.overlap



    processcontroller = ProcessController(project_id=project_id)
    file_content = processcontroller.get_file_content(file_id=file_id)
    file_chunks = processcontroller.process_file_content(file_content=file_content,
                        file_id=file_id , chunk_size=chunk_size,overlap=overlap )
    
    if file_chunks is None or len(file_chunks)==0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"signal":ResponseEnum.FILE_CHUNKS_FAILED.value}
        )
    else:
        return file_chunks




