from pydantic import functional_serializers
from fastapi import FastAPI , APIRouter , Depends , UploadFile ,status
from fastapi.responses import JSONResponse
from mini_rag.helpers import get_settings , settings
from mini_rag.controllers import DataController , ProjectController
import aiofiles
from mini_rag.models import ResponseEnum
import os
import logging

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
    file_path = data_controller.generate_uniuqe_filename(orig_file_name=file.filename , project_id=project_id)


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
            content={"signal":ResponseEnum.FILE_UPLOADED_SUCCESS.value}
        )


    
    




