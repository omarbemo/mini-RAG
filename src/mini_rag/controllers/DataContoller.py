from mini_rag.controllers import BaseController
from mini_rag.controllers import ProjectController
from fastapi import FastAPI , APIRouter , Depends , UploadFile
from mini_rag.models import ResponseEnum
import os
import re



class DataController(BaseController):
    def __init__(self):
        super().__init__()
        self.size_scale = 1048576   # convert Mb to bytes
    
    def validate_uploaded_file(self , file : UploadFile):
        if(file.content_type not in self.app_settings.FILE_ALLOWED_TYPES):
            return False , ResponseEnum.FILE_TYPE_NOT_SUPPORTED.value
        if(file.size > self.app_settings.FILE_MAX_SIZE * self.size_scale):
            return False , ResponseEnum.FILE_SIZE_EXCEEDED.value

        return True , ResponseEnum.FILE_VALIDATED_SUCCESSFUL.value



    def generate_uniuqe_filename(self, orig_file_name:str , project_id : str):
        random_key = self.generate_rand_string()
        project_path = ProjectController().get_project_path(project_id=project_id)
        cleaned_filename=self.get_clean_filename(orig_file_name=orig_file_name)
        new_file_path = os.path.join(project_path,
                        random_key + "_" + cleaned_filename)
        while os.path.exists(new_file_path):
            new_file_path = os.join(project_path,
                        random_key + "_" + cleaned_filename)
        return new_file_path




    def get_clean_filename ( self , orig_file_name:str):
        cleaned_filename  = re.sub(r'[^\w.]','',orig_file_name.strip())
        cleaned_filename = cleaned_filename.replace(" ","_")
        return cleaned_filename