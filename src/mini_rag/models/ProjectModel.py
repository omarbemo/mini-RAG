from mini_rag.models import BaseDataModel
from mini_rag.models.db_schemas import Project
from mini_rag.models.enums.DataBaseEnum import DataBaseEnum

class ProjectModel(BaseDataModel):
    def __init__(self , db_client : object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_PROJEECT_NAME.value]

    async def create_project(self,project : Project):
        result = await self.collection.insert_one(project.dict())
        project._id = result.inserted_id
        return project
        

    async def get_project_or_create_one(self,project_id : str):
        recrod = await self.collection.find_one({"project_id":project_id})
        if recrod is None :
            project = Project(project_id=project_id)
            project = await self.create_project(project=project)
            return project
        return Project(**recrod)

    async def get_all_projects(self , page : int=1 , page_size : int=10):
        ### PAGINATION

        #count total number of documents
        total_documents = await self.collection.count_documents({})
        #count total number of pages
        total_pages = total_documents // page_size
        if total_documents % page_size > 0:
            total_pages +=1

        cursor = self.collection.find().skip((page-1)*page_size).limit(page_size)
        projects=[]
        async for document in cursor:
            projects.append(Project(**document))
        return projects , total_pages
