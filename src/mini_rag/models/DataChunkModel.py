from mini_rag.models import BaseDataModel
from mini_rag.models.db_schemas import DataChunk
from mini_rag.models.enums.DataBaseEnum import DataBaseEnum
from bson.objectid import ObjectId
from pymongo import InsertOne

class DataChunkModel(BaseDataModel):
    def __init__(self , db_client : object):
        super().__init__(db_client=db_client)
        self.collection = self.db_client[DataBaseEnum.COLLECTION_CHUNKS_NAME.value]

    async def create_project(self,chunk : DataChunk):
        result = await self.collection.insert_one(chunk.dict())
        chunk._id = result.inserted_id
        return chunk

    async def get_chunk(self , chunk_id : str):
        chunk_id = ObjectId(chunk_id)
        result = self.collection.find_one({"chunk_id":chunk_id})
        if result is None :
            return None
        else :
            return DataChunk(**result)
    async def insert_many_chunks(self , chunks : list , batch_size : int = 100):
        for i in range(0 , len(chunks) , batch_size):
            batch = chunks[i : i+batch_size]
            operations = [
                InsertOne(chunk.dict()) 
                for chunk in batch
            ]
            await self.collection.bulk_write(operations)
        return len(chunks)
        
    async def delete_chunk_by_project_id(self,project_id : ObjectId):
        result = await self.collection.delete_many({"chunk_project_id":project_id})
        return result.deleted_count