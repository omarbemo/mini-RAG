from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
import uuid
from mini_rag.models.db_schemas.data_chunk import RetrievedDocument

class QdrantDBProvider(VectorDBInterface):

    def __init__(self, db_path: str, distance_method: str):

        self.client = None
        self.db_path = db_path
        self.distance_method = None

        if distance_method == DistanceMethodEnums.COSINE.value:
            self.distance_method = models.Distance.COSINE
        elif distance_method == DistanceMethodEnums.DOT.value:
            self.distance_method = models.Distance.DOT

        self.logger = logging.getLogger(__name__)

    def connect(self):
        self.client = QdrantClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        return self.client.collection_exists(collection_name=collection_name)
    
    def list_all_collections(self) -> List:
        return self.client.get_collections()
    
    def get_collection_info(self, collection_name: str) -> dict:
        return self.client.get_collection(collection_name=collection_name)
    
    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            return self.client.delete_collection(collection_name=collection_name)
        
    def create_collection(self, collection_name: str, 
                                embedding_size: int,
                                do_reset: bool = False):
        if do_reset:
            _ = self.delete_collection(collection_name=collection_name)
        
        if not self.is_collection_existed(collection_name):
            _ = self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=embedding_size,
                    distance=self.distance_method
                )
            )

            return True
        
        return False
    
    def insert_one(self, collection_name: str, text: str, vector: list,
               metadata: dict = None, record_id: str = None):

        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Can not insert new record to non-existed collection: {collection_name}")
            return False

        if record_id is None:
            record_id = str(uuid.uuid4())

        point = models.PointStruct(
            id=record_id,
            vector=vector,
            payload={"text": text, "metadata": metadata},
        )

        try:
            self.client.upsert(
                collection_name=collection_name,
                points=[point],
                wait=True,
            )
        except Exception as e:
            self.logger.error(f"Error while inserting point: {e}")
            return False

        return True
    
    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadata: list = None,
                    record_ids: list = None, batch_size: int = 32):

        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = [str(uuid.uuid4()) for _ in texts]

        points = [
            models.PointStruct(
                id=record_ids[i],
                vector=vectors[i],
                payload={"text": texts[i], "metadata": metadata[i]},
            )
            for i in range(len(texts))
        ]

        try:
            self.client.upload_points(
                collection_name=collection_name,
                points=points,
                batch_size=batch_size,  # lower to 16 if it still times out
                parallel=1,             # keep at 1 on a slow connection
                max_retries=5,
                wait=True,
            )
        except Exception as e:
            self.logger.error(f"Error while inserting points: {e}")
            return False

        return True
        
    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        results = self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=limit,
            with_payload=True,
        )
        if not results or len(results) == 0:
            return None
        
        return [
            RetrievedDocument(**{
                "score": result.score,
                "text": result.payload["text"],
            })
            for result in results
        ]