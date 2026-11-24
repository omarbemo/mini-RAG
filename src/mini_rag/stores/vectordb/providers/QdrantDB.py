from qdrant_client import models, QdrantClient
from ..VectorDBInterface import VectorDBInterface
from ..VectorDBEnums import DistanceMethodEnums
import logging
from typing import List
import uuid
from mini_rag.models.db_schemas.data_chunk import RetrievedDocument

# Named sparse vector for BM25 keyword search; the dense vector stays unnamed.
SPARSE_VECTOR_NAME = "bm25"

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
                ),
                # Qdrant applies IDF itself, so BM25 scores track the corpus.
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
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
    
    def has_sparse_vectors(self, collection_name: str) -> bool:
        """Collections made before hybrid search have no BM25 vector; reset to add it."""
        params = self.client.get_collection(collection_name=collection_name).config.params
        return SPARSE_VECTOR_NAME in (params.sparse_vectors or {})

    def insert_many(self, collection_name: str, texts: list,
                    vectors: list, metadata: list = None,
                    record_ids: list = None, batch_size: int = 32,
                    sparse_vectors: list = None):

        if metadata is None:
            metadata = [None] * len(texts)

        if record_ids is None:
            record_ids = [str(uuid.uuid4()) for _ in texts]

        if sparse_vectors is not None and self.has_sparse_vectors(collection_name):
            vectors = [
                {"": dense, SPARSE_VECTOR_NAME: sparse}
                for dense, sparse in zip(vectors, sparse_vectors)
            ]

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
        ).points
        if not results or len(results) == 0:
            return None
        
        return [self._to_document(result) for result in results]

    def search_lists(self, collection_name: str, vector: list,
                     sparse_vector: models.SparseVector = None, limit: int = 30):
        """Ranked lists for hybrid search, sent in one batch: dense first, then BM25.

        The BM25 list is left out when the query has no keywords (stop words only)
        or the collection has no sparse vector.
        """
        requests = [models.QueryRequest(query=vector, limit=limit, with_payload=True)]
        if (sparse_vector is not None and sparse_vector.indices
                and self.has_sparse_vectors(collection_name)):
            requests.append(models.QueryRequest(
                query=sparse_vector, using=SPARSE_VECTOR_NAME, limit=limit, with_payload=True
            ))
        responses = self.client.query_batch_points(
            collection_name=collection_name, requests=requests
        )
        return [[self._to_document(p) for p in r.points] for r in responses]

    @staticmethod
    def _to_document(point) -> RetrievedDocument:
        return RetrievedDocument(
            id=str(point.id),
            score=point.score,
            text=point.payload["text"],
            metadata=point.payload.get("metadata"),
        )