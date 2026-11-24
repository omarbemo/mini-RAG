from .BaseController import BaseController
from mini_rag.models.db_schemas.data_chunk import DataChunk
from mini_rag.models.db_schemas.project import Project
from mini_rag.stores.llm.LLMEnums import DocumentTypeEnum
from mini_rag.rag.fusion import rrf_fuse
from mini_rag.rag.sparse import Bm25Encoder
from typing import List
import json
import uuid

class NLPController(BaseController):

    def __init__(self, vectordb_client, generation_client, 
                 embedding_client , template_parser=None,
                 reranker=None, sparse_encoder=None):
        super().__init__()

        self.vectordb_client = vectordb_client
        self.generation_client = generation_client
        self.embedding_client = embedding_client
        self.template_parser = template_parser
        self.reranker = reranker
        self.sparse_encoder = sparse_encoder or Bm25Encoder()

    @classmethod
    def from_app(cls, app):
        """Controller wired to the clients created at startup (see main.py)."""
        return cls(
            vectordb_client=app.vectordb_client,
            generation_client=app.generation_client,
            embedding_client=app.embedding_client,
            template_parser=app.template_parser,
            reranker=getattr(app, "reranker", None),
            sparse_encoder=getattr(app, "sparse_encoder", None),
        )

    @staticmethod
    def chunk_point_id(chunk_id) -> str:
        """Stable Qdrant id per Mongo chunk, so indexing a chunk twice overwrites it."""
        if chunk_id is None:
            return str(uuid.uuid4())
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"mini-rag/chunk/{chunk_id}"))
        

    def create_collection_name(self, project_id: str):
        return f"collection_{project_id}".strip()
    
    def reset_vector_db_collection(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        return self.vectordb_client.delete_collection(collection_name=collection_name)
    
    def get_vector_db_collection_info(self, project: Project):
        collection_name = self.create_collection_name(project_id=project.project_id)
        collection_info = self.vectordb_client.get_collection_info(collection_name=collection_name)

        return json.loads(
            json.dumps(collection_info, default=lambda x: x.__dict__)
        )
    
    def index_into_vector_db(self, project: Project, chunks: List[DataChunk],
                                   chunks_ids: List = None, 
                                   do_reset: bool = False):
        """Embed chunks (dense + BM25) and upsert them; ids default to chunk_point_id."""
        if chunks_ids is None:
            chunks_ids = [self.chunk_point_id(c.id) for c in chunks]
        
        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)

        # step2: manage items
        texts = [ c.chunk_text for c in chunks ]
        metadata = [ c.chunk_metadata for c in  chunks]
        vectors = [
            self.embedding_client.embed_text(text=text, 
                                             document_type=DocumentTypeEnum.DOCUMENT.value)
            for text in texts
        ]
        sparse_vectors = self.sparse_encoder.encode_docs(texts)

        # step3: create collection if not exists
        _ = self.vectordb_client.create_collection(
            collection_name=collection_name,
            embedding_size=self.embedding_client.embedding_size,
            do_reset=do_reset,
        )

        # step4: insert into vector db
        return self.vectordb_client.insert_many(
            collection_name=collection_name,
            texts=texts,
            metadata=metadata,
            vectors=vectors,
            record_ids=chunks_ids,
            sparse_vectors=sparse_vectors,
        )

    def search_vector_db_collection(self, project: Project, text: str, limit: int = 10,
                                    rerank: bool = True, pool: int = 30):
        """Hybrid search: dense + BM25 lists fused with RRF, then the reranker
        (when loaded and `rerank`) reorders the top `pool`.

        `score` is the reranker logit when reranked, else the RRF score.
        """

        # step1: get collection name
        collection_name = self.create_collection_name(project_id=project.project_id)
        if not self.vectordb_client.is_collection_existed(collection_name):
            return False

        # step2: get text embedding vector
        vector = self.embedding_client.embed_text(text=text, 
                                                 document_type=DocumentTypeEnum.QUERY.value)

        if not vector or len(vector) == 0:
            return False

        # step3: dense + keyword lists, fused
        lists = self.vectordb_client.search_lists(
            collection_name=collection_name,
            vector=vector,
            sparse_vector=self.sparse_encoder.encode_query(text),
            limit=max(pool, limit),
        )
        docs = {d.id: d for ranked in lists for d in ranked}
        results = [
            docs[i].model_copy(update={"score": score})
            for i, score in rrf_fuse([[d.id for d in ranked] for ranked in lists])
        ]

        # step4: cross-encoder rerank of the head
        if rerank and self.reranker is not None and results:
            head = results[:pool]
            scores = self.reranker.score(text, [d.text for d in head])
            head = [
                d.model_copy(update={"score": s})
                for d, s in sorted(zip(head, scores), key=lambda ds: -ds[1])
            ]
            results = head + results[pool:]

        return results[:limit] or False
    
    def answer_rag_question(self, project: Project, query: str, limit: int = 10):
        
        answer, full_prompt, chat_history = None, None, None

        # step1: retrieve related documents
        retrieved_documents = self.search_vector_db_collection(
            project=project,
            text=query,
            limit=limit,
        )

        if not retrieved_documents or len(retrieved_documents) == 0:
            return answer, full_prompt, chat_history
        
        # step2: Construct LLM prompt
        system_prompt = self.template_parser.get("rag", "system_prompt")

        documents_prompts = "\n".join([
            self.template_parser.get("rag", "document_prompt", {
                    "doc_num": idx + 1,
                    "chunk_text": doc.text,
            })
            for idx, doc in enumerate(retrieved_documents)
        ])

        footer_prompt = self.template_parser.get("rag", "footer_prompt", {"query": query})

        # step3: Construct Generation Client Prompts
        chat_history = [
            self.generation_client.construct_prompt(
                prompt=system_prompt,
                role=self.generation_client.enums.SYSTEM.value,
            )
        ]

        full_prompt = "\n\n".join([ documents_prompts,  footer_prompt])

        # step4: Retrieve the Answer
        answer = self.generation_client.generate_text(
            prompt=full_prompt,
            chat_history=chat_history
        )

        return answer, full_prompt, chat_history

