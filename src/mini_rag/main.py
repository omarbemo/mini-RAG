from contextlib import asynccontextmanager

from fastapi import FastAPI
from mini_rag.helpers import get_settings
from mini_rag.api import base, data, nlp
from mini_rag.stores.llm import LLMProviderFactory
from mini_rag.stores.vectordb import VectorDBProviderFactory
from mini_rag.stores.llm.templates.template_parser import TemplateParser
from pymongo import AsyncMongoClient
from mini_rag.rag.rerank import load_reranker
from mini_rag.rag.sparse import Bm25Encoder


async def startup_span(app: FastAPI):
    app_settings = get_settings()

    app.mongo_conn = AsyncMongoClient(app_settings.MONGODB_URL)
    app.db_client = app.mongo_conn[app_settings.MONGODB_DATABASE]

    llm_provider_factory = LLMProviderFactory(app_settings)
    vectordb_provider_factory = VectorDBProviderFactory(app_settings)

    app.generation_client = llm_provider_factory.create(
        provider=app_settings.GENERATION_BACKEND
    )
    app.generation_client.set_generation_model(
        model_id=app_settings.GENERATION_MODEL_ID
    )

    app.embedding_client = llm_provider_factory.create(
        provider=app_settings.EMBEDDING_BACKEND
    )
    app.embedding_client.set_embedding_model(
        model_id=app_settings.EMBEDDING_MODEL_ID,
        embedding_size=app_settings.EMBEDDING_MODEL_SIZE,
    )

    app.vectordb_client = vectordb_provider_factory.create(
        provider=app_settings.VECTOR_DB_BACKEND
    )
    app.vectordb_client.connect()

    # Hybrid search: BM25 keyword vectors + optional cross-encoder reranker.
    app.sparse_encoder = Bm25Encoder()
    app.reranker = load_reranker() if app_settings.RERANKER_ENABLED else None

    app.template_parser = TemplateParser(
        language=app_settings.PRIMARY_LANG,
        default_language=app_settings.DEFAULT_LANG,
    )


async def shutdown_span(app: FastAPI):
    await app.mongo_conn.close()
    app.vectordb_client.disconnect()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await startup_span(app)
    yield
    await shutdown_span(app)


app = FastAPI(lifespan=lifespan)

app.include_router(base.base_router)
app.include_router(data.data_router)
app.include_router(nlp.nlp_router)