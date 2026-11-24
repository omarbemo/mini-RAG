"""NLPController + QdrantDBProvider end to end on an on-disk local Qdrant.

A hashing embedder stands in for the model, so dense similarity only sees exact
word overlap; stemming-only matches must come from the BM25 list.
"""

import hashlib

import numpy as np
import pytest
from bson.objectid import ObjectId
from qdrant_client import models

from mini_rag.controllers.NLPController import NLPController
from mini_rag.models.db_schemas import DataChunk, Project
from mini_rag.stores.vectordb.providers import QdrantDBProvider

DIM = 64
DOCS = [
    "The invoice must be paid within thirty days of delivery.",
    "Employees accrue vacation leave monthly.",
    "Late payment of an invoice incurs a penalty fee.",
    "The office closes at six on Fridays.",
]


class HashEmbeddingClient:
    embedding_size = DIM

    def embed_text(self, text, document_type=None):
        v = np.zeros(DIM, dtype=np.float32)
        for w in text.lower().split():
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIM] += 1
        return (v / max(np.linalg.norm(v), 1e-12)).tolist()


class WordReranker:
    def __init__(self, word):
        self.word = word

    def score(self, query, texts):
        return [float(self.word in t) for t in texts]


@pytest.fixture
def vectordb(tmp_path):
    db = QdrantDBProvider(db_path=str(tmp_path / "qdrant"), distance_method="cosine")
    db.connect()
    yield db
    db.client.close()


def controller(vectordb, reranker=None):
    return NLPController(vectordb, None, HashEmbeddingClient(), reranker=reranker)


def chunks(texts, project):
    return [
        DataChunk(id=ObjectId(), chunk_text=t, chunk_metadata={"n": i}, chunk_order=i + 1,
                  chunk_project_id=project.id, chunk_asset_id=ObjectId())
        for i, t in enumerate(texts)
    ]


@pytest.fixture
def project():
    return Project(id=ObjectId(), project_id="p1")


def test_keyword_list_finds_stemmed_matches(vectordb, project):
    nlp = controller(vectordb)
    assert nlp.index_into_vector_db(project, chunks(DOCS, project))
    # "invoices" never appears verbatim, so only BM25 (stemmed) can match it.
    results = nlp.search_vector_db_collection(project, "invoices", limit=2, rerank=False)
    assert {r.text for r in results} == {DOCS[0], DOCS[2]}
    assert results[0].metadata in ({"n": 0}, {"n": 2})


def test_reranker_reorders_and_scores(vectordb, project):
    nlp = controller(vectordb, reranker=WordReranker("penalty"))
    nlp.index_into_vector_db(project, chunks(DOCS, project))
    results = nlp.search_vector_db_collection(project, "invoice", limit=3)
    assert results[0].text == DOCS[2] and results[0].score == 1.0
    plain = nlp.search_vector_db_collection(project, "invoice", limit=3, rerank=False)
    assert plain[0].score < 1  # RRF score, reranker skipped


def test_reindexing_same_chunks_overwrites(vectordb, project):
    nlp = controller(vectordb)
    same = chunks(DOCS, project)
    nlp.index_into_vector_db(project, same)
    nlp.index_into_vector_db(project, same)
    name = nlp.create_collection_name(project.project_id)
    assert vectordb.client.count(name).count == len(DOCS)


def test_missing_collection_returns_false(vectordb, project):
    assert controller(vectordb).search_vector_db_collection(project, "anything") is False


def test_collection_without_sparse_vector_still_searches_dense(vectordb, project):
    nlp = controller(vectordb)
    name = nlp.create_collection_name(project.project_id)
    # Shape of collections created before hybrid search.
    vectordb.client.create_collection(
        name, vectors_config=models.VectorParams(size=DIM, distance=models.Distance.COSINE)
    )
    assert nlp.index_into_vector_db(project, chunks(DOCS, project))
    results = nlp.search_vector_db_collection(project, "office fridays", limit=1, rerank=False)
    assert results[0].text == DOCS[3]
