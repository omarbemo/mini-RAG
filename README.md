how to run the main app
uv run uvicorn mini_rag.main:app --app-dir src --reload

## flow

1. `POST /api/v1/data/upload/{project_id}` (form `file`) -> `file_id`
2. `POST /api/v1/data/process/{project_id}` `{"file_id": "...", "do_reset": 0, "do_push": 1}`
   -> chunks saved in Mongo **and** embedded into Qdrant (`do_push: 0` = Mongo only)
3. `POST /api/v1/nlp/index/push/{project_id}` `{"do_reset": 1}` -> re-index all Mongo chunks
4. `POST /api/v1/nlp/index/search/{project_id}` `{"text": "...", "limit": 5, "rerank": true}`
5. `POST /api/v1/nlp/index/answer/{project_id}` -> LLM answer from the top chunks

## search

Hybrid: dense vectors (embedding backend) + BM25 keyword vectors (English
stemming), fused with RRF, then reranked by bge-reranker-v2-m3 (ONNX, local).
`EMBEDDING_BACKEND="ONNX"` embeds locally with bge-small-en-v1.5 instead of Cohere.
Collections made before this need one push with `do_reset: 1` to get BM25.

## tests

    uv run pytest
