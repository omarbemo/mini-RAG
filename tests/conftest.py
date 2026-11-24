import os

# Settings are required at import time by the controllers; tests need no real
# services, so give harmless values when no .env is present.
for key, value in {
    "APP_NAME": "MINI_RAG_TEST",
    "VERSION": "0",
    "KEY": "x",
    "FILE_ALLOWED_TYPES": '["text/plain","application/pdf"]',
    "FILE_MAX_SIZE": "10",
    "FILE_DEFAULT_CHUNK_SIZE": "512000",
    "MONGODB_URL": "mongodb://localhost:1",
    "MONGODB_DATABASE": "unused",
    "GENERATION_BACKEND": "OPENAI",
    "EMBEDDING_BACKEND": "ONNX",
    "VECTOR_DB_BACKEND": "QDRANT",
    "VECTOR_DB_PATH": "unused",
    "VECTOR_DB_DISTANCE_METHOD": "cosine",
    "OPENAI_API_KEY": "x",
    "OPENAI_API_URL": "http://localhost:1",
    "COHERE_API_KEY": "x",
    "GENERATION_MODEL_ID_LITERAL": '["x"]',
    "GENERATION_MODEL_ID": "x",
    "EMBEDDING_MODEL_ID": "x",
    "EMBEDDING_MODEL_SIZE": "384",
    "INPUT_DAFAULT_MAX_CHARACTERS": "1024",
    "GENERATION_DAFAULT_MAX_TOKENS": "200",
    "GENERATION_DAFAULT_TEMPERATURE": "0.1",
}.items():
    os.environ.setdefault(key, value)
