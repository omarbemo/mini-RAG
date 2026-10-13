from mini_rag.main import app_settings
from fastapi import FastAPI
from mini_rag.api import base , data
from mini_rag.helpers import get_settings , settings
from pymongo import AsyncMongoClient

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    app_settings = get_settings()
    app.mongo_conn = AsyncMongoClient(app_settings.MONGODB_URL)
    app.db_client = app.mongo_conn[app_settings.MONGODB_DATABASE]

@app.on_event("shutdown")
async def shutdown_db_client():
    app.mongo_conn.close()



app.include_router(base.base_router)
app.include_router(data.data_router)