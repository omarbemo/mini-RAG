from pydantic import functional_serializers
from fastapi import FastAPI , APIRouter , Depends , UploadFile ,status , Request
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from mini_rag.helpers import get_settings , settings
from mini_rag.controllers import DataController , ProjectController , ProcessController
import aiofiles
from mini_rag.models import ResponseEnum 
from mini_rag.models.ProjectModel import ProjectModel
from mini_rag.models.DataChunkModel import DataChunkModel
from mini_rag.models.AssetModel import AssetModel
import os
import logging
from .schemas.data import ProcessRequest 
from .schemas.nlp import PushRequest , SearchRequest
from mini_rag.models.db_schemas.data_chunk import DataChunk
from mini_rag.models.db_schemas.asset import Asset
from mini_rag.controllers.NLPController import NLPController

logger = logging.getLogger("uvicorn_error")

nlp_router = APIRouter(
    prefix="/api/v1/nlp",
    tags=["api_v1","nlp"]
)

@nlp_router.post("/index/push/{project_id}")
async def index_project(request: Request, project_id: str, push_request: PushRequest):

    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    chunk_model = await DataChunkModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    if not project:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={
                "signal": ResponseEnum.PROJECT_NOT_FOUND_ERROR.value
            }
        )
    
    nlp_controller = NLPController.from_app(request.app)

    has_records = True
    page_no = 1
    inserted_items_count = 0

    while has_records:
        page_chunks = await chunk_model.get_poject_chunks(project_id=project.id, page_no=page_no)
        if len(page_chunks):
            page_no += 1
        
        if not page_chunks or len(page_chunks) == 0:
            has_records = False
            break

        # Point ids come from the Mongo chunk ids, so re-pushing overwrites.
        # Reset only before the first page, or each page would wipe the last.
        is_inserted = await run_in_threadpool(
            nlp_controller.index_into_vector_db,
            project=project,
            chunks=page_chunks,
            do_reset=push_request.do_reset and page_no == 2,
        )

        if not is_inserted:
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnum.INSERT_INTO_VECTORDB_ERROR.value
                }
            )
        
        inserted_items_count += len(page_chunks)
        
    return JSONResponse(
        content={
            "signal": ResponseEnum.INSERT_INTO_VECTORDB_SUCCESS.value,
            "inserted_items_count": inserted_items_count
        }
    )

@nlp_router.get("/index/info/{project_id}")
async def get_project_index_info(request: Request, project_id: str):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController.from_app(request.app)

    collection_info = nlp_controller.get_vector_db_collection_info(project=project)

    return JSONResponse(
        content={
            "signal": ResponseEnum.VECTORDB_COLLECTION_RETRIEVED.value,
            "collection_info": collection_info
        }
    )

@nlp_router.post("/index/search/{project_id}")
async def search_index(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController.from_app(request.app)

    results = await run_in_threadpool(
        nlp_controller.search_vector_db_collection,
        project=project, text=search_request.text, limit=search_request.limit,
        rerank=bool(search_request.rerank),
    )

    if not results:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnum.VECTORDB_SEARCH_ERROR.value
                }
            )
    
    return JSONResponse(
        content={
            "signal": ResponseEnum.VECTORDB_SEARCH_SUCCESS.value,
            "results": [ result.model_dump()  for result in results ]
        }
    )

@nlp_router.post("/index/answer/{project_id}")
async def answer_rag(request: Request, project_id: str, search_request: SearchRequest):
    
    project_model = await ProjectModel.create_instance(
        db_client=request.app.db_client
    )

    project = await project_model.get_project_or_create_one(
        project_id=project_id
    )

    nlp_controller = NLPController.from_app(request.app)

    answer, full_prompt, chat_history = await run_in_threadpool(
        nlp_controller.answer_rag_question,
        project=project,
        query=search_request.text,
        limit=search_request.limit,
    )

    if not answer:
        return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={
                    "signal": ResponseEnum.RAG_ANSWER_ERROR.value
                }
        )
    
    return JSONResponse(
        content={
            "signal": ResponseEnum.RAG_ANSWER_SUCCESS.value,
            "answer": answer,
            "full_prompt": full_prompt,
            "chat_history": chat_history
        }
    )
