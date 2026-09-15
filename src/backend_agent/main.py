import logging
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import Redis

from backend_agent.api.router import router
from backend_agent.api.schemas import ErrorDetail, ErrorResponse
from backend_agent.core.config import Settings, get_settings
from backend_agent.core.logging import bind_trace, configure_logging, new_trace_id
from backend_agent.errors import AppError
from backend_agent.queue.rabbitmq import RabbitTaskQueue
from backend_agent.rag.knowledge_base import SQLiteKnowledgeBase
from backend_agent.repositories.redis_store import RedisStore
from backend_agent.services.task_service import TaskService


logger = logging.getLogger(__name__)
_trace_id_pattern = re.compile(r"^[A-Za-z0-9._-]{1,128}$")


@dataclass(slots=True)
class AppContainer:
    settings: Settings
    redis: Redis
    store: RedisStore
    queue: RabbitTaskQueue
    knowledge_base: SQLiteKnowledgeBase
    task_service: TaskService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.redis_socket_timeout_seconds,
        socket_connect_timeout=settings.redis_connect_timeout_seconds,
    )
    store = RedisStore(
        redis,
        task_ttl_seconds=settings.task_ttl_seconds,
        session_ttl_seconds=settings.session_ttl_seconds,
        event_stream_max_length=settings.event_stream_max_length,
        sse_block_milliseconds=settings.sse_block_milliseconds,
    )
    queue = RabbitTaskQueue(
        settings.rabbitmq_url,
        prefetch=settings.worker_prefetch,
        heartbeat=settings.rabbitmq_heartbeat_seconds,
        operation_timeout_seconds=settings.queue_operation_timeout_seconds,
    )
    knowledge_base = SQLiteKnowledgeBase(
        settings.knowledge_db_path,
        settings.knowledge_source_dir,
    )
    await store.ping()
    await queue.connect()
    await knowledge_base.initialize()
    task_service = TaskService(store=store, queue=queue, knowledge_base=knowledge_base)
    app.state.container = AppContainer(
        settings=settings,
        redis=redis,
        store=store,
        queue=queue,
        knowledge_base=knowledge_base,
        task_service=task_service,
    )
    logger.info("api.started")
    try:
        yield
    finally:
        await queue.close()
        await redis.aclose()
        logger.info("api.stopped")


app = FastAPI(title="Backend Agent", version="0.1.0", lifespan=lifespan)
app.include_router(router)


@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    provided_trace_id = request.headers.get("X-Trace-ID", "")
    trace_id = provided_trace_id if _trace_id_pattern.fullmatch(provided_trace_id) else new_trace_id()
    bind_trace(trace_id=trace_id)
    request.state.trace_id = trace_id
    response = await call_next(request)
    response.headers["X-Trace-ID"] = trace_id
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    trace_id = getattr(request.state, "trace_id", "")
    body = ErrorResponse(
        code=exc.code,
        message=exc.message,
        data=ErrorDetail(code=exc.code, message=exc.message, trace_id=trace_id),
    )
    return JSONResponse(status_code=exc.http_status, content=body.model_dump(mode="json"))


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    del exc
    trace_id = getattr(request.state, "trace_id", "")
    message = "请求参数校验失败"
    body = ErrorResponse(
        code="VALIDATION_ERROR",
        message=message,
        data=ErrorDetail(code="VALIDATION_ERROR", message=message, trace_id=trace_id),
    )
    return JSONResponse(status_code=422, content=body.model_dump(mode="json"))


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error(
        "api.unhandled_error",
        exc_info=(type(exc), exc, exc.__traceback__),
    )
    trace_id = getattr(request.state, "trace_id", "")
    message = "服务内部错误"
    body = ErrorResponse(
        code="INTERNAL_ERROR",
        message=message,
        data=ErrorDetail(code="INTERNAL_ERROR", message=message, trace_id=trace_id),
    )
    return JSONResponse(status_code=500, content=body.model_dump(mode="json"))
