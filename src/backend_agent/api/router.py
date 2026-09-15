import json
import re
from collections.abc import AsyncIterator

from fastapi import APIRouter, Header, Query, Request, status
from fastapi.responses import StreamingResponse

from backend_agent.api.schemas import (
    AddKnowledgeDocumentData,
    AddKnowledgeDocumentRequest,
    ApiResponse,
    CreateTaskData,
    CreateTaskRequest,
    HealthData,
    ResumeTaskData,
    TaskStatusData,
)
from backend_agent.errors import AppError


router = APIRouter(prefix="/api/v1")
_event_id_pattern = re.compile(r"^\d+-\d+$")


@router.get("/healthz", response_model=ApiResponse[HealthData])
async def health(request: Request) -> ApiResponse[HealthData]:
    container = request.app.state.container
    await container.store.ping()
    return ApiResponse(data=HealthData(status="ok"))


@router.post(
    "/tasks",
    response_model=ApiResponse[CreateTaskData],
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_task(
    payload: CreateTaskRequest,
    request: Request,
    idempotency_key: str | None = Header(
        default=None,
        alias="Idempotency-Key",
        min_length=8,
        max_length=128,
    ),
) -> ApiResponse[CreateTaskData]:
    container = request.app.state.container
    task, created = await container.task_service.create_task(
        prompt=payload.prompt,
        session_id=payload.session_id,
        idempotency_key=idempotency_key,
        trace_id=request.state.trace_id,
    )
    return ApiResponse(
        data=CreateTaskData(
            task_id=task.task_id,
            session_id=task.session_id,
            trace_id=task.trace_id,
            status=task.status,
            created=created,
            events_path=f"/api/v1/tasks/{task.task_id}/events",
        )
    )


@router.get("/tasks/{task_id}", response_model=ApiResponse[TaskStatusData])
async def get_task(task_id: str, request: Request) -> ApiResponse[TaskStatusData]:
    container = request.app.state.container
    task = await container.task_service.get_task(task_id)
    return ApiResponse(data=TaskStatusData.from_record(task))


@router.post(
    "/tasks/{task_id}/resume",
    response_model=ApiResponse[ResumeTaskData],
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume_task(task_id: str, request: Request) -> ApiResponse[ResumeTaskData]:
    container = request.app.state.container
    task = await container.task_service.resume_task(task_id)
    return ApiResponse(
        data=ResumeTaskData(
            task_id=task.task_id,
            session_id=task.session_id,
            trace_id=task.trace_id,
            status=task.status,
            events_path=f"/api/v1/tasks/{task.task_id}/events",
        )
    )


@router.get("/tasks/{task_id}/events")
async def stream_task_events(
    task_id: str,
    request: Request,
    after: str | None = Query(default=None),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    container = request.app.state.container
    await container.task_service.get_task(task_id)
    cursor = last_event_id or after or "0-0"
    if not _event_id_pattern.fullmatch(cursor):
        raise AppError("INVALID_EVENT_ID", "Last-Event-ID 格式无效", http_status=422)

    async def event_generator() -> AsyncIterator[str]:
        yield "retry: 3000\n\n"
        async for event in container.store.iter_events(task_id, cursor):
            if await request.is_disconnected():
                return
            if event is None:
                yield ": ping\n\n"
                continue
            data = json.dumps(
                {
                    "sequence_number": event.sequence_number,
                    **event.payload,
                },
                ensure_ascii=False,
                default=str,
            )
            yield f"id: {event.event_id}\nevent: {event.event_type}\ndata: {data}\n\n"
            if event.event_type in {"task.completed", "task.failed"}:
                return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/knowledge/documents",
    response_model=ApiResponse[AddKnowledgeDocumentData],
    status_code=status.HTTP_201_CREATED,
)
async def add_knowledge_document(
    payload: AddKnowledgeDocumentRequest,
    request: Request,
) -> ApiResponse[AddKnowledgeDocumentData]:
    container = request.app.state.container
    document_id = await container.task_service.add_knowledge_document(
        title=payload.title,
        content=payload.content,
        metadata=payload.metadata,
    )
    return ApiResponse(data=AddKnowledgeDocumentData(document_id=document_id))
