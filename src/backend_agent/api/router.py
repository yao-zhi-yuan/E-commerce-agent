import json
import re
import secrets
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Depends, Header, Query, Request, status
from fastapi.responses import StreamingResponse

from backend_agent.commerce import DEMO_MERCHANT_ID
from backend_agent.api.schemas import (
    ApiResponse,
    CreateTaskData,
    CreateTaskRequest,
    HealthData,
    ResumeTaskData,
    SubmitApprovalData,
    SubmitApprovalRequest,
    SubmitInputData,
    SubmitInputRequest,
    TaskStatusData,
    TaskTraceData,
    TraceEventData,
)
from backend_agent.errors import AppError


router = APIRouter(prefix="/api/v1")
_event_id_pattern = re.compile(r"^\d+-\d+$")


def authenticated_merchant(
    request: Request,
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> str:
    configured_token = request.app.state.container.settings.demo_api_token.get_secret_value()
    if not configured_token or configured_token == "<SET_ME>":
        raise AppError("AUTH_NOT_CONFIGURED", "演示 API Token 尚未配置", http_status=503)
    scheme, _, provided_token = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(provided_token, configured_token):
        raise AppError("UNAUTHORIZED", "认证失败", http_status=401)
    return DEMO_MERCHANT_ID


MerchantId = Annotated[str, Depends(authenticated_merchant)]


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
    merchant_id: MerchantId,
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
        idempotency_key=(f"{merchant_id}:{idempotency_key}" if idempotency_key else None),
        trace_id=request.state.trace_id,
        merchant_id=merchant_id,
        product_id=payload.product_id,
    )
    return ApiResponse(
        data=CreateTaskData(
            task_id=task.task_id,
            session_id=task.session_id,
            trace_id=task.trace_id,
            merchant_id=task.merchant_id,
            product_id=task.product_id,
            status=task.status,
            created=created,
            events_path=f"/api/v1/tasks/{task.task_id}/events",
        )
    )


@router.get("/tasks/{task_id}", response_model=ApiResponse[TaskStatusData])
async def get_task(
    task_id: str,
    request: Request,
    merchant_id: MerchantId,
) -> ApiResponse[TaskStatusData]:
    container = request.app.state.container
    task = await container.task_service.get_task_for_merchant(task_id, merchant_id)
    return ApiResponse(data=TaskStatusData.from_record(task))


@router.get("/tasks/{task_id}/trace", response_model=ApiResponse[TaskTraceData])
async def get_task_trace(
    task_id: str,
    request: Request,
    merchant_id: MerchantId,
) -> ApiResponse[TaskTraceData]:
    container = request.app.state.container
    task, events = await container.task_service.get_trace(task_id, merchant_id)
    model_calls = sum(event.event_type == "agent.model" for event in events)
    tool_calls = sum(event.event_type == "tool.started" for event in events)
    retries = sum(event.event_type == "task.retrying" for event in events)
    reflection_count = sum(event.event_type == "agent.reflection" for event in events)
    token_events = [event for event in events if event.event_type == "agent.model"]
    input_values = [event.payload.get("input_tokens") for event in token_events]
    output_values = [event.payload.get("output_tokens") for event in token_events]
    input_tokens = (
        sum(int(value) for value in input_values if value is not None)
        if any(value is not None for value in input_values)
        else None
    )
    output_tokens = (
        sum(int(value) for value in output_values if value is not None)
        if any(value is not None for value in output_values)
        else None
    )
    return ApiResponse(
        data=TaskTraceData(
            task_id=task.task_id,
            trace_id=task.trace_id,
            status=task.status,
            model_calls=model_calls,
            tool_calls=tool_calls,
            retries=retries,
            reflection_count=reflection_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            events=[
                TraceEventData(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    sequence_number=event.sequence_number,
                    payload=event.payload,
                )
                for event in events
            ],
        )
    )


@router.post(
    "/tasks/{task_id}/inputs",
    response_model=ApiResponse[SubmitInputData],
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_task_input(
    task_id: str,
    payload: SubmitInputRequest,
    request: Request,
    merchant_id: MerchantId,
) -> ApiResponse[SubmitInputData]:
    container = request.app.state.container
    task = await container.task_service.submit_input(task_id, merchant_id, payload.product_id)
    return ApiResponse(
        data=SubmitInputData(
            task_id=task.task_id,
            status=task.status,
            trace_id=task.trace_id,
            events_path=f"/api/v1/tasks/{task.task_id}/events",
        )
    )


@router.post(
    "/tasks/{task_id}/approvals",
    response_model=ApiResponse[SubmitApprovalData],
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_task_approval(
    task_id: str,
    payload: SubmitApprovalRequest,
    request: Request,
    merchant_id: MerchantId,
) -> ApiResponse[SubmitApprovalData]:
    container = request.app.state.container
    task, approval_id = await container.task_service.submit_approval(
        task_id=task_id,
        merchant_id=merchant_id,
        plan_id=payload.plan_id,
        plan_version=payload.plan_version,
        action_id=payload.action_id,
        decision=payload.decision,
    )
    return ApiResponse(
        data=SubmitApprovalData(
            task_id=task.task_id,
            approval_id=approval_id,
            status=task.status,
            trace_id=task.trace_id,
            events_path=f"/api/v1/tasks/{task.task_id}/events",
        )
    )


@router.post(
    "/tasks/{task_id}/resume",
    response_model=ApiResponse[ResumeTaskData],
    status_code=status.HTTP_202_ACCEPTED,
)
async def resume_task(
    task_id: str,
    request: Request,
    merchant_id: MerchantId,
) -> ApiResponse[ResumeTaskData]:
    container = request.app.state.container
    task = await container.task_service.resume_task(task_id, merchant_id)
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
    merchant_id: MerchantId,
    after: str | None = Query(default=None),
    last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
) -> StreamingResponse:
    container = request.app.state.container
    await container.task_service.get_task_for_merchant(task_id, merchant_id)
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
            if event.event_type in {
                "task.awaiting_input",
                "task.awaiting_approval",
                "task.rejected",
                "task.completed",
                "task.failed",
            }:
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
