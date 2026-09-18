from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from backend_agent.domain import ApprovalDecision, TaskRecord, TaskStatus


T = TypeVar("T")


class ApiResponse(BaseModel, Generic[T]):
    code: str = "OK"
    message: str = "success"
    data: T


class ErrorDetail(BaseModel):
    code: str
    message: str
    trace_id: str


class ErrorResponse(BaseModel):
    code: str
    message: str
    data: ErrorDetail


class CreateTaskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    product_id: str | None = Field(default=None, min_length=3, max_length=128)
    session_id: str | None = Field(
        default=None,
        min_length=16,
        max_length=128,
        pattern=r"^[A-Za-z0-9_-]+$",
    )


class CreateTaskData(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    merchant_id: str
    product_id: str | None
    status: TaskStatus
    created: bool
    events_path: str


class TaskStatusData(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    merchant_id: str
    product_id: str | None
    status: TaskStatus
    attempt: int
    result: str | None
    interaction: dict[str, object] | None
    error_code: str | None
    error_message: str | None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_record(cls, task: TaskRecord) -> "TaskStatusData":
        return cls(**task.model_dump())


class ResumeTaskData(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    status: TaskStatus
    events_path: str


class SubmitInputRequest(BaseModel):
    product_id: str = Field(min_length=3, max_length=128)


class SubmitInputData(BaseModel):
    task_id: str
    status: TaskStatus
    trace_id: str
    events_path: str


class SubmitApprovalRequest(BaseModel):
    plan_id: str = Field(min_length=8, max_length=128)
    plan_version: int = Field(ge=1)
    action_id: str = Field(min_length=1, max_length=128)
    decision: ApprovalDecision


class SubmitApprovalData(BaseModel):
    task_id: str
    approval_id: str
    status: TaskStatus
    trace_id: str
    events_path: str


class TraceEventData(BaseModel):
    event_id: str
    event_type: str
    sequence_number: int
    payload: dict[str, object]


class TaskTraceData(BaseModel):
    task_id: str
    trace_id: str
    status: TaskStatus
    model_calls: int
    tool_calls: int
    retries: int
    reflection_count: int
    input_tokens: int | None
    output_tokens: int | None
    events: list[TraceEventData]


class HealthData(BaseModel):
    status: str
