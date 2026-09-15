from datetime import datetime
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

from backend_agent.domain import TaskRecord, TaskStatus


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
    status: TaskStatus
    created: bool
    events_path: str


class TaskStatusData(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    status: TaskStatus
    attempt: int
    result: str | None
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


class AddKnowledgeDocumentRequest(BaseModel):
    title: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=200_000)
    metadata: dict[str, str] = Field(default_factory=dict)


class AddKnowledgeDocumentData(BaseModel):
    document_id: str


class HealthData(BaseModel):
    status: str
