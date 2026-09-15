from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskRecord(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    prompt: str
    status: TaskStatus = TaskStatus.QUEUED
    attempt: int = 0
    result: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class QueueMessage(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    attempt: int = 0


class AgentCheckpoint(BaseModel):
    task_id: str
    session_id: str
    messages: list[dict[str, object]] = Field(default_factory=list)
    iteration: int = 0
    repeated_steps: int = 0
    last_signature: str = ""
    status: str = "running"
    final_output: str | None = None
    updated_at: datetime = Field(default_factory=utc_now)


class StreamEvent(BaseModel):
    event_id: str
    event_type: str
    sequence_number: int
    payload: dict[str, object]

