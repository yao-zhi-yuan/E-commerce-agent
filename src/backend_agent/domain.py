from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


def utc_now() -> datetime:
    return datetime.now(UTC)


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    RETRYING = "retrying"
    AWAITING_INPUT = "awaiting_input"
    AWAITING_APPROVAL = "awaiting_approval"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"


class ApprovalDecision(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"


class RunStatus(StrEnum):
    COMPLETED = "completed"
    AWAITING_INPUT = "awaiting_input"
    AWAITING_APPROVAL = "awaiting_approval"
    REJECTED = "rejected"


class TaskRecord(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    prompt: str
    merchant_id: str = "demo-merchant"
    product_id: str | None = None
    status: TaskStatus = TaskStatus.QUEUED
    attempt: int = 0
    pending_attempt: int | None = None
    pending_resume_kind: str | None = None
    pending_resume_id: str | None = None
    result: str | None = None
    interaction: dict[str, object] | None = None
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)


class QueueMessage(BaseModel):
    task_id: str
    session_id: str
    trace_id: str
    attempt: int = 0
    resume_kind: str | None = None
    resume_id: str | None = None


class FunnelWindow(BaseModel):
    impressions: int = Field(ge=0)
    clicks: int = Field(ge=0)
    paid_orders: int = Field(ge=0)
    click_through_rate: float | None
    conversion_rate: float | None


class FunnelComparison(BaseModel):
    comparison: FunnelWindow
    current: FunnelWindow
    order_change: int
    order_change_rate: float | None
    traffic_contribution: float | None
    conversion_contribution: float | None


class Finding(BaseModel):
    category: str
    statement: str
    kind: str = Field(pattern=r"^(fact|hypothesis|gap)$")
    evidence_ids: list[str] = Field(default_factory=list)
    confidence: str = Field(default="medium", pattern=r"^(low|medium|high)$")


class ProductPatch(BaseModel):
    field: str = Field(pattern=r"^description$")
    new_value: str = Field(min_length=1, max_length=4_000)
    reason: str = Field(min_length=1, max_length=1_000)


class Diagnosis(BaseModel):
    summary: str = Field(min_length=1, max_length=4_000)
    findings: list[Finding] = Field(min_length=1, max_length=12)
    actions: list[str] = Field(min_length=1, max_length=10)
    proposed_patch: ProductPatch | None = None
    missing_data: list[str] = Field(default_factory=list)


class ActionPlan(BaseModel):
    plan_id: str
    task_id: str
    merchant_id: str
    product_id: str
    plan_version: int
    action_id: str
    field: str
    before_value: str
    after_value: str
    product_version: int
    reason: str
    created_at: datetime


class ActionReceipt(BaseModel):
    receipt_id: str
    idempotency_key: str
    merchant_id: str
    product_id: str
    action_id: str
    product_version: int
    applied: bool
    created_at: datetime


class RunOutcome(BaseModel):
    status: RunStatus
    result: str | None = None
    interaction: dict[str, object] | None = None


class StreamEvent(BaseModel):
    event_id: str
    event_type: str
    sequence_number: int
    payload: dict[str, object]
