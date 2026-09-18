import secrets

from backend_agent.commerce import CommerceRepository
from backend_agent.core.logging import bind_trace, new_trace_id
from backend_agent.domain import ApprovalDecision, QueueMessage, StreamEvent, TaskRecord, TaskStatus
from backend_agent.errors import (
    DependencyError,
    MerchantForbiddenError,
    TaskNotFoundError,
    TaskNotResumableError,
    TaskStateConflictError,
)
from backend_agent.queue.rabbitmq import RabbitTaskQueue
from backend_agent.rag.knowledge_base import SQLiteKnowledgeBase
from backend_agent.repositories.redis_store import RedisStore


class TaskService:
    def __init__(
        self,
        *,
        store: RedisStore,
        queue: RabbitTaskQueue,
        knowledge_base: SQLiteKnowledgeBase,
        commerce: CommerceRepository,
    ) -> None:
        self._store = store
        self._queue = queue
        self._knowledge_base = knowledge_base
        self._commerce = commerce

    async def create_task(
        self,
        *,
        prompt: str,
        session_id: str | None,
        idempotency_key: str | None,
        trace_id: str,
        merchant_id: str,
        product_id: str | None,
    ) -> tuple[TaskRecord, bool]:
        task = TaskRecord(
            task_id=secrets.token_urlsafe(18),
            session_id=session_id or secrets.token_urlsafe(24),
            trace_id=trace_id or new_trace_id(),
            prompt=prompt,
            merchant_id=merchant_id,
            product_id=product_id,
            pending_attempt=0,
        )
        bind_trace(trace_id=task.trace_id, task_id=task.task_id, session_id=task.session_id)
        resolved_task, created = await self._store.create_task(task, idempotency_key)
        if not created:
            return resolved_task, False

        await self._store.append_event(
            task.task_id,
            "task.queued",
            {"task_id": task.task_id, "session_id": task.session_id, "attempt": 0},
        )
        try:
            await self._queue.publish(
                QueueMessage(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    trace_id=task.trace_id,
                    attempt=0,
                )
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error_code = "QUEUE_PUBLISH_FAILED"
            task.error_message = "任务入队失败，可通过 resume 接口重试"
            await self._store.save_task(task)
            await self._store.append_event(
                task.task_id,
                "task.failed",
                {
                    "task_id": task.task_id,
                    "error_code": task.error_code,
                    "message": task.error_message,
                },
            )
            raise DependencyError("QUEUE_PUBLISH_FAILED", task.error_message) from exc
        return task, True

    async def get_task(self, task_id: str) -> TaskRecord:
        task = await self._store.get_task(task_id)
        if task is None:
            raise TaskNotFoundError(task_id)
        return task

    async def get_task_for_merchant(self, task_id: str, merchant_id: str) -> TaskRecord:
        task = await self.get_task(task_id)
        if task.merchant_id != merchant_id:
            raise MerchantForbiddenError()
        return task

    async def submit_input(self, task_id: str, merchant_id: str, product_id: str) -> TaskRecord:
        task = await self.get_task_for_merchant(task_id, merchant_id)
        bind_trace(trace_id=task.trace_id, task_id=task.task_id, session_id=task.session_id)
        if task.status is not TaskStatus.AWAITING_INPUT:
            raise TaskStateConflictError("任务当前不等待商品输入")
        matches = await self._commerce.resolve_product(task.merchant_id, product_id)
        if not any(item["product_id"] == product_id for item in matches):
            raise TaskStateConflictError("所选商品不属于当前商家")
        task.product_id = product_id
        task.status = TaskStatus.QUEUED
        task.interaction = None
        task.pending_attempt = 0
        task.pending_resume_kind = "input"
        task.pending_resume_id = product_id
        transitioned = await self._store.transition_task(
            task,
            expected_status=TaskStatus.AWAITING_INPUT.value,
        )
        if not transitioned:
            current = await self.get_task_for_merchant(task_id, merchant_id)
            if current.product_id == product_id and current.status in {
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.AWAITING_APPROVAL,
                TaskStatus.COMPLETED,
                TaskStatus.REJECTED,
            }:
                return current
            raise TaskStateConflictError("任务输入已被其他请求提交")
        await self._store.append_event(
            task.task_id,
            "input.submitted",
            {"product_id": product_id},
        )
        try:
            await self._queue.publish(
                QueueMessage(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    trace_id=task.trace_id,
                    resume_kind="input",
                    resume_id=product_id,
                )
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error_code = "QUEUE_PUBLISH_FAILED"
            task.error_message = "商品输入已保存，但恢复任务入队失败"
            task.interaction = {"type": "input.resume", "product_id": product_id}
            await self._store.save_task(task)
            raise DependencyError("QUEUE_PUBLISH_FAILED", task.error_message) from exc
        return task

    async def submit_approval(
        self,
        *,
        task_id: str,
        merchant_id: str,
        plan_id: str,
        plan_version: int,
        action_id: str,
        decision: ApprovalDecision,
    ) -> tuple[TaskRecord, str]:
        task = await self.get_task_for_merchant(task_id, merchant_id)
        bind_trace(trace_id=task.trace_id, task_id=task.task_id, session_id=task.session_id)
        original_error_code = task.error_code
        if task.status in {
            TaskStatus.QUEUED,
            TaskStatus.RUNNING,
            TaskStatus.COMPLETED,
            TaskStatus.REJECTED,
        }:
            existing = await self._commerce.get_approval_for_task(task.task_id)
            if (
                existing is not None
                and existing["merchant_id"] == task.merchant_id
                and existing["plan_id"] == plan_id
                and int(existing["plan_version"]) == plan_version
                and existing["action_id"] == action_id
                and existing["decision"] == decision.value
            ):
                return task, str(existing["approval_id"])
            raise TaskStateConflictError("任务尚未等待审批或审批请求与已有记录不一致")
        if task.status is TaskStatus.FAILED and task.error_code != "QUEUE_PUBLISH_FAILED":
            raise TaskStateConflictError("任务执行失败，不能通过重复审批重新执行")
        if task.status not in {TaskStatus.AWAITING_APPROVAL, TaskStatus.FAILED}:
            raise TaskStateConflictError("任务当前不等待审批")
        approval_id = await self._commerce.record_approval(
            task_id=task.task_id,
            merchant_id=task.merchant_id,
            plan_id=plan_id,
            plan_version=plan_version,
            action_id=action_id,
            decision=decision,
        )
        task.status = TaskStatus.QUEUED
        task.interaction = None
        task.error_code = None
        task.error_message = None
        task.pending_attempt = 0
        task.pending_resume_kind = "approval"
        task.pending_resume_id = approval_id
        expected_status = (
            TaskStatus.FAILED.value
            if original_error_code == "QUEUE_PUBLISH_FAILED"
            else TaskStatus.AWAITING_APPROVAL.value
        )
        transitioned = await self._store.transition_task(task, expected_status=expected_status)
        if not transitioned:
            current = await self.get_task_for_merchant(task_id, merchant_id)
            if current.status in {
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.COMPLETED,
                TaskStatus.REJECTED,
            }:
                return current, approval_id
            raise TaskStateConflictError("审批恢复已被其他请求提交")
        await self._store.append_event(
            task.task_id,
            "approval.submitted",
            {
                "plan_id": plan_id,
                "action_id": action_id,
                "decision": decision.value,
            },
        )
        try:
            await self._queue.publish(
                QueueMessage(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    trace_id=task.trace_id,
                    resume_kind="approval",
                    resume_id=approval_id,
                )
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error_code = "QUEUE_PUBLISH_FAILED"
            task.error_message = "审批已保存，但恢复任务入队失败"
            task.interaction = {"type": "approval.resume", "approval_id": approval_id}
            await self._store.save_task(task)
            raise DependencyError("QUEUE_PUBLISH_FAILED", task.error_message) from exc
        if original_error_code == "QUEUE_PUBLISH_FAILED":
            await self._store.append_event(
                task.task_id,
                "approval.requeued",
                {"approval_id": approval_id},
            )
        return task, approval_id

    async def get_trace(
        self, task_id: str, merchant_id: str
    ) -> tuple[TaskRecord, list[StreamEvent]]:
        task = await self.get_task_for_merchant(task_id, merchant_id)
        return task, await self._store.list_events(task_id)

    async def resume_task(self, task_id: str, merchant_id: str) -> TaskRecord:
        task = await self.get_task_for_merchant(task_id, merchant_id)
        bind_trace(trace_id=task.trace_id, task_id=task.task_id, session_id=task.session_id)
        original_error_code = task.error_code
        original_interaction = task.interaction
        original_pending_attempt = task.pending_attempt
        if task.error_code in {
            "AGENT_LOOP_DETECTED",
            "MODEL_REQUEST_REJECTED",
            "MODEL_EMPTY_RESPONSE",
            "MODEL_INVALID_TOOL_ARGUMENTS",
        }:
            raise TaskNotResumableError(task_id)
        stale_running = task.status is TaskStatus.RUNNING and not await self._store.has_task_lease(
            task.task_id
        )
        if task.status is not TaskStatus.FAILED and not stale_running:
            raise TaskNotResumableError(task_id)
        resume_kind = task.pending_resume_kind
        resume_id = task.pending_resume_id
        if original_error_code == "QUEUE_PUBLISH_FAILED" and resume_kind is None:
            if isinstance(original_interaction, dict) and original_interaction.get("type") == "input.resume":
                resume_kind = "input"
                resume_id = str(original_interaction["product_id"])
            elif (
                isinstance(original_interaction, dict)
                and original_interaction.get("type") == "approval.resume"
            ):
                resume_kind = "approval"
                resume_id = str(original_interaction["approval_id"])
            elif (
                isinstance(original_interaction, dict)
                and original_interaction.get("type") == "retry.resume"
            ):
                resume_kind = "retry"
                resume_id = str(original_interaction["resume_id"])
            else:
                resume_kind = None
                resume_id = None
        elif resume_kind is None:
            resume_kind = "retry"
            resume_id = task.task_id
        task.status = TaskStatus.QUEUED
        task.result = None
        task.error_code = None
        task.error_message = None
        task.interaction = None
        resume_attempt = (
            original_pending_attempt
            if original_error_code == "QUEUE_PUBLISH_FAILED"
            and original_pending_attempt is not None
            else 0
        )
        task.pending_attempt = resume_attempt
        task.pending_resume_kind = resume_kind
        task.pending_resume_id = resume_id
        await self._store.save_task(task)
        await self._store.append_event(
            task.task_id,
            "task.resumed",
            {"task_id": task.task_id, "session_id": task.session_id},
        )
        try:
            await self._queue.publish(
                QueueMessage(
                    task_id=task.task_id,
                    session_id=task.session_id,
                    trace_id=task.trace_id,
                    attempt=resume_attempt,
                    resume_kind=resume_kind,
                    resume_id=resume_id,
                )
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error_code = "QUEUE_PUBLISH_FAILED"
            task.error_message = "续跑任务入队失败"
            if resume_kind == "input" and resume_id:
                task.interaction = {"type": "input.resume", "product_id": resume_id}
            elif resume_kind == "approval" and resume_id:
                task.interaction = {"type": "approval.resume", "approval_id": resume_id}
            elif resume_kind == "retry" and resume_id:
                task.interaction = {"type": "retry.resume", "resume_id": resume_id}
            else:
                task.interaction = original_interaction
            await self._store.save_task(task)
            raise DependencyError("QUEUE_PUBLISH_FAILED", task.error_message) from exc
        return task
