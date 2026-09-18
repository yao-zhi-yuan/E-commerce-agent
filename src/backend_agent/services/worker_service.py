import asyncio
import logging
import secrets
from datetime import UTC, datetime

from aio_pika import IncomingMessage

from backend_agent.agent.graph import AgentOrchestrator
from backend_agent.core.logging import bind_trace, trace_span
from backend_agent.domain import QueueMessage, RunStatus, TaskRecord, TaskStatus
from backend_agent.errors import AgentLoopDetectedError, AppError, DependencyError
from backend_agent.queue.rabbitmq import RabbitTaskQueue
from backend_agent.repositories.redis_store import RedisStore


logger = logging.getLogger(__name__)


class WorkerService:
    def __init__(
        self,
        *,
        store: RedisStore,
        queue: RabbitTaskQueue,
        orchestrator: AgentOrchestrator,
        task_timeout_seconds: float,
        max_task_attempts: int,
    ) -> None:
        self._store = store
        self._queue = queue
        self._orchestrator = orchestrator
        self._task_timeout_seconds = task_timeout_seconds
        self._max_task_attempts = max_task_attempts

    async def handle_message(self, message: IncomingMessage) -> None:
        try:
            payload = self._queue.decode(message)
        except Exception:
            logger.exception("queue.invalid_message", extra={"span": "queue.consume"})
            await message.reject(requeue=False)
            return

        bind_trace(
            trace_id=payload.trace_id,
            task_id=payload.task_id,
            session_id=payload.session_id,
        )
        logger.info(
            "task.received",
            extra={"attempt": payload.attempt, "span": "queue.consume"},
        )
        lease_owner = secrets.token_urlsafe(18)
        lease_ttl = max(int(self._task_timeout_seconds) + 30, 60)
        acquired = await self._store.acquire_task_lease(
            payload.task_id,
            lease_owner,
            lease_ttl,
        )
        if not acquired:
            logger.info("task.duplicate_delayed", extra={"attempt": payload.attempt})
            try:
                await self._queue.publish_retry(payload)
            except Exception:
                logger.exception(
                    "task.duplicate_delay_failed",
                    extra={"attempt": payload.attempt, "span": "queue.publish_retry"},
                )
                await message.reject(requeue=True)
                return
            await message.ack()
            return

        graph_started = False
        try:
            task = await self._store.get_task(payload.task_id)
            if task is None:
                logger.error("task.state_missing", extra={"error_code": "TASK_NOT_FOUND"})
                await message.reject(requeue=False)
                return
            if task.status in {TaskStatus.COMPLETED, TaskStatus.REJECTED}:
                await message.ack()
                return
            queue_publish_recovery = (
                task.status is TaskStatus.FAILED and task.error_code == "QUEUE_PUBLISH_FAILED"
            )
            if task.status not in {
                TaskStatus.QUEUED,
                TaskStatus.RUNNING,
                TaskStatus.RETRYING,
            } and not queue_publish_recovery:
                logger.info(
                    "task.waiting_duplicate_ignored",
                    extra={"attempt": payload.attempt, "span": "queue.consume"},
                )
                await message.ack()
                return
            if not self._matches_pending_message(task, payload):
                logger.info(
                    "task.stale_message_ignored",
                    extra={"attempt": payload.attempt, "span": "queue.consume"},
                )
                await message.ack()
                return

            await self._mark_running(task, payload)
            async with trace_span(logger, "task.execute", attempt=payload.attempt):
                async with asyncio.timeout(self._task_timeout_seconds):
                    graph_started = True
                    outcome = await self._orchestrator.run(
                        task,
                        resume_kind=payload.resume_kind,
                        resume_id=payload.resume_id,
                    )
            task.result = outcome.result
            task.interaction = outcome.interaction
            task.error_code = None
            task.error_message = None
            task.pending_attempt = None
            task.pending_resume_kind = None
            task.pending_resume_id = None
            if outcome.status is RunStatus.AWAITING_INPUT:
                task.status = TaskStatus.AWAITING_INPUT
                terminal_event = "task.awaiting_input"
            elif outcome.status is RunStatus.AWAITING_APPROVAL:
                task.status = TaskStatus.AWAITING_APPROVAL
                terminal_event = "task.awaiting_approval"
            elif outcome.status is RunStatus.REJECTED:
                task.status = TaskStatus.REJECTED
                terminal_event = "task.rejected"
            else:
                task.status = TaskStatus.COMPLETED
                terminal_event = "task.completed"
            await self._store.save_task(task)
            if outcome.status is RunStatus.AWAITING_APPROVAL and outcome.interaction:
                await self._store.append_event(
                    task.task_id,
                    "approval.required",
                    {"node_name": "plan", **outcome.interaction},
                )
            await self._store.append_event(
                task.task_id,
                terminal_event,
                {
                    "result": outcome.result,
                    "interaction": outcome.interaction,
                    "attempt": payload.attempt,
                },
            )
            await message.ack()
        except Exception as exc:
            await self._handle_failure(message, payload, exc, graph_started=graph_started)
        finally:
            try:
                await self._store.release_task_lease(payload.task_id, lease_owner)
            except Exception:
                logger.exception("task.lease_release_failed", extra={"span": "task.lease"})

    async def _mark_running(self, task: TaskRecord, payload: QueueMessage) -> None:
        task.status = TaskStatus.RUNNING
        task.attempt = payload.attempt
        queue_wait_ms = round((datetime.now(UTC) - task.updated_at).total_seconds() * 1_000, 2)
        await self._store.save_task(task)
        await self._store.append_event(
            task.task_id,
            "task.started",
            {
                "task_id": task.task_id,
                "attempt": payload.attempt,
                "queue_wait_ms": queue_wait_ms,
            },
        )

    async def _handle_failure(
        self,
        message: IncomingMessage,
        payload: QueueMessage,
        exc: Exception,
        *,
        graph_started: bool,
    ) -> None:
        task = await self._store.get_task(payload.task_id)
        if task is None:
            logger.exception(
                "task.failed_without_state",
                extra={"span": "task.execute", "error_code": "TASK_NOT_FOUND"},
            )
            await message.reject(requeue=False)
            return

        next_attempt = payload.attempt + 1
        error_code, error_message = self._safe_error(exc)
        task.error_code = error_code
        task.error_message = error_message
        task.attempt = payload.attempt

        retryable = self._is_retryable(exc)
        if graph_started and payload.resume_kind not in {"input", "approval"}:
            retry_kind = "retry"
            retry_id = payload.task_id
        else:
            retry_kind = payload.resume_kind
            retry_id = payload.resume_id
        task.pending_attempt = next_attempt
        task.pending_resume_kind = retry_kind
        task.pending_resume_id = retry_id

        if retryable and next_attempt < self._max_task_attempts:
            task.status = TaskStatus.RETRYING
            await self._store.save_task(task)
            retry_payload = QueueMessage(
                task_id=payload.task_id,
                session_id=payload.session_id,
                trace_id=payload.trace_id,
                attempt=next_attempt,
                resume_kind=retry_kind,
                resume_id=retry_id,
            )
            try:
                await self._queue.publish_retry(retry_payload)
            except Exception:
                logger.exception(
                    "task.retry_publish_failed",
                    extra={"attempt": next_attempt, "span": "queue.publish_retry"},
                )
                task.status = TaskStatus.FAILED
                task.error_code = "QUEUE_PUBLISH_FAILED"
                task.error_message = "重试任务入队失败，可通过 resume 接口继续"
                task.interaction = self._resume_interaction(retry_payload)
                await self._store.save_task(task)
                await self._store.append_event(
                    task.task_id,
                    "task.failed",
                    {
                        "task_id": task.task_id,
                        "attempt": payload.attempt,
                        "error_code": task.error_code,
                        "message": task.error_message,
                    },
                )
                await message.reject(requeue=False)
                return
            await self._store.append_event(
                task.task_id,
                "task.retrying",
                {
                    "task_id": task.task_id,
                    "attempt": payload.attempt,
                    "next_attempt": next_attempt,
                    "error_code": error_code,
                    "message": error_message,
                },
            )
            await message.ack()
            return

        task.status = TaskStatus.FAILED
        await self._store.save_task(task)
        await self._store.append_event(
            task.task_id,
            "task.failed",
            {
                "task_id": task.task_id,
                "attempt": payload.attempt,
                "error_code": error_code,
                "message": error_message,
            },
        )
        logger.error(
            "task.permanently_failed",
            extra={"attempt": payload.attempt, "error_code": error_code},
        )
        await message.reject(requeue=False)

    @staticmethod
    def _matches_pending_message(task: TaskRecord, payload: QueueMessage) -> bool:
        if task.pending_attempt is None:
            return True
        return (
            payload.attempt == task.pending_attempt
            and payload.resume_kind == task.pending_resume_kind
            and payload.resume_id == task.pending_resume_id
        )

    @staticmethod
    def _resume_interaction(payload: QueueMessage) -> dict[str, object] | None:
        if payload.resume_kind == "input" and payload.resume_id:
            return {"type": "input.resume", "product_id": payload.resume_id}
        if payload.resume_kind == "approval" and payload.resume_id:
            return {"type": "approval.resume", "approval_id": payload.resume_id}
        if payload.resume_kind == "retry" and payload.resume_id:
            return {"type": "retry.resume", "resume_id": payload.resume_id}
        return None

    @staticmethod
    def _safe_error(exc: Exception) -> tuple[str, str]:
        if isinstance(exc, AppError):
            return exc.code, exc.message
        if isinstance(exc, TimeoutError):
            return "TASK_TIMEOUT", "任务执行超过总超时时间"
        logger.exception("task.execution_failed", extra={"span": "task.execute"})
        return "TASK_EXECUTION_FAILED", "任务执行失败，请根据 trace_id 查看日志"

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        if isinstance(exc, AgentLoopDetectedError | ValueError):
            return False
        if isinstance(exc, DependencyError):
            return exc.code not in {
                "MODEL_REQUEST_REJECTED",
                "MODEL_EMPTY_RESPONSE",
                "MODEL_INVALID_TOOL_ARGUMENTS",
            }
        if isinstance(exc, AppError):
            return False
        return True
