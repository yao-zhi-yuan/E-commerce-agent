import asyncio
import logging
import secrets

from aio_pika import IncomingMessage

from backend_agent.agent.graph import AgentOrchestrator
from backend_agent.core.logging import bind_trace, trace_span
from backend_agent.domain import AgentCheckpoint, QueueMessage, TaskRecord, TaskStatus
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

        try:
            task = await self._store.get_task(payload.task_id)
            if task is None:
                logger.error("task.state_missing", extra={"error_code": "TASK_NOT_FOUND"})
                await message.reject(requeue=False)
                return
            if task.status is TaskStatus.COMPLETED:
                await message.ack()
                return

            await self._mark_running(task, payload)
            async with trace_span(logger, "task.execute", attempt=payload.attempt):
                async with asyncio.timeout(self._task_timeout_seconds):
                    result = await self._orchestrator.run(task)
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.error_code = None
            task.error_message = None
            await self._store.save_task(task)
            await self._store.append_event(
                task.task_id,
                "task.completed",
                {"task_id": task.task_id, "result": result, "attempt": payload.attempt},
            )
            await message.ack()
        except Exception as exc:
            await self._handle_failure(message, payload, exc)
        finally:
            try:
                await self._store.release_task_lease(payload.task_id, lease_owner)
            except Exception:
                logger.exception("task.lease_release_failed", extra={"span": "task.lease"})

    async def _mark_running(self, task: TaskRecord, payload: QueueMessage) -> None:
        task.status = TaskStatus.RUNNING
        task.attempt = payload.attempt
        await self._store.save_task(task)
        await self._store.append_event(
            task.task_id,
            "task.started",
            {"task_id": task.task_id, "attempt": payload.attempt},
        )

    async def _handle_failure(
        self,
        message: IncomingMessage,
        payload: QueueMessage,
        exc: Exception,
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
        await self._mark_checkpoint_failed(task, error_message)

        if self._is_retryable(exc) and next_attempt < self._max_task_attempts:
            task.status = TaskStatus.RETRYING
            await self._store.save_task(task)
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
            retry_payload = QueueMessage(
                task_id=payload.task_id,
                session_id=payload.session_id,
                trace_id=payload.trace_id,
                attempt=next_attempt,
            )
            try:
                await self._queue.publish_retry(retry_payload)
            except Exception:
                logger.exception(
                    "task.retry_publish_failed",
                    extra={"attempt": next_attempt, "span": "queue.publish_retry"},
                )
                await message.reject(requeue=True)
                return
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

    async def _mark_checkpoint_failed(self, task: TaskRecord, message: str) -> None:
        checkpoint = await self._store.get_checkpoint(task.task_id)
        if checkpoint is None:
            return
        await self._store.save_checkpoint(
            AgentCheckpoint(
                task_id=checkpoint.task_id,
                session_id=checkpoint.session_id,
                messages=checkpoint.messages,
                iteration=checkpoint.iteration,
                repeated_steps=checkpoint.repeated_steps,
                last_signature=checkpoint.last_signature,
                status="failed",
                final_output=message,
            )
        )

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
        return True
