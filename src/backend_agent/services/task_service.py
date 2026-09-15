import secrets

from backend_agent.core.logging import new_trace_id
from backend_agent.domain import QueueMessage, TaskRecord, TaskStatus
from backend_agent.errors import DependencyError, TaskNotFoundError, TaskNotResumableError
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
    ) -> None:
        self._store = store
        self._queue = queue
        self._knowledge_base = knowledge_base

    async def create_task(
        self,
        *,
        prompt: str,
        session_id: str | None,
        idempotency_key: str | None,
        trace_id: str,
    ) -> tuple[TaskRecord, bool]:
        task = TaskRecord(
            task_id=secrets.token_urlsafe(18),
            session_id=session_id or secrets.token_urlsafe(24),
            trace_id=trace_id or new_trace_id(),
            prompt=prompt,
        )
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

    async def resume_task(self, task_id: str) -> TaskRecord:
        task = await self.get_task(task_id)
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
        task.status = TaskStatus.QUEUED
        task.result = None
        task.error_code = None
        task.error_message = None
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
                    attempt=0,
                )
            )
        except Exception as exc:
            task.status = TaskStatus.FAILED
            task.error_code = "QUEUE_PUBLISH_FAILED"
            task.error_message = "续跑任务入队失败"
            await self._store.save_task(task)
            raise DependencyError("QUEUE_PUBLISH_FAILED", task.error_message) from exc
        return task

    async def add_knowledge_document(
        self,
        *,
        title: str,
        content: str,
        metadata: dict[str, str],
    ) -> str:
        return await self._knowledge_base.add_document(
            title=title,
            content=content,
            metadata=metadata,
        )
