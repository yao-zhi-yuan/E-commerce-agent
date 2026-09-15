import hashlib
import json
from collections.abc import AsyncIterator

from redis.asyncio import Redis

from backend_agent.domain import AgentCheckpoint, StreamEvent, TaskRecord, utc_now


class RedisStore:
    _prefix = "backend-agent:v1"

    def __init__(
        self,
        redis: Redis,
        *,
        task_ttl_seconds: int,
        session_ttl_seconds: int,
        event_stream_max_length: int,
        sse_block_milliseconds: int,
    ) -> None:
        self._redis = redis
        self._task_ttl = task_ttl_seconds
        self._session_ttl = session_ttl_seconds
        self._event_stream_max_length = event_stream_max_length
        self._sse_block_milliseconds = sse_block_milliseconds

    @classmethod
    def task_key(cls, task_id: str) -> str:
        return f"{cls._prefix}:task:{task_id}"

    @classmethod
    def idempotency_key(cls, key: str) -> str:
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
        return f"{cls._prefix}:idempotency:{digest}"

    @classmethod
    def checkpoint_key(cls, task_id: str) -> str:
        return f"{cls._prefix}:task:{task_id}:checkpoint"

    @classmethod
    def conversation_key(cls, session_id: str) -> str:
        return f"{cls._prefix}:session:{session_id}:conversation"

    @classmethod
    def event_key(cls, task_id: str) -> str:
        return f"{cls._prefix}:task:{task_id}:events"

    @classmethod
    def sequence_key(cls, task_id: str) -> str:
        return f"{cls._prefix}:task:{task_id}:sequence"

    @classmethod
    def lease_key(cls, task_id: str) -> str:
        return f"{cls._prefix}:task:{task_id}:lease"

    async def ping(self) -> None:
        await self._redis.ping()

    async def create_task(
        self,
        task: TaskRecord,
        idempotency_key: str | None,
    ) -> tuple[TaskRecord, bool]:
        task_json = task.model_dump_json()
        if not idempotency_key:
            created = await self._redis.set(
                self.task_key(task.task_id),
                task_json,
                ex=self._task_ttl,
                nx=True,
            )
            return task, bool(created)

        script = """
        local existing = redis.call('GET', KEYS[2])
        if existing then
            return {existing, 0}
        end
        local created = redis.call('SET', KEYS[1], ARGV[1], 'EX', ARGV[2], 'NX')
        if not created then
            return {'', 0}
        end
        redis.call('SET', KEYS[2], ARGV[3], 'EX', ARGV[2])
        return {ARGV[3], 1}
        """
        result = await self._redis.eval(
            script,
            2,
            self.task_key(task.task_id),
            self.idempotency_key(idempotency_key),
            task_json,
            self._task_ttl,
            task.task_id,
        )
        resolved_task_id = str(result[0])
        created = bool(int(result[1]))
        if created:
            return task, True
        existing = await self.get_task(resolved_task_id)
        if existing is None:
            await self._redis.delete(self.idempotency_key(idempotency_key))
            return await self.create_task(task, idempotency_key)
        return existing, False

    async def get_task(self, task_id: str) -> TaskRecord | None:
        raw = await self._redis.get(self.task_key(task_id))
        return TaskRecord.model_validate_json(raw) if raw else None

    async def save_task(self, task: TaskRecord) -> None:
        task.updated_at = utc_now()
        await self._redis.set(
            self.task_key(task.task_id),
            task.model_dump_json(),
            ex=self._task_ttl,
        )

    async def save_checkpoint(self, checkpoint: AgentCheckpoint) -> None:
        checkpoint.updated_at = utc_now()
        await self._redis.set(
            self.checkpoint_key(checkpoint.task_id),
            checkpoint.model_dump_json(),
            ex=self._task_ttl,
        )

    async def get_checkpoint(self, task_id: str) -> AgentCheckpoint | None:
        raw = await self._redis.get(self.checkpoint_key(task_id))
        return AgentCheckpoint.model_validate_json(raw) if raw else None

    async def save_conversation(
        self,
        session_id: str,
        messages: list[dict[str, object]],
    ) -> None:
        await self._redis.set(
            self.conversation_key(session_id),
            json.dumps(messages, ensure_ascii=False, default=str),
            ex=self._session_ttl,
        )

    async def get_conversation(self, session_id: str) -> list[dict[str, object]]:
        raw = await self._redis.get(self.conversation_key(session_id))
        if not raw:
            return []
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, dict)]

    async def acquire_task_lease(self, task_id: str, owner: str, ttl_seconds: int) -> bool:
        acquired = await self._redis.set(
            self.lease_key(task_id),
            owner,
            ex=ttl_seconds,
            nx=True,
        )
        return bool(acquired)

    async def release_task_lease(self, task_id: str, owner: str) -> None:
        script = """
        if redis.call('GET', KEYS[1]) == ARGV[1] then
            return redis.call('DEL', KEYS[1])
        end
        return 0
        """
        await self._redis.eval(script, 1, self.lease_key(task_id), owner)

    async def has_task_lease(self, task_id: str) -> bool:
        return bool(await self._redis.exists(self.lease_key(task_id)))

    async def append_event(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> StreamEvent:
        sequence = await self._redis.incr(self.sequence_key(task_id))
        data = {
            "event_type": event_type,
            "sequence_number": str(sequence),
            "payload": json.dumps(payload, ensure_ascii=False, default=str),
        }
        event_id = await self._redis.xadd(
            self.event_key(task_id),
            data,
            maxlen=self._event_stream_max_length,
            approximate=True,
        )
        async with self._redis.pipeline(transaction=False) as pipeline:
            pipeline.expire(self.event_key(task_id), self._task_ttl)
            pipeline.expire(self.sequence_key(task_id), self._task_ttl)
            await pipeline.execute()
        return StreamEvent(
            event_id=str(event_id),
            event_type=event_type,
            sequence_number=int(sequence),
            payload=payload,
        )

    async def iter_events(
        self,
        task_id: str,
        after_id: str,
    ) -> AsyncIterator[StreamEvent | None]:
        stream_key = self.event_key(task_id)
        current_id = after_id
        backlog = await self._redis.xrange(stream_key, min=f"({current_id}", max="+")
        for event_id, fields in backlog:
            current_id = str(event_id)
            yield self._decode_event(current_id, fields)

        while True:
            batches = await self._redis.xread(
                {stream_key: current_id},
                count=100,
                block=self._sse_block_milliseconds,
            )
            if not batches:
                yield None
                continue
            for _, messages in batches:
                for event_id, fields in messages:
                    current_id = str(event_id)
                    yield self._decode_event(current_id, fields)

    @staticmethod
    def _decode_event(event_id: str, fields: dict[str, str]) -> StreamEvent:
        payload = json.loads(fields["payload"])
        if not isinstance(payload, dict):
            payload = {"value": payload}
        return StreamEvent(
            event_id=event_id,
            event_type=fields["event_type"],
            sequence_number=int(fields["sequence_number"]),
            payload=payload,
        )
