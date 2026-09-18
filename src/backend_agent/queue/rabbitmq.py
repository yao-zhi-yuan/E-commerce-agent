import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import aio_pika
from aio_pika import DeliveryMode, ExchangeType, IncomingMessage, Message
from aio_pika.abc import (
    AbstractChannel,
    AbstractRobustConnection,
    AbstractRobustExchange,
    AbstractRobustQueue,
)

from backend_agent.domain import QueueMessage


logger = logging.getLogger(__name__)


class RabbitTaskQueue:
    exchange_name = "e-commerce-agent.tasks.v1"
    routing_key = "agent.run"
    queue_name = "e-commerce-agent.tasks.v1"
    retry_exchange_name = "e-commerce-agent.retry.v1"
    dead_exchange_name = "e-commerce-agent.dead.v1"
    dead_queue_name = "e-commerce-agent.dead.v1"
    retry_delays_ms = (1_000, 5_000, 20_000)

    def __init__(
        self,
        url: str,
        *,
        prefetch: int,
        heartbeat: int,
        operation_timeout_seconds: float,
    ) -> None:
        self._url = url
        self._prefetch = prefetch
        self._heartbeat = heartbeat
        self._operation_timeout_seconds = operation_timeout_seconds
        self._connection: AbstractRobustConnection | None = None
        self._channel: AbstractChannel | None = None
        self._task_exchange: AbstractRobustExchange | None = None
        self._retry_exchange: AbstractRobustExchange | None = None
        self._queue: AbstractRobustQueue | None = None

    async def connect(self) -> None:
        async with asyncio.timeout(self._operation_timeout_seconds * 3):
            self._connection = await aio_pika.connect_robust(
                self._url,
                timeout=self._operation_timeout_seconds,
                heartbeat=self._heartbeat,
            )
            self._channel = await self._connection.channel(publisher_confirms=True)
            await self._channel.set_qos(prefetch_count=self._prefetch)
            self._task_exchange = await self._channel.declare_exchange(
                self.exchange_name,
                ExchangeType.DIRECT,
                durable=True,
            )
            self._retry_exchange = await self._channel.declare_exchange(
                self.retry_exchange_name,
                ExchangeType.DIRECT,
                durable=True,
            )
            dead_exchange = await self._channel.declare_exchange(
                self.dead_exchange_name,
                ExchangeType.DIRECT,
                durable=True,
            )
            self._queue = await self._channel.declare_queue(
                self.queue_name,
                durable=True,
                arguments={
                    "x-dead-letter-exchange": self.dead_exchange_name,
                    "x-dead-letter-routing-key": "agent.dead",
                },
            )
            await self._queue.bind(self._task_exchange, routing_key=self.routing_key)

            dead_queue = await self._channel.declare_queue(self.dead_queue_name, durable=True)
            await dead_queue.bind(dead_exchange, routing_key="agent.dead")

            for index, delay_ms in enumerate(self.retry_delays_ms):
                retry_key = self._retry_routing_key(index)
                retry_queue = await self._channel.declare_queue(
                    f"{self.queue_name}.retry.{index + 1}",
                    durable=True,
                    arguments={
                        "x-message-ttl": delay_ms,
                        "x-dead-letter-exchange": self.exchange_name,
                        "x-dead-letter-routing-key": self.routing_key,
                    },
                )
                await retry_queue.bind(self._retry_exchange, routing_key=retry_key)

    async def close(self) -> None:
        if self._connection is not None:
            await self._connection.close()

    async def publish(self, payload: QueueMessage) -> None:
        exchange = self._require_task_exchange()
        async with asyncio.timeout(self._operation_timeout_seconds):
            await exchange.publish(self._message(payload), routing_key=self.routing_key, mandatory=True)
        logger.info("task.published", extra={"attempt": payload.attempt})

    async def publish_retry(self, payload: QueueMessage) -> None:
        exchange = self._require_retry_exchange()
        retry_index = min(max(payload.attempt - 1, 0), len(self.retry_delays_ms) - 1)
        async with asyncio.timeout(self._operation_timeout_seconds):
            await exchange.publish(
                self._message(payload),
                routing_key=self._retry_routing_key(retry_index),
                mandatory=True,
            )
        logger.info("task.retry_published", extra={"attempt": payload.attempt})

    async def consume(self, callback: Callable[[IncomingMessage], Awaitable[None]]) -> None:
        if self._queue is None:
            raise RuntimeError("RabbitMQ queue is not connected")
        async with asyncio.timeout(self._operation_timeout_seconds):
            await self._queue.consume(callback, no_ack=False)

    @staticmethod
    def decode(message: IncomingMessage) -> QueueMessage:
        return QueueMessage.model_validate_json(message.body)

    @classmethod
    def _message(cls, payload: QueueMessage) -> Message:
        return Message(
            body=payload.model_dump_json().encode("utf-8"),
            content_type="application/json",
            delivery_mode=DeliveryMode.PERSISTENT,
            correlation_id=payload.task_id,
            headers={
                "attempt": payload.attempt,
                "trace_id": payload.trace_id,
                "resume_kind": payload.resume_kind or "",
            },
        )

    @staticmethod
    def _retry_routing_key(index: int) -> str:
        return f"agent.retry.{index + 1}"

    def _require_task_exchange(self) -> AbstractRobustExchange:
        if self._task_exchange is None:
            raise RuntimeError("RabbitMQ task exchange is not connected")
        return self._task_exchange

    def _require_retry_exchange(self) -> AbstractRobustExchange:
        if self._retry_exchange is None:
            raise RuntimeError("RabbitMQ retry exchange is not connected")
        return self._retry_exchange


def message_debug_view(message: IncomingMessage) -> str:
    return json.dumps(
        {
            "correlation_id": message.correlation_id,
            "redelivered": message.redelivered,
            "routing_key": message.routing_key,
        },
        ensure_ascii=False,
    )
