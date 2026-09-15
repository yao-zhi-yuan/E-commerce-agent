import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar


T = TypeVar("T")


async def retry_async(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    retryable: tuple[type[Exception], ...],
    base_delay_seconds: float = 0.2,
    max_delay_seconds: float = 2.0,
) -> T:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation()
        except retryable as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                break
            base_delay = min(base_delay_seconds * (2**attempt), max_delay_seconds)
            jitter = random.uniform(0, base_delay * 0.2)
            await asyncio.sleep(base_delay + jitter)
    if last_error is None:
        raise RuntimeError("retry_async requires at least one attempt")
    raise last_error

