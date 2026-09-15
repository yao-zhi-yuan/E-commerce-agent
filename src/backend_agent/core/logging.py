import contextvars
import json
import logging
import secrets
import sys
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime


_trace_id: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")
_task_id: contextvars.ContextVar[str] = contextvars.ContextVar("task_id", default="")
_session_id: contextvars.ContextVar[str] = contextvars.ContextVar("session_id", default="")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        trace_id = _trace_id.get()
        task_id = _task_id.get()
        session_id = _session_id.get()
        if trace_id:
            payload["trace_id"] = trace_id
        if task_id:
            payload["task_id"] = task_id
        if session_id:
            payload["session_id"] = session_id

        for key in ("span", "duration_ms", "attempt", "tool_name", "event_type", "error_code"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())


def new_trace_id() -> str:
    return secrets.token_urlsafe(18)


def bind_trace(*, trace_id: str, task_id: str = "", session_id: str = "") -> None:
    _trace_id.set(trace_id)
    _task_id.set(task_id)
    _session_id.set(session_id)


@asynccontextmanager
async def trace_span(
    logger: logging.Logger,
    span: str,
    **fields: object,
) -> AsyncIterator[None]:
    started = time.perf_counter()
    logger.info("span.started", extra={"span": span, **fields})
    try:
        yield
    except Exception:
        duration_ms = round((time.perf_counter() - started) * 1_000, 2)
        logger.exception(
            "span.failed",
            extra={"span": span, "duration_ms": duration_ms, **fields},
        )
        raise
    else:
        duration_ms = round((time.perf_counter() - started) * 1_000, 2)
        logger.info(
            "span.completed",
            extra={"span": span, "duration_ms": duration_ms, **fields},
        )

