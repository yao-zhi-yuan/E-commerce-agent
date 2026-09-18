import asyncio
import logging
import signal

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from redis.asyncio import Redis

from backend_agent.agent.graph import AgentOrchestrator
from backend_agent.commerce import CommerceRepository
from backend_agent.core.config import Settings, get_settings
from backend_agent.core.logging import configure_logging
from backend_agent.llm.client import MockModelClient, ModelClient, OpenAICompatibleModelClient
from backend_agent.queue.rabbitmq import RabbitTaskQueue
from backend_agent.rag.knowledge_base import SQLiteKnowledgeBase
from backend_agent.repositories.redis_store import RedisStore
from backend_agent.services.worker_service import WorkerService
from backend_agent.tools.base import ToolRegistry
from backend_agent.tools.builtin import MetricsTool, RagSearchTool
from backend_agent.tools.mcp import McpToolClient, build_mcp_tools


logger = logging.getLogger(__name__)


def build_model(settings: Settings) -> ModelClient:
    if settings.llm_provider == "mock":
        return MockModelClient()
    if settings.llm_provider != "openai_compatible":
        raise ValueError(f"不支持的 LLM_PROVIDER：{settings.llm_provider}")
    if settings.llm_api_key is None or settings.llm_api_key.get_secret_value() in {"", "<SET_ME>"}:
        raise ValueError("LLM_PROVIDER=openai_compatible 时必须设置 LLM_API_KEY")
    return OpenAICompatibleModelClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.model_timeout_seconds,
        retry_attempts=settings.dependency_retry_attempts,
    )


async def run_worker() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    redis = Redis.from_url(
        settings.redis_url,
        decode_responses=True,
        socket_timeout=settings.redis_socket_timeout_seconds,
        socket_connect_timeout=settings.redis_connect_timeout_seconds,
    )
    store = RedisStore(
        redis,
        task_ttl_seconds=settings.task_ttl_seconds,
        session_ttl_seconds=settings.session_ttl_seconds,
        event_stream_max_length=settings.event_stream_max_length,
        sse_block_milliseconds=settings.sse_block_milliseconds,
    )
    queue = RabbitTaskQueue(
        settings.rabbitmq_url,
        prefetch=settings.worker_prefetch,
        heartbeat=settings.rabbitmq_heartbeat_seconds,
        operation_timeout_seconds=settings.queue_operation_timeout_seconds,
    )
    knowledge_base = SQLiteKnowledgeBase(
        settings.knowledge_db_path,
        settings.knowledge_source_dir,
    )
    commerce = CommerceRepository(settings.commerce_db_path)
    model = build_model(settings)
    mcp_client = McpToolClient(
        settings.mcp_server_url,
        timeout_seconds=settings.tool_timeout_seconds,
        retry_attempts=settings.dependency_retry_attempts,
    )
    await store.ping()
    await knowledge_base.initialize()
    await commerce.initialize()
    await queue.connect()
    settings.checkpoint_db_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.checkpoint_db_path)) as checkpointer:
        await checkpointer.setup()
        tools = ToolRegistry(
            [
                MetricsTool(commerce),
                RagSearchTool(knowledge_base),
                *build_mcp_tools(mcp_client),
            ]
        )
        orchestrator = AgentOrchestrator(
            model=model,
            tools=tools,
            store=store,
            commerce=commerce,
            checkpointer=checkpointer,
            model_timeout_seconds=settings.model_timeout_seconds,
            tool_timeout_seconds=settings.tool_timeout_seconds,
            max_iterations=settings.max_agent_iterations,
            max_repeated_steps=settings.max_repeated_steps,
            max_tool_calls_per_turn=settings.max_tool_calls_per_turn,
            max_parallel_tools=settings.max_parallel_tools,
            max_reflections=settings.max_reflections,
            prompt_version=settings.prompt_version,
            skill_version=settings.skill_version,
        )
        worker = WorkerService(
            store=store,
            queue=queue,
            orchestrator=orchestrator,
            task_timeout_seconds=settings.task_timeout_seconds,
            max_task_attempts=settings.max_task_attempts,
        )
        await queue.consume(worker.handle_message)

        stop_event = asyncio.Event()
        loop = asyncio.get_running_loop()
        for signum in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(signum, stop_event.set)
        logger.info("worker.started")
        try:
            await stop_event.wait()
        finally:
            await queue.close()
            await redis.aclose()
            if isinstance(model, OpenAICompatibleModelClient):
                await model.close()
            logger.info("worker.stopped")


if __name__ == "__main__":
    asyncio.run(run_worker())
