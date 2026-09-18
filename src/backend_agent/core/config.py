from functools import lru_cache
from pathlib import Path
from urllib.parse import quote_plus

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "development"
    log_level: str = "INFO"
    demo_api_token: SecretStr = Field(default=SecretStr(""))

    redis_url: str = "redis://localhost:6379/0"
    redis_connect_timeout_seconds: float = Field(default=3.0, gt=0)
    redis_socket_timeout_seconds: float = Field(default=10.0, gt=0)
    session_ttl_seconds: int = Field(default=24 * 60 * 60, ge=60)
    task_ttl_seconds: int = Field(default=24 * 60 * 60, ge=60)
    event_stream_max_length: int = Field(default=2_000, ge=100)
    sse_block_milliseconds: int = Field(default=5_000, ge=1_000, le=60_000)

    rabbitmq_host: str = "localhost"
    rabbitmq_port: int = Field(default=5672, ge=1, le=65_535)
    rabbitmq_user: str = "agent"
    rabbitmq_password: SecretStr = Field(default=SecretStr(""))
    rabbitmq_vhost: str = "/"
    rabbitmq_heartbeat_seconds: int = Field(default=30, ge=5)
    queue_operation_timeout_seconds: float = Field(default=5.0, gt=0)
    worker_prefetch: int = Field(default=4, ge=1, le=32)
    max_task_attempts: int = Field(default=4, ge=1, le=10)

    mcp_server_url: str = "http://localhost:8000/mcp"
    knowledge_db_path: Path = Path("data/knowledge.db")
    knowledge_source_dir: Path = Path("knowledge")
    commerce_db_path: Path = Path("data/commerce.db")
    checkpoint_db_path: Path = Path("data/checkpoints.db")

    llm_provider: str = "mock"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: SecretStr | None = None
    llm_model: str = "gpt-4.1-mini"
    llm_temperature: float = Field(default=0.1, ge=0, le=2)
    llm_max_output_tokens: int = Field(default=2_048, ge=128, le=32_768)

    task_timeout_seconds: float = Field(default=180.0, gt=0)
    model_timeout_seconds: float = Field(default=45.0, gt=0)
    tool_timeout_seconds: float = Field(default=15.0, gt=0)
    max_agent_iterations: int = Field(default=12, ge=1, le=100)
    max_repeated_steps: int = Field(default=3, ge=1, le=20)
    max_tool_calls_per_turn: int = Field(default=8, ge=1, le=32)
    max_parallel_tools: int = Field(default=4, ge=1, le=16)
    dependency_retry_attempts: int = Field(default=3, ge=1, le=10)
    max_reflections: int = Field(default=1, ge=0, le=3)
    prompt_version: str = "sales-diagnosis-v1"
    skill_version: str = "sales-drop-diagnosis-v1"

    @property
    def rabbitmq_url(self) -> str:
        user = quote_plus(self.rabbitmq_user)
        password = quote_plus(self.rabbitmq_password.get_secret_value())
        vhost = quote_plus(self.rabbitmq_vhost, safe="")
        return f"amqp://{user}:{password}@{self.rabbitmq_host}:{self.rabbitmq_port}/{vhost}"

    @model_validator(mode="after")
    def validate_redis_timeouts(self) -> "Settings":
        blocking_seconds = self.sse_block_milliseconds / 1_000
        if self.redis_socket_timeout_seconds <= blocking_seconds:
            raise ValueError("REDIS_SOCKET_TIMEOUT_SECONDS 必须大于 SSE_BLOCK_MILLISECONDS")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
