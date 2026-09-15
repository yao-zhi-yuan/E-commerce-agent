import asyncio
from typing import cast

import httpx
from mcp import Client

from backend_agent.core.retry import retry_async
from backend_agent.tools.base import AgentTool, ToolDefinition


class McpToolClient:
    def __init__(self, server_url: str, *, timeout_seconds: float, retry_attempts: int) -> None:
        self._server_url = server_url
        self._timeout_seconds = timeout_seconds
        self._retry_attempts = retry_attempts

    async def call(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        async def operation() -> dict[str, object]:
            async with asyncio.timeout(self._timeout_seconds):
                async with Client(self._server_url) as client:
                    result = await client.call_tool(name, arguments)
                    structured = getattr(result, "structured_content", None)
                    if isinstance(structured, dict):
                        return cast(dict[str, object], structured)
                    if hasattr(result, "model_dump"):
                        dumped = result.model_dump(mode="json")
                        if isinstance(dumped, dict):
                            return cast(dict[str, object], dumped)
                    return {"content": str(result)}

        return await retry_async(
            operation,
            attempts=self._retry_attempts,
            retryable=(httpx.HTTPError, TimeoutError, ConnectionError),
        )


class McpRemoteTool(AgentTool):
    def __init__(
        self,
        client: McpToolClient,
        *,
        name: str,
        description: str,
        parameters: dict[str, object],
    ) -> None:
        self._client = client
        self._definition = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
        )

    @property
    def definition(self) -> ToolDefinition:
        return self._definition

    async def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        return await self._client.call(self._definition.name, arguments)


def build_mcp_tools(client: McpToolClient) -> list[McpRemoteTool]:
    service_parameters: dict[str, object] = {
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "description": "服务名，例如 order-service",
            }
        },
        "required": ["service"],
        "additionalProperties": False,
    }
    return [
        McpRemoteTool(
            client,
            name="get_service_health",
            description="通过 MCP Server 查询服务健康状态和关键依赖。",
            parameters=service_parameters,
        ),
        McpRemoteTool(
            client,
            name="get_incident_runbook",
            description="通过 MCP Server 获取服务故障处理手册。",
            parameters=service_parameters,
        ),
    ]

