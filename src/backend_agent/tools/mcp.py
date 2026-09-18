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
    product_parameters: dict[str, object] = {
        "type": "object",
        "properties": {
            "merchant_id": {
                "type": "string",
                "description": "认证上下文中的模拟商家 ID",
            },
            "product_id": {"type": "string", "description": "模拟商品 ID"},
            "sections": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["product", "inventory", "price", "reviews", "competitors"],
                },
            },
        },
        "required": ["merchant_id", "product_id"],
        "additionalProperties": False,
    }
    return [
        McpRemoteTool(
            client,
            name="get_product_context",
            description="通过 MCP Server 查询商品详情、库存、价格、评论与竞品证据。",
            parameters=product_parameters,
        ),
    ]
