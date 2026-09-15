import json
import re
from abc import ABC, abstractmethod

import httpx
from pydantic import BaseModel, Field, SecretStr

from backend_agent.core.retry import retry_async
from backend_agent.errors import DependencyError
from backend_agent.tools.base import ToolDefinition


class ToolCall(BaseModel):
    call_id: str
    name: str
    arguments: dict[str, object] = Field(default_factory=dict)


class AssistantTurn(BaseModel):
    content: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)


class ModelClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AssistantTurn:
        raise NotImplementedError


class OpenAICompatibleModelClient(ModelClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr,
        model: str,
        temperature: float,
        max_output_tokens: int,
        timeout_seconds: float,
        retry_attempts: int,
    ) -> None:
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_output_tokens = max_output_tokens
        self._retry_attempts = retry_attempts
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AssistantTurn:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "你是可靠的后端运维 Agent。需要事实时优先使用工具；"
                        "不要重复相同工具调用；信息足够后给出简洁结论。"
                    ),
                },
                *self._wire_messages(messages),
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    },
                }
                for tool in tools
            ],
            "tool_choice": "auto",
        }

        async def operation() -> AssistantTurn:
            response = await self._client.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._api_key.get_secret_value()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            if response.status_code >= 500 or response.status_code == 429:
                raise httpx.HTTPStatusError(
                    "模型服务暂时不可用",
                    request=response.request,
                    response=response,
                )
            if response.is_error:
                raise DependencyError(
                    "MODEL_REQUEST_REJECTED",
                    f"模型请求失败，HTTP 状态码 {response.status_code}",
                )
            body = response.json()
            choices = body.get("choices", [])
            if not choices:
                raise DependencyError("MODEL_EMPTY_RESPONSE", "模型未返回 choices")
            message = choices[0].get("message", {})
            tool_calls: list[ToolCall] = []
            for raw_call in message.get("tool_calls", []) or []:
                function = raw_call.get("function", {})
                raw_arguments = function.get("arguments", "{}")
                if isinstance(raw_arguments, dict):
                    arguments = raw_arguments
                else:
                    try:
                        arguments = json.loads(str(raw_arguments))
                    except json.JSONDecodeError as exc:
                        raise DependencyError(
                            "MODEL_INVALID_TOOL_ARGUMENTS",
                            "模型返回了无法解析的工具参数",
                        ) from exc
                if not isinstance(arguments, dict):
                    raise DependencyError(
                        "MODEL_INVALID_TOOL_ARGUMENTS",
                        "模型工具参数必须是 JSON 对象",
                    )
                tool_calls.append(
                    ToolCall(
                        call_id=str(raw_call.get("id") or f"call-{len(tool_calls) + 1}"),
                        name=str(function.get("name", "")),
                        arguments=arguments,
                    )
                )
            return AssistantTurn(content=str(message.get("content") or ""), tool_calls=tool_calls)

        return await retry_async(
            operation,
            attempts=self._retry_attempts,
            retryable=(httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError),
        )

    @staticmethod
    def _wire_messages(messages: list[dict[str, object]]) -> list[dict[str, object]]:
        wire_messages: list[dict[str, object]] = []
        for message in messages:
            role = str(message.get("role", ""))
            if role == "assistant" and isinstance(message.get("tool_calls"), list):
                calls = [ToolCall.model_validate(item) for item in message["tool_calls"]]
                wire_messages.append(
                    {
                        "role": "assistant",
                        "content": message.get("content", ""),
                        "tool_calls": [
                            {
                                "id": call.call_id,
                                "type": "function",
                                "function": {
                                    "name": call.name,
                                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                                },
                            }
                            for call in calls
                        ],
                    }
                )
                continue
            wire_messages.append(message)
        return wire_messages


class MockModelClient(ModelClient):
    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
    ) -> AssistantTurn:
        del tools
        latest_user = next(
            (
                str(message.get("content", ""))
                for message in reversed(messages)
                if message.get("role") == "user"
            ),
            "",
        )
        latest_user_index = max(
            (index for index, message in enumerate(messages) if message.get("role") == "user"),
            default=-1,
        )
        tool_messages = [
            message
            for message in messages[latest_user_index + 1 :]
            if message.get("role") == "tool"
        ]
        tool_names = {str(message.get("name", "")) for message in tool_messages}

        if "rag_search" not in tool_names:
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        call_id="mock-rag-1",
                        name="rag_search",
                        arguments={"query": latest_user, "limit": 3},
                    )
                ]
            )
        if "get_service_health" not in tool_names:
            service_match = re.search(r"[a-zA-Z][a-zA-Z0-9-]*-service", latest_user)
            service = service_match.group(0) if service_match else "order-service"
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        call_id="mock-mcp-1",
                        name="get_service_health",
                        arguments={"service": service},
                    )
                ]
            )

        summaries = [
            f"{message.get('name')}: {message.get('content')}" for message in tool_messages[-2:]
        ]
        return AssistantTurn(
            content="离线演示模型已完成 RAG 与 MCP 调用。\n" + "\n".join(summaries)
        )
