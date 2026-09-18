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
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class ModelClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
        *,
        system_prompt: str,
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
        *,
        system_prompt: str,
    ) -> AssistantTurn:
        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_output_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
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
            choice = choices[0]
            message = choice.get("message", {})
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
            usage = body.get("usage", {})
            return AssistantTurn(
                content=str(message.get("content") or ""),
                tool_calls=tool_calls,
                input_tokens=_optional_int(usage.get("prompt_tokens")),
                output_tokens=_optional_int(usage.get("completion_tokens")),
                finish_reason=str(choice.get("finish_reason") or "") or None,
            )

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
        *,
        system_prompt: str,
    ) -> AssistantTurn:
        del tools, system_prompt
        task_context_index = max(
            (
                index
                for index, message in enumerate(messages)
                if message.get("role") == "user"
                and "<task_context>" in str(message.get("content", ""))
            ),
            default=-1,
        )
        tool_messages = [
            message
            for message in messages[task_context_index + 1 :]
            if message.get("role") == "tool"
        ]
        tool_names = {str(message.get("name", "")) for message in tool_messages}

        task_context = _latest_task_context(messages)
        merchant_id = str(task_context.get("merchant_id", "demo-merchant"))
        product_id = str(task_context.get("product_id", "wireless-headphones"))
        if "get_product_context" not in tool_names:
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        call_id="mock-context-1",
                        name="get_product_context",
                        arguments={
                            "merchant_id": merchant_id,
                            "product_id": product_id,
                            "sections": ["product", "inventory", "price", "reviews", "competitors"],
                        },
                    )
                ],
                input_tokens=240,
                output_tokens=32,
                finish_reason="tool_calls",
            )
        if "search_knowledge" not in tool_names:
            return AssistantTurn(
                tool_calls=[
                    ToolCall(
                        call_id="mock-knowledge-1",
                        name="search_knowledge",
                        arguments={"query": "销量下降 库存 商品内容 兼容性", "limit": 3},
                    )
                ],
                input_tokens=420,
                output_tokens=28,
                finish_reason="tool_calls",
            )
        metrics = task_context.get("metrics", {})
        comparison = metrics.get("comparison", {}) if isinstance(metrics, dict) else {}
        current = comparison.get("current", {}) if isinstance(comparison, dict) else {}
        previous = comparison.get("comparison", {}) if isinstance(comparison, dict) else {}
        current_orders = int(current.get("paid_orders", 112)) if isinstance(current, dict) else 112
        previous_orders = int(previous.get("paid_orders", 200)) if isinstance(previous, dict) else 200
        diagnosis = {
            "summary": (
                f"支付订单从 {previous_orders} 降至 {current_orders}。"
                "流量下降和点击到支付转化下降共同贡献，热门 SKU 缺货与兼容信息不清需要优先处理。"
            ),
            "findings": [
                {
                    "category": "traffic",
                    "statement": "搜索渠道曝光下降，是订单减少的已确认贡献因素。",
                    "kind": "fact",
                    "evidence_ids": ["metric_search_impressions"],
                    "confidence": "high",
                },
                {
                    "category": "inventory",
                    "statement": "主销黑色 SKU 连续三天缺货，可能压低转化。",
                    "kind": "hypothesis",
                    "evidence_ids": ["inventory_primary_sku"],
                    "confidence": "medium",
                },
                {
                    "category": "content",
                    "statement": "评论中反复出现兼容性疑问，详情页缺少清晰说明。",
                    "kind": "fact",
                    "evidence_ids": ["review_compatibility_theme"],
                    "confidence": "high",
                },
                {
                    "category": "price",
                    "statement": "两个窗口价格未变化，没有证据支持立即降价。",
                    "kind": "fact",
                    "evidence_ids": ["price_unchanged"],
                    "confidence": "high",
                },
            ],
            "actions": [
                "优先恢复主销黑色 SKU 的稳定库存并观察对应转化率。",
                "在核实支持设备后补充兼容性说明。",
                "继续监控搜索渠道曝光，不在缺少价格证据时直接降价。",
            ],
            "proposed_patch": {
                "field": "description",
                "new_value": (
                    "Lightweight wireless headphones with active noise cancellation. "
                    "Compatibility varies by device; verify the device audio interface before purchase."
                ),
                "reason": "评论证据显示买家反复询问兼容性，草稿只加入可核实的购买前确认提示。",
            },
            "missing_data": ["缺货与转化下降之间尚无实验性因果证据"],
        }
        return AssistantTurn(
            tool_calls=[
                ToolCall(
                    call_id="mock-submit-1",
                    name="submit_diagnosis",
                    arguments=diagnosis,
                )
            ],
            input_tokens=780,
            output_tokens=310,
            finish_reason="tool_calls",
        )


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _latest_task_context(messages: list[dict[str, object]]) -> dict[str, object]:
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        match = re.search(r"<task_context>(.+)</task_context>", content, re.DOTALL)
        if not match:
            continue
        try:
            parsed = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}
