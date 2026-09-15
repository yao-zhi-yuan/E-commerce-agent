import asyncio
import hashlib
import json
import logging
import operator
from typing import Annotated, Literal, TypedDict, cast

from langgraph.graph import END, START, StateGraph

from backend_agent.core.logging import trace_span
from backend_agent.domain import AgentCheckpoint, TaskRecord
from backend_agent.errors import AgentLoopDetectedError
from backend_agent.llm.client import AssistantTurn, ModelClient, ToolCall
from backend_agent.repositories.redis_store import RedisStore
from backend_agent.tools.base import ToolRegistry


logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    task_id: str
    session_id: str
    messages: Annotated[list[dict[str, object]], operator.add]
    iteration: int
    repeated_steps: int
    last_signature: str
    status: str
    final_output: str | None


class AgentOrchestrator:
    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolRegistry,
        store: RedisStore,
        model_timeout_seconds: float,
        tool_timeout_seconds: float,
        max_iterations: int,
        max_repeated_steps: int,
        max_tool_calls_per_turn: int,
        max_parallel_tools: int,
    ) -> None:
        self._model = model
        self._tools = tools
        self._store = store
        self._model_timeout_seconds = model_timeout_seconds
        self._tool_timeout_seconds = tool_timeout_seconds
        self._max_iterations = max_iterations
        self._max_repeated_steps = max_repeated_steps
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._tool_semaphore = asyncio.Semaphore(max_parallel_tools)
        self._graph = self._build_graph()

    async def run(self, task: TaskRecord) -> str:
        checkpoint = await self._store.get_checkpoint(task.task_id)
        if checkpoint and checkpoint.status == "completed" and checkpoint.final_output:
            return checkpoint.final_output
        conversation = await self._store.get_conversation(task.session_id)
        state = self._initial_state(task, checkpoint, conversation)
        await self._save_checkpoint(state)
        await self._store.append_event(
            task.task_id,
            "agent.started",
            {"task_id": task.task_id, "session_id": task.session_id},
        )

        final_state: AgentState = state
        config = {"recursion_limit": self._max_iterations * 2 + 4}
        async for snapshot in self._graph.astream(state, config=config, stream_mode="values"):
            final_state = cast(AgentState, snapshot)

        final_output = final_state.get("final_output")
        if not final_output:
            raise AgentLoopDetectedError("Agent 未在循环上限内产生最终结果")
        return final_output

    def _build_graph(self):
        builder = StateGraph(AgentState)
        builder.add_node("model", self._model_node)
        builder.add_node("tools", self._tool_node)
        builder.add_conditional_edges(
            START,
            self._route_start,
            {"model": "model", "tools": "tools"},
        )
        builder.add_conditional_edges(
            "model",
            self._route_after_model,
            {"tools": "tools", "end": END},
        )
        builder.add_edge("tools", "model")
        return builder.compile()

    def _initial_state(
        self,
        task: TaskRecord,
        checkpoint: AgentCheckpoint | None,
        conversation: list[dict[str, object]],
    ) -> AgentState:
        if checkpoint and checkpoint.task_id == task.task_id and checkpoint.status != "completed":
            return AgentState(
                task_id=task.task_id,
                session_id=task.session_id,
                messages=checkpoint.messages,
                iteration=checkpoint.iteration,
                repeated_steps=checkpoint.repeated_steps,
                last_signature=checkpoint.last_signature,
                status="running",
                final_output=checkpoint.final_output,
            )

        return AgentState(
            task_id=task.task_id,
            session_id=task.session_id,
            messages=[*conversation, {"role": "user", "content": task.prompt}],
            iteration=0,
            repeated_steps=0,
            last_signature="",
            status="running",
            final_output=None,
        )

    async def _model_node(self, state: AgentState) -> dict[str, object]:
        if state["iteration"] >= self._max_iterations:
            raise AgentLoopDetectedError(
                f"Agent 超过最大迭代次数 {self._max_iterations}"
            )

        async with trace_span(logger, "agent.model"):
            async with asyncio.timeout(self._model_timeout_seconds):
                turn = await self._model.complete(
                    state["messages"],
                    self._tools.definitions(),
                )

        repeated_steps, signature = self._detect_repetition(state, turn)
        if len(turn.tool_calls) > self._max_tool_calls_per_turn:
            raise AgentLoopDetectedError(
                f"单轮工具调用数超过上限 {self._max_tool_calls_per_turn}"
            )
        if repeated_steps >= self._max_repeated_steps:
            raise AgentLoopDetectedError(
                f"连续 {self._max_repeated_steps} 次产生相同工具调用，已终止"
            )

        assistant_message = self._assistant_message(turn)
        next_messages = [*state["messages"], assistant_message]
        next_iteration = state["iteration"] + 1
        has_tool_calls = bool(turn.tool_calls)
        status = "running" if has_tool_calls else "completed"
        final_output = None if has_tool_calls else turn.content
        next_state = AgentState(
            task_id=state["task_id"],
            session_id=state["session_id"],
            messages=next_messages,
            iteration=next_iteration,
            repeated_steps=repeated_steps,
            last_signature=signature,
            status=status,
            final_output=final_output,
        )
        await self._save_checkpoint(next_state)
        if status == "completed":
            await self._store.save_conversation(state["session_id"], next_messages)
        await self._store.append_event(
            state["task_id"],
            "agent.model",
            {
                "task_id": state["task_id"],
                "iteration": next_iteration,
                "tool_calls": [tool_call.name for tool_call in turn.tool_calls],
                "completed": not has_tool_calls,
                "content": turn.content if not has_tool_calls else "",
            },
        )
        return {
            "messages": [assistant_message],
            "iteration": next_iteration,
            "repeated_steps": repeated_steps,
            "last_signature": signature,
            "status": status,
            "final_output": final_output,
        }

    async def _tool_node(self, state: AgentState) -> dict[str, object]:
        last_message = state["messages"][-1]
        raw_calls = last_message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("assistant tool_calls must be a list")
        calls = [ToolCall.model_validate(raw_call) for raw_call in raw_calls]
        tool_messages = await asyncio.gather(
            *(self._execute_tool(state["task_id"], tool_call) for tool_call in calls)
        )
        next_state = AgentState(
            task_id=state["task_id"],
            session_id=state["session_id"],
            messages=[*state["messages"], *tool_messages],
            iteration=state["iteration"],
            repeated_steps=state["repeated_steps"],
            last_signature=state["last_signature"],
            status="running",
            final_output=None,
        )
        await self._save_checkpoint(next_state)
        return {"messages": tool_messages, "status": "running", "final_output": None}

    async def _execute_tool(
        self,
        task_id: str,
        tool_call: ToolCall,
    ) -> dict[str, object]:
        async with self._tool_semaphore:
            tool = self._tools.get(tool_call.name)
            if tool is None:
                result: dict[str, object] = {
                    "error": "TOOL_NOT_FOUND",
                    "message": f"未知工具：{tool_call.name}",
                }
            else:
                await self._store.append_event(
                    task_id,
                    "tool.started",
                    {
                        "task_id": task_id,
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.call_id,
                    },
                )
                try:
                    async with trace_span(logger, "agent.tool", tool_name=tool_call.name):
                        async with asyncio.timeout(self._tool_timeout_seconds):
                            result = await tool.execute(tool_call.arguments)
                except Exception as exc:
                    await self._store.append_event(
                        task_id,
                        "tool.failed",
                        {
                            "task_id": task_id,
                            "tool_name": tool_call.name,
                            "tool_call_id": tool_call.call_id,
                            "error": type(exc).__name__,
                        },
                    )
                    raise
                await self._store.append_event(
                    task_id,
                    "tool.completed",
                    {
                        "task_id": task_id,
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.call_id,
                        "result": result,
                    },
                )

        return {
            "role": "tool",
            "tool_call_id": tool_call.call_id,
            "name": tool_call.name,
            "content": json.dumps(result, ensure_ascii=False, default=str),
        }

    @staticmethod
    def _assistant_message(turn: AssistantTurn) -> dict[str, object]:
        message: dict[str, object] = {"role": "assistant", "content": turn.content}
        if turn.tool_calls:
            message["tool_calls"] = [tool_call.model_dump(mode="json") for tool_call in turn.tool_calls]
        return message

    @staticmethod
    def _route_after_model(state: AgentState) -> Literal["tools", "end"]:
        last_message = state["messages"][-1]
        tool_calls = last_message.get("tool_calls")
        return "tools" if isinstance(tool_calls, list) and tool_calls else "end"

    @staticmethod
    def _route_start(state: AgentState) -> Literal["model", "tools"]:
        if not state["messages"]:
            return "model"
        last_message = state["messages"][-1]
        tool_calls = last_message.get("tool_calls")
        if last_message.get("role") == "assistant" and isinstance(tool_calls, list) and tool_calls:
            return "tools"
        return "model"

    @staticmethod
    def _detect_repetition(state: AgentState, turn: AssistantTurn) -> tuple[int, str]:
        if not turn.tool_calls:
            return 0, ""
        normalized = [
            {
                "name": call.name,
                "arguments": call.arguments,
            }
            for call in turn.tool_calls
        ]
        signature = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        repeated_steps = state["repeated_steps"] + 1 if signature == state["last_signature"] else 1
        return repeated_steps, signature

    async def _save_checkpoint(self, state: AgentState) -> None:
        await self._store.save_checkpoint(
            AgentCheckpoint(
                task_id=state["task_id"],
                session_id=state["session_id"],
                messages=state["messages"],
                iteration=state["iteration"],
                repeated_steps=state["repeated_steps"],
                last_signature=state["last_signature"],
                status=state["status"],
                final_output=state["final_output"],
            )
        )
