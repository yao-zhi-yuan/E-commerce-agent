import asyncio
import hashlib
import json
import logging
import operator
from typing import Annotated, Literal, TypedDict, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import ValidationError

from backend_agent.commerce import CommerceRepository
from backend_agent.core.logging import trace_span
from backend_agent.domain import Diagnosis, RunOutcome, RunStatus, TaskRecord
from backend_agent.errors import AgentLoopDetectedError, AppError
from backend_agent.llm.client import AssistantTurn, ModelClient, ToolCall
from backend_agent.repositories.redis_store import RedisStore
from backend_agent.tools.base import ToolDefinition, ToolRegistry


logger = logging.getLogger(__name__)


class AgentState(TypedDict, total=False):
    task_id: str
    session_id: str
    merchant_id: str
    product_id: str | None
    prompt: str
    messages: Annotated[list[dict[str, object]], operator.add]
    metrics: dict[str, object]
    diagnosis: dict[str, object] | None
    iteration: int
    repeated_steps: int
    last_signature: str
    reflection_count: int
    validation_errors: list[str]
    interaction: dict[str, object] | None
    final_status: str | None
    final_output: str | None


class AgentOrchestrator:
    def __init__(
        self,
        *,
        model: ModelClient,
        tools: ToolRegistry,
        store: RedisStore,
        commerce: CommerceRepository,
        checkpointer: BaseCheckpointSaver,
        model_timeout_seconds: float,
        tool_timeout_seconds: float,
        max_iterations: int,
        max_repeated_steps: int,
        max_tool_calls_per_turn: int,
        max_parallel_tools: int,
        max_reflections: int,
        prompt_version: str,
        skill_version: str,
    ) -> None:
        self._model = model
        self._tools = tools
        self._store = store
        self._commerce = commerce
        self._model_timeout_seconds = model_timeout_seconds
        self._tool_timeout_seconds = tool_timeout_seconds
        self._max_iterations = max_iterations
        self._max_repeated_steps = max_repeated_steps
        self._max_tool_calls_per_turn = max_tool_calls_per_turn
        self._tool_semaphore = asyncio.Semaphore(max_parallel_tools)
        self._max_reflections = max_reflections
        self._prompt_version = prompt_version
        self._skill_version = skill_version
        self._checkpointer = checkpointer
        self._graph = self._build_graph(checkpointer)

    async def run(
        self,
        task: TaskRecord,
        *,
        resume_kind: str | None = None,
        resume_id: str | None = None,
    ) -> RunOutcome:
        config = {
            "configurable": {"thread_id": task.task_id},
            "recursion_limit": self._max_iterations * 3 + 12,
        }
        if resume_kind not in {None, "retry", "input", "approval"}:
            raise AppError("INVALID_RESUME_KIND", "恢复任务类型无效", http_status=422)
        checkpoint = await self._checkpointer.aget_tuple(config)
        if checkpoint is None:
            if resume_kind in {"input", "approval"}:
                raise AppError(
                    "CHECKPOINT_NOT_FOUND",
                    "恢复任务缺少图检查点",
                    http_status=409,
                )
            graph_input: AgentState | Command | None = self._initial_state(task)
        else:
            snapshot = await self._graph.aget_state(config)
            pending_interaction = self._pending_interaction(snapshot.tasks)
            if pending_interaction is not None:
                expected_kind = self._resume_kind_for_interaction(pending_interaction)
                if resume_kind != expected_kind:
                    return self._interrupted_outcome(pending_interaction)
                if not resume_id:
                    raise AppError(
                        "RESUME_VALUE_REQUIRED",
                        "恢复任务缺少恢复值",
                        http_status=422,
                    )
                graph_input = Command(
                    resume={
                        "resume_kind": resume_kind,
                        "resume_id": resume_id,
                    }
                )
            else:
                graph_input = None
        if resume_kind in {"input", "approval"} and not resume_id:
            raise AppError("RESUME_VALUE_REQUIRED", "恢复任务缺少恢复值", http_status=422)
        result = cast(dict[str, object], await self._graph.ainvoke(graph_input, config=config))
        raw_interrupts = result.get("__interrupt__")
        if isinstance(raw_interrupts, tuple | list) and raw_interrupts:
            value = getattr(raw_interrupts[0], "value", None)
            interaction = value if isinstance(value, dict) else {"type": "input.required"}
            return self._interrupted_outcome(interaction)
        final_status = str(result.get("final_status") or RunStatus.COMPLETED.value)
        return RunOutcome(
            status=RunStatus(final_status),
            result=str(result.get("final_output") or ""),
            interaction=cast(dict[str, object] | None, result.get("interaction")),
        )

    @staticmethod
    def _pending_interaction(tasks: tuple[object, ...]) -> dict[str, object] | None:
        for task in tasks:
            for interruption in getattr(task, "interrupts", ()):
                value = getattr(interruption, "value", None)
                return value if isinstance(value, dict) else {"type": "input.required"}
        return None

    @staticmethod
    def _resume_kind_for_interaction(interaction: dict[str, object]) -> str:
        return "approval" if interaction.get("type") == "approval.required" else "input"

    @classmethod
    def _interrupted_outcome(cls, interaction: dict[str, object]) -> RunOutcome:
        status = (
            RunStatus.AWAITING_APPROVAL
            if cls._resume_kind_for_interaction(interaction) == "approval"
            else RunStatus.AWAITING_INPUT
        )
        return RunOutcome(status=status, interaction=interaction)

    @staticmethod
    def _initial_state(task: TaskRecord) -> AgentState:
        return AgentState(
            task_id=task.task_id,
            session_id=task.session_id,
            merchant_id=task.merchant_id,
            product_id=task.product_id,
            prompt=task.prompt,
            messages=[{"role": "user", "content": task.prompt}],
            iteration=0,
            repeated_steps=0,
            last_signature="",
            reflection_count=0,
            validation_errors=[],
            diagnosis=None,
            interaction=None,
            final_status=None,
            final_output=None,
        )

    def _build_graph(self, checkpointer: BaseCheckpointSaver):
        builder = StateGraph(AgentState)
        builder.add_node("scope", self._scope_node)
        builder.add_node("metrics", self._metrics_node)
        builder.add_node("model", self._model_node)
        builder.add_node("tools", self._tool_node)
        builder.add_node("validate", self._validation_node)
        builder.add_node("plan", self._plan_node)
        builder.add_node("approval", self._approval_node)
        builder.add_edge(START, "scope")
        builder.add_edge("scope", "metrics")
        builder.add_edge("metrics", "model")
        builder.add_conditional_edges(
            "model",
            self._route_after_model,
            {"tools": "tools", "validate": "validate"},
        )
        builder.add_edge("tools", "model")
        builder.add_conditional_edges(
            "validate",
            self._route_after_validation,
            {"model": "model", "plan": "plan"},
        )
        builder.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"approval": "approval", "end": END},
        )
        builder.add_edge("approval", END)
        return builder.compile(checkpointer=checkpointer)

    async def _scope_node(self, state: AgentState) -> dict[str, object]:
        async with trace_span(logger, "graph.node", node_name="scope"):
            product_id = state.get("product_id")
            if not product_id:
                candidates = await self._commerce.resolve_product(
                    state["merchant_id"], state["prompt"]
                )
                if len(candidates) == 1:
                    product_id = str(candidates[0]["product_id"])
                else:
                    decision = interrupt(
                        {
                            "type": "input.required",
                            "field": "product_id",
                            "message": "请确认要诊断的商品",
                            "candidates": candidates,
                        }
                    )
                    if not isinstance(decision, dict) or decision.get("resume_kind") != "input":
                        raise AppError("INVALID_INPUT_RESUME", "商品选择恢复参数无效", http_status=422)
                    product_id = str(decision.get("resume_id", ""))
                    matches = await self._commerce.resolve_product(state["merchant_id"], product_id)
                    if not any(item["product_id"] == product_id for item in matches):
                        raise AppError("PRODUCT_NOT_FOUND", "所选商品不存在", http_status=404)
            await self._store.append_event(
                state["task_id"],
                "graph.scope_resolved",
                {"node_name": "scope", "merchant_id": state["merchant_id"], "product_id": product_id},
            )
            return {"product_id": product_id}

    async def _metrics_node(self, state: AgentState) -> dict[str, object]:
        async with trace_span(logger, "graph.node", node_name="metrics"):
            product_id = self._require_product_id(state)
            metrics = await self._commerce.get_metrics(state["merchant_id"], product_id)
            context_message = {
                "role": "user",
                "content": (
                    "以下是服务端固定的任务上下文，身份和指标不可由模型修改：\n"
                    f"<task_context>{json.dumps({'merchant_id': state['merchant_id'], 'product_id': product_id, 'metrics': metrics}, ensure_ascii=False)}</task_context>"
                ),
            }
            await self._store.append_event(
                state["task_id"],
                "graph.metrics_computed",
                {
                    "node_name": "metrics",
                    "snapshot_id": metrics["snapshot_id"],
                    "metric_version": metrics["metric_version"],
                },
            )
            return {"metrics": metrics, "messages": [context_message]}

    async def _model_node(self, state: AgentState) -> dict[str, object]:
        iteration = int(state.get("iteration", 0))
        if iteration >= self._max_iterations:
            raise AgentLoopDetectedError(f"Agent 超过最大迭代次数 {self._max_iterations}")
        async with trace_span(
            logger,
            "agent.model",
            node_name="diagnose",
            role="main_agent",
            iteration=iteration + 1,
            reflection_count=int(state.get("reflection_count", 0)),
            prompt_version=self._prompt_version,
            skill_version=self._skill_version,
        ):
            async with asyncio.timeout(self._model_timeout_seconds):
                turn = await self._model.complete(
                    state["messages"],
                    [*self._tools.definitions(), self._submit_definition()],
                    system_prompt=self._system_prompt(state),
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
        diagnosis: dict[str, object] | None = None
        submit_calls = [call for call in turn.tool_calls if call.name == "submit_diagnosis"]
        if submit_calls:
            if len(submit_calls) != 1 or len(turn.tool_calls) != 1:
                raise AppError(
                    "INVALID_DIAGNOSIS_SUBMISSION",
                    "submit_diagnosis 必须单独调用一次",
                    http_status=422,
                )
            try:
                diagnosis = Diagnosis.model_validate(submit_calls[0].arguments).model_dump(mode="json")
            except ValidationError as exc:
                diagnosis = {"validation_parse_error": str(exc)}
        elif not turn.tool_calls:
            diagnosis = {"validation_parse_error": "模型未调用 submit_diagnosis"}
        message = self._assistant_message(turn)
        await self._store.append_event(
            state["task_id"],
            "agent.model",
            {
                "node_name": "diagnose",
                "role": "main_agent",
                "iteration": iteration + 1,
                "reflection_count": int(state.get("reflection_count", 0)),
                "prompt_version": self._prompt_version,
                "skill_version": self._skill_version,
                "input_tokens": turn.input_tokens,
                "output_tokens": turn.output_tokens,
                "finish_reason": turn.finish_reason,
                "tool_calls": [call.name for call in turn.tool_calls],
            },
        )
        return {
            "messages": [message],
            "diagnosis": diagnosis,
            "iteration": iteration + 1,
            "repeated_steps": repeated_steps,
            "last_signature": signature,
        }

    async def _tool_node(self, state: AgentState) -> dict[str, object]:
        last_message = state["messages"][-1]
        raw_calls = last_message.get("tool_calls", [])
        if not isinstance(raw_calls, list):
            raise ValueError("assistant tool_calls must be a list")
        calls = [ToolCall.model_validate(call) for call in raw_calls]
        messages = await asyncio.gather(*(self._execute_tool(state, call) for call in calls))
        return {"messages": messages}

    async def _validation_node(self, state: AgentState) -> dict[str, object]:
        async with trace_span(logger, "graph.node", node_name="validate"):
            errors = self._validate_diagnosis(state)
            reflection_count = int(state.get("reflection_count", 0))
            if errors and reflection_count < self._max_reflections:
                tool_call_id = self._diagnosis_tool_call_id(state)
                correction_messages: list[dict[str, object]] = []
                if tool_call_id is not None:
                    correction_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "name": "submit_diagnosis",
                            "content": json.dumps(
                                {"accepted": False, "validation_errors": errors},
                                ensure_ascii=False,
                            ),
                        }
                    )
                correction_messages.append(
                    {
                        "role": "user",
                        "content": "诊断校验失败，请只修正这些问题后重新调用 submit_diagnosis："
                        + json.dumps(errors, ensure_ascii=False),
                    }
                )
                await self._store.append_event(
                    state["task_id"],
                    "agent.reflection",
                    {
                        "node_name": "validate",
                        "reflection_count": reflection_count + 1,
                        "errors": errors,
                    },
                )
                return {
                    "diagnosis": None,
                    "validation_errors": errors,
                    "reflection_count": reflection_count + 1,
                    "messages": correction_messages,
                }
            if errors:
                raise AppError(
                    "DIAGNOSIS_VALIDATION_FAILED",
                    "诊断在一次修正后仍未通过：" + "；".join(errors),
                    http_status=422,
                )
            await self._store.append_event(
                state["task_id"],
                "graph.diagnosis_validated",
                {
                    "node_name": "validate",
                    "reflection_count": reflection_count,
                    "validation": "passed",
                },
            )
            return {"validation_errors": []}

    async def _plan_node(self, state: AgentState) -> dict[str, object]:
        diagnosis = Diagnosis.model_validate(state["diagnosis"])
        result = self._render_diagnosis(state, diagnosis)
        if diagnosis.proposed_patch is None:
            return {
                "final_status": RunStatus.COMPLETED.value,
                "final_output": result,
                "interaction": None,
            }
        plan = await self._commerce.create_plan(
            task_id=state["task_id"],
            merchant_id=state["merchant_id"],
            product_id=self._require_product_id(state),
            patch=diagnosis.proposed_patch,
        )
        interaction = {
            "type": "approval.required",
            "plan_id": plan.plan_id,
            "plan_version": plan.plan_version,
            "action_id": plan.action_id,
            "product_id": plan.product_id,
            "product_version": plan.product_version,
            "before": plan.before_value,
            "after": plan.after_value,
            "reason": plan.reason,
        }
        return {"final_output": result, "interaction": interaction}

    async def _approval_node(self, state: AgentState) -> dict[str, object]:
        interaction = state.get("interaction")
        if not isinstance(interaction, dict):
            raise AppError("APPROVAL_STATE_MISSING", "批准上下文丢失", http_status=500)
        decision = interrupt(interaction)
        if not isinstance(decision, dict) or decision.get("resume_kind") != "approval":
            raise AppError("INVALID_APPROVAL_RESUME", "审批恢复参数无效", http_status=422)
        approval_id = str(decision.get("resume_id", ""))
        approval = await self._commerce.get_approval(approval_id)
        if approval is None:
            raise AppError("APPROVAL_NOT_FOUND", "审批记录不存在", http_status=404)
        if approval["decision"] == "reject":
            await self._store.append_event(
                state["task_id"],
                "approval.rejected",
                {"node_name": "approval", "plan_id": approval["plan_id"]},
            )
            return {
                "final_status": RunStatus.REJECTED.value,
                "final_output": (state.get("final_output") or "") + "\n\n商家已拒绝执行商品修改。",
            }
        receipt = await self._commerce.apply_approved_plan(approval_id)
        await self._store.append_event(
            state["task_id"],
            "action.applied",
            {
                "node_name": "execute",
                "plan_id": approval["plan_id"],
                "action_id": approval["action_id"],
                "receipt_id": receipt.receipt_id,
                "product_version": receipt.product_version,
                "idempotency_hit": not receipt.applied,
            },
        )
        return {
            "final_status": RunStatus.COMPLETED.value,
            "final_output": (
                (state.get("final_output") or "")
                + "\n\n执行结果：商品详情已按批准计划更新。"
                + f" 回执 {receipt.receipt_id}，商品版本 {receipt.product_version}。"
            ),
        }

    async def _execute_tool(self, state: AgentState, call: ToolCall) -> dict[str, object]:
        async with self._tool_semaphore:
            task_id = state["task_id"]
            tool = self._tools.get(call.name)
            if tool is None:
                raise AppError("TOOL_NOT_FOUND", f"未知读工具：{call.name}", http_status=422)
            expected_product_id = self._require_product_id(state)
            if call.name in {"get_metrics", "get_product_context"}:
                if call.arguments.get("merchant_id") != state["merchant_id"]:
                    raise AppError("MERCHANT_FORBIDDEN", "工具参数不能覆盖认证商家", http_status=403)
                if call.arguments.get("product_id") != expected_product_id:
                    raise AppError("PRODUCT_SCOPE_FORBIDDEN", "工具参数不能越过当前商品范围", http_status=403)
            await self._store.append_event(
                task_id,
                "tool.started",
                {"tool_name": call.name, "tool_call_id": call.call_id},
            )
            try:
                async with trace_span(
                    logger,
                    "agent.tool",
                    tool_name=call.name,
                    tool_call_id=call.call_id,
                ):
                    async with asyncio.timeout(self._tool_timeout_seconds):
                        result = await tool.execute(call.arguments)
            except Exception as exc:
                await self._store.append_event(
                    task_id,
                    "tool.failed",
                    {
                        "tool_name": call.name,
                        "tool_call_id": call.call_id,
                        "error": type(exc).__name__,
                    },
                )
                raise
            await self._store.append_event(
                task_id,
                "tool.completed",
                {
                    "tool_name": call.name,
                    "tool_call_id": call.call_id,
                    "evidence_ids": result.get("evidence_ids", []),
                    "snapshot_id": result.get("snapshot_id"),
                },
            )
            return {
                "role": "tool",
                "tool_call_id": call.call_id,
                "name": call.name,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            }

    def _validate_diagnosis(self, state: AgentState) -> list[str]:
        raw = state.get("diagnosis")
        if not isinstance(raw, dict) or "validation_parse_error" in raw:
            return ["submit_diagnosis 参数不符合 Diagnosis schema"]
        diagnosis = Diagnosis.model_validate(raw)
        evidence_ids = self._collect_evidence_ids(state)
        errors: list[str] = []
        for finding in diagnosis.findings:
            if finding.kind == "fact" and not finding.evidence_ids:
                errors.append(f"事实结论缺少证据：{finding.statement}")
            unknown = set(finding.evidence_ids) - evidence_ids
            if unknown:
                errors.append("引用了不存在的证据：" + ", ".join(sorted(unknown)))
        if diagnosis.proposed_patch:
            content_findings = [
                finding
                for finding in diagnosis.findings
                if finding.category == "content" and finding.kind == "fact"
            ]
            if not any(finding.evidence_ids for finding in content_findings):
                errors.append("商品详情修改缺少有证据支持的内容事实")
        return errors

    @staticmethod
    def _collect_evidence_ids(state: AgentState) -> set[str]:
        evidence: set[str] = set()
        metrics = state.get("metrics", {})
        if isinstance(metrics, dict):
            evidence.update(str(item) for item in metrics.get("evidence_ids", []))
        for message in state.get("messages", []):
            if message.get("role") != "tool":
                continue
            try:
                content = json.loads(str(message.get("content", "{}")))
            except json.JSONDecodeError:
                continue
            if isinstance(content, dict):
                evidence.update(str(item) for item in content.get("evidence_ids", []))
        return evidence

    def _system_prompt(self, state: AgentState) -> str:
        return (
            "你是电商经营诊断 Agent。服务端已经固定商家、商品和指标口径。"
            "只使用读工具查证库存、价格、评论、竞品和知识规则；不得请求或执行写操作。"
            "区分事实、假设和缺口，不把算术贡献说成因果。"
            "信息足够时必须单独调用 submit_diagnosis，不能输出自由文本作为最终答案。"
            f"当前 Prompt 版本 {self._prompt_version}，Skill 版本 {self._skill_version}。"
            f"当前商家 {state['merchant_id']}，商品 {self._require_product_id(state)}。"
        )

    @staticmethod
    def _submit_definition() -> ToolDefinition:
        return ToolDefinition(
            name="submit_diagnosis",
            description="提交结构化销量下降诊断并结束查证循环。",
            parameters=Diagnosis.model_json_schema(),
        )

    @staticmethod
    def _assistant_message(turn: AssistantTurn) -> dict[str, object]:
        message: dict[str, object] = {"role": "assistant", "content": turn.content}
        if turn.tool_calls:
            message["tool_calls"] = [call.model_dump(mode="json") for call in turn.tool_calls]
        return message

    @staticmethod
    def _route_after_model(state: AgentState) -> Literal["tools", "validate"]:
        if state.get("diagnosis") is not None:
            return "validate"
        return "tools"

    @staticmethod
    def _route_after_validation(state: AgentState) -> Literal["model", "plan"]:
        return "model" if state.get("validation_errors") else "plan"

    @staticmethod
    def _route_after_plan(state: AgentState) -> Literal["approval", "end"]:
        return "approval" if state.get("interaction") else "end"

    @staticmethod
    def _detect_repetition(state: AgentState, turn: AssistantTurn) -> tuple[int, str]:
        calls = [call for call in turn.tool_calls if call.name != "submit_diagnosis"]
        if not calls:
            return 0, ""
        normalized = [{"name": call.name, "arguments": call.arguments} for call in calls]
        signature = hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
        repeated = int(state.get("repeated_steps", 0)) + 1 if signature == state.get("last_signature") else 1
        return repeated, signature

    @staticmethod
    def _require_product_id(state: AgentState) -> str:
        product_id = state.get("product_id")
        if not product_id:
            raise AppError("PRODUCT_SCOPE_MISSING", "商品范围尚未确定", http_status=422)
        return product_id

    @staticmethod
    def _diagnosis_tool_call_id(state: AgentState) -> str | None:
        last_message = state["messages"][-1]
        raw_calls = last_message.get("tool_calls", [])
        if isinstance(raw_calls, list):
            for raw_call in raw_calls:
                call = ToolCall.model_validate(raw_call)
                if call.name == "submit_diagnosis":
                    return call.call_id
        return None

    @staticmethod
    def _render_diagnosis(state: AgentState, diagnosis: Diagnosis) -> str:
        metrics = cast(dict[str, object], state["metrics"])["comparison"]
        comparison = cast(dict[str, object], metrics)
        order_change = int(comparison["order_change"])
        order_rate = comparison["order_change_rate"]
        traffic = comparison["traffic_contribution"]
        conversion = comparison["conversion_contribution"]
        rate_text = "无法计算" if order_rate is None else f"{float(order_rate):.1%}"
        traffic_text = "无法计算" if traffic is None else f"{float(traffic):.0f}"
        conversion_text = "无法计算" if conversion is None else f"{float(conversion):.0f}"
        lines = [
            diagnosis.summary,
            f"订单变化：{order_change}（{rate_text}）",
            f"流量贡献：{traffic_text}；转化贡献：{conversion_text}",
            "主要发现：",
        ]
        lines.extend(
            f"- [{finding.kind}] {finding.statement}（证据：{', '.join(finding.evidence_ids) or '无'}）"
            for finding in diagnosis.findings
        )
        lines.append("行动计划：")
        lines.extend(f"- {action}" for action in diagnosis.actions)
        if diagnosis.missing_data:
            lines.append("仍需验证：" + "；".join(diagnosis.missing_data))
        return "\n".join(lines)
