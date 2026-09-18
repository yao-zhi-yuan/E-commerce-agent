import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from backend_agent.agent.graph import AgentOrchestrator
from backend_agent.commerce import CommerceRepository, DEMO_MERCHANT_ID, DEMO_PRODUCT_ID
from backend_agent.domain import ApprovalDecision, QueueMessage, RunStatus, TaskRecord
from backend_agent.evals.assertions import (
    EvaluationResult,
    assert_funnel_math,
    assert_outcome,
    merge_results,
)
from backend_agent.errors import AppError
from backend_agent.llm.client import AssistantTurn, MockModelClient, ModelClient, ToolCall
from backend_agent.rag.knowledge_base import SQLiteKnowledgeBase
from backend_agent.tools.base import AgentTool, ToolDefinition, ToolRegistry
from backend_agent.tools.builtin import RagSearchTool


class EvaluationEventStore:
    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    async def append_event(
        self,
        task_id: str,
        event_type: str,
        payload: dict[str, object],
    ) -> None:
        self.events.append({"task_id": task_id, "event_type": event_type, **payload})


class LocalProductContextTool(AgentTool):
    def __init__(self, commerce: CommerceRepository, *, fail: bool = False) -> None:
        self._commerce = commerce
        self._fail = fail

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="get_product_context",
            description="评测中读取与 MCP 同契约的模拟商品上下文。",
            parameters={
                "type": "object",
                "properties": {
                    "merchant_id": {"type": "string"},
                    "product_id": {"type": "string"},
                    "sections": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["merchant_id", "product_id"],
            },
        )

    async def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        if self._fail:
            raise TimeoutError("injected MCP timeout")
        raw_sections = arguments.get("sections")
        sections = [str(item) for item in raw_sections] if isinstance(raw_sections, list) else None
        return await self._commerce.get_product_context(
            str(arguments["merchant_id"]),
            str(arguments["product_id"]),
            sections,
        )


class InvalidOnceModelClient(ModelClient):
    def __init__(self) -> None:
        self._delegate = MockModelClient()
        self._invalid_sent = False

    async def complete(
        self,
        messages: list[dict[str, object]],
        tools: list[ToolDefinition],
        *,
        system_prompt: str,
    ) -> AssistantTurn:
        turn = await self._delegate.complete(messages, tools, system_prompt=system_prompt)
        submit = next((call for call in turn.tool_calls if call.name == "submit_diagnosis"), None)
        if submit is not None and not self._invalid_sent:
            self._invalid_sent = True
            invalid = dict(submit.arguments)
            invalid["findings"] = [
                {
                    "category": "traffic",
                    "statement": "没有证据的事实结论",
                    "kind": "fact",
                    "evidence_ids": [],
                    "confidence": "high",
                }
            ]
            return AssistantTurn(
                tool_calls=[ToolCall(call_id="invalid-submit", name="submit_diagnosis", arguments=invalid)],
                input_tokens=turn.input_tokens,
                output_tokens=turn.output_tokens,
                finish_reason="tool_calls",
            )
        return turn


async def _build_orchestrator(
    root: Path,
    *,
    model: ModelClient | None = None,
    fail_context: bool = False,
) -> tuple[AgentOrchestrator, CommerceRepository, EvaluationEventStore]:
    commerce = CommerceRepository(root / "commerce.db")
    knowledge = SQLiteKnowledgeBase(root / "knowledge.db", Path("knowledge"))
    await commerce.initialize()
    await knowledge.initialize()
    store = EvaluationEventStore()
    tools = ToolRegistry(
        [
            LocalProductContextTool(commerce, fail=fail_context),
            RagSearchTool(knowledge),
        ]
    )
    orchestrator = AgentOrchestrator(
        model=model or MockModelClient(),
        tools=tools,
        store=store,  # type: ignore[arg-type]
        commerce=commerce,
        checkpointer=InMemorySaver(),
        model_timeout_seconds=5,
        tool_timeout_seconds=2,
        max_iterations=6,
        max_repeated_steps=3,
        max_tool_calls_per_turn=2,
        max_parallel_tools=2,
        max_reflections=1,
        prompt_version="sales-diagnosis-v1",
        skill_version="sales-drop-diagnosis-v1",
    )
    return orchestrator, commerce, store


async def _run_case(case: dict[str, object]) -> dict[str, object]:
    case_id = str(case["case_id"])
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="commerce-eval-") as temp_dir:
        root = Path(temp_dir)
        scenario = str(case.get("scenario", "diagnosis"))
        model: ModelClient | None = InvalidOnceModelClient() if scenario == "reflection" else None
        orchestrator, commerce, store = await _build_orchestrator(
            root,
            model=model,
            fail_context=scenario == "tool_failure",
        )
        task = TaskRecord(
            task_id=f"eval-{case_id}",
            session_id=f"eval-session-{case_id}",
            trace_id=f"eval-trace-{case_id}",
            prompt=str(case["prompt"]),
            merchant_id=DEMO_MERCHANT_ID,
            product_id=case.get("product_id"),
        )
        result: EvaluationResult
        try:
            outcome = await orchestrator.run(task)
            expected = str(case["expected_status"])
            result = assert_outcome(
                case_id=case_id,
                outcome=outcome,
                expected_status=expected,
                required_text=[str(item) for item in case.get("required_text", [])],
                forbidden_text=[str(item) for item in case.get("forbidden_text", [])],
            )
            if scenario == "diagnosis":
                result = merge_results(result, assert_funnel_math(case_id))
            elif scenario == "missing_input":
                result.check(outcome.interaction is not None, "返回商品选择交互")
            elif scenario == "reflection":
                reflection_events = [e for e in store.events if e["event_type"] == "agent.reflection"]
                result.check(len(reflection_events) == 1, "只执行一次 Reflection")
            elif scenario in {"approve", "duplicate", "reject"}:
                interaction = outcome.interaction or {}
                decision = (
                    ApprovalDecision.REJECT if scenario == "reject" else ApprovalDecision.APPROVE
                )
                if scenario == "reject":
                    try:
                        await commerce.record_approval(
                            task_id=task.task_id,
                            merchant_id="other-merchant",
                            plan_id=str(interaction["plan_id"]),
                            plan_version=int(interaction["plan_version"]),
                            action_id=str(interaction["action_id"]),
                            decision=decision,
                        )
                    except AppError as exc:
                        result.check(exc.code == "MERCHANT_FORBIDDEN", "跨商家审批被拒绝")
                    else:
                        result.check(False, "跨商家审批被拒绝")
                approval_id = await commerce.record_approval(
                    task_id=task.task_id,
                    merchant_id=task.merchant_id,
                    plan_id=str(interaction["plan_id"]),
                    plan_version=int(interaction["plan_version"]),
                    action_id=str(interaction["action_id"]),
                    decision=decision,
                )
                resumed = await orchestrator.run(
                    task,
                    resume_kind="approval",
                    resume_id=approval_id,
                )
                expected_final = "rejected" if scenario == "reject" else "completed"
                result.check(resumed.status.value == expected_final, f"审批后状态为 {expected_final}")
                if scenario == "reject":
                    context = await commerce.get_product_context(
                        task.merchant_id, DEMO_PRODUCT_ID, ["product"]
                    )
                    product = context["data"]["product"]  # type: ignore[index]
                    result.check(product["version"] == 1, "拒绝后商品版本不变")
                if scenario == "duplicate":
                    repeated = await commerce.apply_approved_plan(approval_id)
                    result.check(not repeated.applied, "重复执行命中同一回执")
            result.check(
                len([event for event in store.events if event["event_type"] == "agent.model"]) <= 4,
                "模型调用不超过预算",
            )
        except Exception as exc:
            result = EvaluationResult(case_id=case_id)
            expected_error = str(case.get("expected_error", ""))
            result.check(bool(expected_error) and expected_error in type(exc).__name__, f"出现预期错误 {expected_error}")
        return {
            "case_id": case_id,
            "passed": result.passed,
            "passed_checks": result.passed_checks,
            "failed_checks": result.failed_checks,
            "model_calls": len([e for e in store.events if e["event_type"] == "agent.model"]),
            "tool_calls": len([e for e in store.events if e["event_type"] == "tool.started"]),
            "reflection_count": len([e for e in store.events if e["event_type"] == "agent.reflection"]),
            "latency_ms": round((time.perf_counter() - started) * 1_000, 2),
            "trace_id": task.trace_id,
        }


async def run(cases_dir: Path, output: Path) -> int:
    cases = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(cases_dir.glob("*.json"))]
    results = [await _run_case(case) for case in cases]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(result["passed"] for result in results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="运行电商经营诊断助手离线评测")
    parser.add_argument("--cases", type=Path, default=Path("evals/cases"))
    parser.add_argument("--output", type=Path, default=Path("evals/reports/latest.json"))
    args = parser.parse_args()
    raise SystemExit(asyncio.run(run(args.cases, args.output)))


if __name__ == "__main__":
    main()
