from dataclasses import dataclass, field

from backend_agent.commerce import compare_funnel
from backend_agent.domain import RunOutcome


@dataclass(slots=True)
class EvaluationResult:
    case_id: str
    passed_checks: list[str] = field(default_factory=list)
    failed_checks: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failed_checks

    def check(self, condition: bool, label: str) -> None:
        target = self.passed_checks if condition else self.failed_checks
        target.append(label)


def assert_funnel_math(case_id: str) -> EvaluationResult:
    result = EvaluationResult(case_id=case_id)
    comparison = compare_funnel(
        comparison_impressions=100_000,
        comparison_clicks=4_000,
        comparison_orders=200,
        current_impressions=80_000,
        current_clicks=3_200,
        current_orders=112,
    )
    result.check(comparison.order_change == -88, "订单变化为 -88")
    result.check(comparison.order_change_rate == -0.44, "订单降幅为 44%")
    result.check(round(comparison.traffic_contribution or 0) == -34, "流量贡献为 -34")
    result.check(round(comparison.conversion_contribution or 0) == -54, "转化贡献为 -54")
    return result


def assert_outcome(
    *,
    case_id: str,
    outcome: RunOutcome,
    expected_status: str,
    required_text: list[str],
    forbidden_text: list[str],
) -> EvaluationResult:
    result = EvaluationResult(case_id=case_id)
    result.check(outcome.status.value == expected_status, f"状态为 {expected_status}")
    rendered = outcome.result or ""
    for text in required_text:
        result.check(text in rendered, f"结果包含：{text}")
    for text in forbidden_text:
        result.check(text not in rendered, f"结果不包含：{text}")
    return result


def merge_results(*results: EvaluationResult) -> EvaluationResult:
    merged = EvaluationResult(case_id=results[0].case_id)
    for result in results:
        merged.passed_checks.extend(result.passed_checks)
        merged.failed_checks.extend(result.failed_checks)
    return merged
