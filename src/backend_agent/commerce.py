import asyncio
import hashlib
import json
import secrets
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from backend_agent.domain import (
    ActionPlan,
    ActionReceipt,
    ApprovalDecision,
    FunnelComparison,
    FunnelWindow,
    ProductPatch,
)
from backend_agent.errors import AppError


DEMO_MERCHANT_ID = "demo-merchant"
DEMO_PRODUCT_ID = "wireless-headphones"
SECOND_PRODUCT_ID = "portable-speaker"


def _rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def compare_funnel(
    *,
    comparison_impressions: int,
    comparison_clicks: int,
    comparison_orders: int,
    current_impressions: int,
    current_clicks: int,
    current_orders: int,
) -> FunnelComparison:
    comparison_conversion = _rate(comparison_orders, comparison_clicks)
    current_conversion = _rate(current_orders, current_clicks)
    traffic_contribution = None
    conversion_contribution = None
    if comparison_conversion is not None and current_conversion is not None:
        traffic_contribution = (current_clicks - comparison_clicks) * (
            comparison_conversion + current_conversion
        ) / 2
        conversion_contribution = (current_conversion - comparison_conversion) * (
            comparison_clicks + current_clicks
        ) / 2
    return FunnelComparison(
        comparison=FunnelWindow(
            impressions=comparison_impressions,
            clicks=comparison_clicks,
            paid_orders=comparison_orders,
            click_through_rate=_rate(comparison_clicks, comparison_impressions),
            conversion_rate=comparison_conversion,
        ),
        current=FunnelWindow(
            impressions=current_impressions,
            clicks=current_clicks,
            paid_orders=current_orders,
            click_through_rate=_rate(current_clicks, current_impressions),
            conversion_rate=current_conversion,
        ),
        order_change=current_orders - comparison_orders,
        order_change_rate=_rate(current_orders - comparison_orders, comparison_orders),
        traffic_contribution=traffic_contribution,
        conversion_contribution=conversion_contribution,
    )


class CommerceRepository:
    def __init__(self, database_path: Path) -> None:
        self._database_path = database_path

    async def initialize(self) -> None:
        await asyncio.to_thread(self._initialize_sync)

    def initialize_sync(self) -> None:
        self._initialize_sync()

    async def resolve_product(self, merchant_id: str, query: str) -> list[dict[str, object]]:
        return await asyncio.to_thread(self._resolve_product_sync, merchant_id, query)

    async def get_metrics(self, merchant_id: str, product_id: str) -> dict[str, object]:
        return await asyncio.to_thread(self._get_metrics_sync, merchant_id, product_id)

    async def get_product_context(
        self,
        merchant_id: str,
        product_id: str,
        sections: list[str] | None = None,
    ) -> dict[str, object]:
        return await asyncio.to_thread(
            self._get_product_context_sync, merchant_id, product_id, sections
        )

    async def create_plan(
        self,
        *,
        task_id: str,
        merchant_id: str,
        product_id: str,
        patch: ProductPatch,
    ) -> ActionPlan:
        return await asyncio.to_thread(
            self._create_plan_sync, task_id, merchant_id, product_id, patch
        )

    async def get_plan(self, plan_id: str) -> ActionPlan | None:
        return await asyncio.to_thread(self._get_plan_sync, plan_id)

    async def record_approval(
        self,
        *,
        task_id: str,
        merchant_id: str,
        plan_id: str,
        plan_version: int,
        action_id: str,
        decision: ApprovalDecision,
    ) -> str:
        return await asyncio.to_thread(
            self._record_approval_sync,
            task_id,
            merchant_id,
            plan_id,
            plan_version,
            action_id,
            decision,
        )

    async def get_approval(self, approval_id: str) -> dict[str, object] | None:
        return await asyncio.to_thread(self._get_approval_sync, approval_id)

    async def get_approval_for_task(self, task_id: str) -> dict[str, object] | None:
        return await asyncio.to_thread(self._get_approval_for_task_sync, task_id)

    async def apply_approved_plan(self, approval_id: str) -> ActionReceipt:
        return await asyncio.to_thread(self._apply_approved_plan_sync, approval_id)

    def get_product_context_sync(
        self,
        merchant_id: str,
        product_id: str,
        sections: list[str] | None = None,
    ) -> dict[str, object]:
        return self._get_product_context_sync(merchant_id, product_id, sections)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize_sync(self) -> None:
        self._database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS merchants (
                    merchant_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    timezone TEXT NOT NULL,
                    currency TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS products (
                    product_id TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    FOREIGN KEY(merchant_id) REFERENCES merchants(merchant_id)
                );
                CREATE TABLE IF NOT EXISTS metric_windows (
                    merchant_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    window_name TEXT NOT NULL,
                    impressions INTEGER NOT NULL,
                    clicks INTEGER NOT NULL,
                    paid_orders INTEGER NOT NULL,
                    channel_json TEXT NOT NULL,
                    PRIMARY KEY(merchant_id, product_id, window_name)
                );
                CREATE TABLE IF NOT EXISTS product_context (
                    merchant_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    inventory_json TEXT NOT NULL,
                    price_json TEXT NOT NULL,
                    reviews_json TEXT NOT NULL,
                    competitors_json TEXT NOT NULL,
                    PRIMARY KEY(merchant_id, product_id)
                );
                CREATE TABLE IF NOT EXISTS action_plans (
                    plan_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    merchant_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    action_id TEXT NOT NULL,
                    field TEXT NOT NULL,
                    before_value TEXT NOT NULL,
                    after_value TEXT NOT NULL,
                    product_version INTEGER NOT NULL,
                    reason TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    approval_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL UNIQUE,
                    merchant_id TEXT NOT NULL,
                    plan_id TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    action_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS action_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    merchant_id TEXT NOT NULL,
                    product_id TEXT NOT NULL,
                    action_id TEXT NOT NULL,
                    product_version INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            self._seed_sync(connection)

    def _seed_sync(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO merchants VALUES (?, ?, ?, ?)",
            (DEMO_MERCHANT_ID, "Demo Outdoor Store", "America/Los_Angeles", "USD"),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO products VALUES (?, ?, ?, ?, ?)",
            [
                (
                    DEMO_PRODUCT_ID,
                    DEMO_MERCHANT_ID,
                    "Wireless Noise Cancelling Headphones",
                    "Lightweight wireless headphones with active noise cancellation.",
                    1,
                ),
                (
                    SECOND_PRODUCT_ID,
                    DEMO_MERCHANT_ID,
                    "Portable Bluetooth Speaker",
                    "Compact speaker for travel and outdoor use.",
                    1,
                ),
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO metric_windows VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    DEMO_MERCHANT_ID,
                    DEMO_PRODUCT_ID,
                    "comparison",
                    100_000,
                    4_000,
                    200,
                    json.dumps({"search": 52_000, "recommendation": 48_000}),
                ),
                (
                    DEMO_MERCHANT_ID,
                    DEMO_PRODUCT_ID,
                    "current",
                    80_000,
                    3_200,
                    112,
                    json.dumps({"search": 32_000, "recommendation": 48_000}),
                ),
                (
                    DEMO_MERCHANT_ID,
                    SECOND_PRODUCT_ID,
                    "comparison",
                    30_000,
                    1_500,
                    90,
                    json.dumps({"search": 15_000, "recommendation": 15_000}),
                ),
                (
                    DEMO_MERCHANT_ID,
                    SECOND_PRODUCT_ID,
                    "current",
                    29_000,
                    1_450,
                    87,
                    json.dumps({"search": 14_000, "recommendation": 15_000}),
                ),
            ],
        )
        connection.execute(
            "INSERT OR IGNORE INTO product_context VALUES (?, ?, ?, ?, ?, ?)",
            (
                DEMO_MERCHANT_ID,
                DEMO_PRODUCT_ID,
                json.dumps(
                    {
                        "evidence_id": "inventory_primary_sku",
                        "primary_black": [42, 0, 0, 18, 36, 51, 67],
                        "secondary_white": [35, 33, 31, 30, 28, 25, 24],
                        "note": "The best-selling black SKU was out of stock for three days.",
                    }
                ),
                json.dumps(
                    {
                        "evidence_id": "price_unchanged",
                        "comparison": 79.99,
                        "current": 79.99,
                        "currency": "USD",
                    }
                ),
                json.dumps(
                    [
                        {
                            "evidence_id": "review_compatibility_theme",
                            "theme": "compatibility",
                            "count": 18,
                            "summary": "Buyers repeatedly ask whether it works with game consoles.",
                        }
                    ]
                ),
                json.dumps(
                    [
                        {
                            "evidence_id": "competitor_clear_compatibility",
                            "name": "Comparable Headset A",
                            "price": 82.0,
                            "observation": "The listing explicitly states supported devices.",
                        }
                    ]
                ),
            ),
        )
        connection.execute(
            "INSERT OR IGNORE INTO product_context VALUES (?, ?, ?, ?, ?, ?)",
            (
                DEMO_MERCHANT_ID,
                SECOND_PRODUCT_ID,
                json.dumps({"evidence_id": "speaker_inventory", "default": [60, 58, 55, 52, 50, 48, 45]}),
                json.dumps(
                    {
                        "evidence_id": "speaker_price_unchanged",
                        "comparison": 39.99,
                        "current": 39.99,
                        "currency": "USD",
                    }
                ),
                json.dumps([]),
                json.dumps([]),
            ),
        )

    def _resolve_product_sync(self, merchant_id: str, query: str) -> list[dict[str, object]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT product_id, name, version FROM products WHERE merchant_id = ? ORDER BY product_id",
                (merchant_id,),
            ).fetchall()
        if not rows:
            raise AppError("MERCHANT_NOT_FOUND", "商家不存在或无可访问商品", http_status=404)
        normalized = query.strip().lower()
        matches = [
            dict(row)
            for row in rows
            if normalized
            and (
                normalized in str(row["product_id"]).lower()
                or normalized in str(row["name"]).lower()
            )
        ]
        return matches or [dict(row) for row in rows]

    def _get_metrics_sync(self, merchant_id: str, product_id: str) -> dict[str, object]:
        with self._connect() as connection:
            product = connection.execute(
                "SELECT name FROM products WHERE merchant_id = ? AND product_id = ?",
                (merchant_id, product_id),
            ).fetchone()
            rows = connection.execute(
                """
                SELECT window_name, impressions, clicks, paid_orders, channel_json
                FROM metric_windows WHERE merchant_id = ? AND product_id = ?
                """,
                (merchant_id, product_id),
            ).fetchall()
        if product is None:
            raise AppError("PRODUCT_NOT_FOUND", "商品不存在或不属于当前商家", http_status=404)
        windows = {str(row["window_name"]): row for row in rows}
        if "comparison" not in windows or "current" not in windows:
            raise AppError("METRICS_INCOMPLETE", "商品对比窗口数据不完整", http_status=422)
        before = windows["comparison"]
        after = windows["current"]
        comparison = compare_funnel(
            comparison_impressions=int(before["impressions"]),
            comparison_clicks=int(before["clicks"]),
            comparison_orders=int(before["paid_orders"]),
            current_impressions=int(after["impressions"]),
            current_clicks=int(after["clicks"]),
            current_orders=int(after["paid_orders"]),
        )
        channels = {
            "comparison": json.loads(str(before["channel_json"])),
            "current": json.loads(str(after["channel_json"])),
        }
        identity = json.dumps(
            {"product_id": product_id, "comparison": comparison.model_dump(), "channels": channels},
            sort_keys=True,
        )
        return {
            "merchant_id": merchant_id,
            "product_id": product_id,
            "product_name": product["name"],
            "snapshot_id": hashlib.sha256(identity.encode()).hexdigest()[:16],
            "as_of": "2026-09-14T00:00:00Z",
            "metric_version": "paid_orders_funnel_v1",
            "timezone": "America/Los_Angeles",
            "comparison": comparison.model_dump(mode="json"),
            "channels": channels,
            "evidence_ids": ["metric_paid_orders", "metric_search_impressions"],
            "missing_fields": [],
        }

    def _get_product_context_sync(
        self,
        merchant_id: str,
        product_id: str,
        sections: list[str] | None,
    ) -> dict[str, object]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT p.name, p.description, p.version, c.inventory_json, c.price_json,
                       c.reviews_json, c.competitors_json
                FROM products p JOIN product_context c
                  ON c.merchant_id = p.merchant_id AND c.product_id = p.product_id
                WHERE p.merchant_id = ? AND p.product_id = ?
                """,
                (merchant_id, product_id),
            ).fetchone()
        if row is None:
            raise AppError("PRODUCT_NOT_FOUND", "商品不存在或不属于当前商家", http_status=404)
        requested = set(sections or ["product", "inventory", "price", "reviews", "competitors"])
        data: dict[str, object] = {}
        if "product" in requested:
            data["product"] = {
                "name": row["name"],
                "description": row["description"],
                "version": row["version"],
            }
        evidence_ids: list[str] = []
        mapping = {
            "inventory": "inventory_json",
            "price": "price_json",
            "reviews": "reviews_json",
            "competitors": "competitors_json",
        }
        for section, column in mapping.items():
            if section in requested:
                section_data = json.loads(str(row[column]))
                data[section] = section_data
                evidence_ids.extend(_extract_evidence_ids(section_data))
        return {
            "merchant_id": merchant_id,
            "product_id": product_id,
            "snapshot_id": f"context-{product_id}-v{row['version']}",
            "as_of": "2026-09-14T00:00:00Z",
            "source": "demo_commerce_db",
            "data": data,
            "evidence_ids": evidence_ids,
            "missing_fields": [],
        }

    def _create_plan_sync(
        self,
        task_id: str,
        merchant_id: str,
        product_id: str,
        patch: ProductPatch,
    ) -> ActionPlan:
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT * FROM action_plans WHERE task_id = ?", (task_id,)
            ).fetchone()
            if existing is not None:
                return self._plan_from_row(existing)
            product = connection.execute(
                "SELECT description, version FROM products WHERE merchant_id = ? AND product_id = ?",
                (merchant_id, product_id),
            ).fetchone()
            if product is None:
                raise AppError("PRODUCT_NOT_FOUND", "商品不存在或不属于当前商家", http_status=404)
            now = datetime.now(UTC)
            plan = ActionPlan(
                plan_id=secrets.token_urlsafe(18),
                task_id=task_id,
                merchant_id=merchant_id,
                product_id=product_id,
                plan_version=1,
                action_id="update_description",
                field=patch.field,
                before_value=str(product["description"]),
                after_value=patch.new_value,
                product_version=int(product["version"]),
                reason=patch.reason,
                created_at=now,
            )
            connection.execute(
                "INSERT INTO action_plans VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    plan.plan_id,
                    plan.task_id,
                    plan.merchant_id,
                    plan.product_id,
                    plan.plan_version,
                    plan.action_id,
                    plan.field,
                    plan.before_value,
                    plan.after_value,
                    plan.product_version,
                    plan.reason,
                    now.isoformat(),
                ),
            )
            return plan

    def _get_plan_sync(self, plan_id: str) -> ActionPlan | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM action_plans WHERE plan_id = ?", (plan_id,)
            ).fetchone()
        return None if row is None else self._plan_from_row(row)

    def _record_approval_sync(
        self,
        task_id: str,
        merchant_id: str,
        plan_id: str,
        plan_version: int,
        action_id: str,
        decision: ApprovalDecision,
    ) -> str:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            plan = connection.execute(
                "SELECT * FROM action_plans WHERE plan_id = ? AND task_id = ?",
                (plan_id, task_id),
            ).fetchone()
            if plan is None:
                raise AppError("PLAN_NOT_FOUND", "行动计划不存在", http_status=404)
            if plan["merchant_id"] != merchant_id:
                raise AppError("MERCHANT_FORBIDDEN", "不能批准其他商家的计划", http_status=403)
            if int(plan["plan_version"]) != plan_version or plan["action_id"] != action_id:
                raise AppError("PLAN_VERSION_CONFLICT", "计划版本或动作不匹配", http_status=409)
            existing = connection.execute(
                "SELECT approval_id, decision FROM approvals WHERE task_id = ?", (task_id,)
            ).fetchone()
            if existing is not None:
                if existing["decision"] != decision.value:
                    raise AppError("APPROVAL_CONFLICT", "任务已经提交不同审批决定", http_status=409)
                connection.commit()
                return str(existing["approval_id"])
            approval_id = secrets.token_urlsafe(18)
            connection.execute(
                "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    approval_id,
                    task_id,
                    merchant_id,
                    plan_id,
                    plan_version,
                    action_id,
                    decision.value,
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.commit()
            return approval_id
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _get_approval_sync(self, approval_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def _get_approval_for_task_sync(self, task_id: str) -> dict[str, object] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE task_id = ?", (task_id,)
            ).fetchone()
        return None if row is None else dict(row)

    def _apply_approved_plan_sync(self, approval_id: str) -> ActionReceipt:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            approval = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?", (approval_id,)
            ).fetchone()
            if approval is None:
                raise AppError("APPROVAL_NOT_FOUND", "审批记录不存在", http_status=404)
            if approval["decision"] != ApprovalDecision.APPROVE.value:
                raise AppError("ACTION_NOT_APPROVED", "计划未获批准，不能执行", http_status=409)
            plan = connection.execute(
                "SELECT * FROM action_plans WHERE plan_id = ?", (approval["plan_id"],)
            ).fetchone()
            if plan is None:
                raise AppError("PLAN_NOT_FOUND", "行动计划不存在", http_status=404)
            idempotency_key = ":".join(
                (
                    str(plan["merchant_id"]),
                    str(plan["task_id"]),
                    str(plan["plan_version"]),
                    str(plan["action_id"]),
                )
            )
            existing = connection.execute(
                "SELECT * FROM action_receipts WHERE idempotency_key = ?", (idempotency_key,)
            ).fetchone()
            if existing is not None:
                connection.commit()
                return self._receipt_from_row(existing, applied=False)
            product = connection.execute(
                "SELECT version FROM products WHERE merchant_id = ? AND product_id = ?",
                (plan["merchant_id"], plan["product_id"]),
            ).fetchone()
            if product is None:
                raise AppError("PRODUCT_NOT_FOUND", "商品不存在或不属于当前商家", http_status=404)
            if int(product["version"]) != int(plan["product_version"]):
                raise AppError("PRODUCT_VERSION_CONFLICT", "商品已变化，需要重新生成计划", http_status=409)
            if plan["field"] != "description":
                raise AppError("ACTION_FIELD_FORBIDDEN", "不允许修改该字段", http_status=403)
            new_version = int(product["version"]) + 1
            connection.execute(
                """
                UPDATE products SET description = ?, version = ?
                WHERE merchant_id = ? AND product_id = ? AND version = ?
                """,
                (
                    plan["after_value"],
                    new_version,
                    plan["merchant_id"],
                    plan["product_id"],
                    plan["product_version"],
                ),
            )
            receipt = ActionReceipt(
                receipt_id=secrets.token_urlsafe(18),
                idempotency_key=idempotency_key,
                merchant_id=str(plan["merchant_id"]),
                product_id=str(plan["product_id"]),
                action_id=str(plan["action_id"]),
                product_version=new_version,
                applied=True,
                created_at=datetime.now(UTC),
            )
            connection.execute(
                "INSERT INTO action_receipts VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    receipt.receipt_id,
                    receipt.idempotency_key,
                    receipt.merchant_id,
                    receipt.product_id,
                    receipt.action_id,
                    receipt.product_version,
                    receipt.created_at.isoformat(),
                ),
            )
            connection.commit()
            return receipt
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    @staticmethod
    def _plan_from_row(row: sqlite3.Row) -> ActionPlan:
        return ActionPlan(
            plan_id=str(row["plan_id"]),
            task_id=str(row["task_id"]),
            merchant_id=str(row["merchant_id"]),
            product_id=str(row["product_id"]),
            plan_version=int(row["plan_version"]),
            action_id=str(row["action_id"]),
            field=str(row["field"]),
            before_value=str(row["before_value"]),
            after_value=str(row["after_value"]),
            product_version=int(row["product_version"]),
            reason=str(row["reason"]),
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )

    @staticmethod
    def _receipt_from_row(row: sqlite3.Row, *, applied: bool) -> ActionReceipt:
        return ActionReceipt(
            receipt_id=str(row["receipt_id"]),
            idempotency_key=str(row["idempotency_key"]),
            merchant_id=str(row["merchant_id"]),
            product_id=str(row["product_id"]),
            action_id=str(row["action_id"]),
            product_version=int(row["product_version"]),
            applied=applied,
            created_at=datetime.fromisoformat(str(row["created_at"])),
        )


def _extract_evidence_ids(value: object) -> list[str]:
    if isinstance(value, dict):
        result = []
        evidence_id = value.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id:
            result.append(evidence_id)
        for nested in value.values():
            result.extend(_extract_evidence_ids(nested))
        return result
    if isinstance(value, list):
        result = []
        for nested in value:
            result.extend(_extract_evidence_ids(nested))
        return result
    return []
