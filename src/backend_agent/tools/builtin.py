import ast
import operator
from collections.abc import Callable

from backend_agent.rag.knowledge_base import SQLiteKnowledgeBase
from backend_agent.tools.base import AgentTool, ToolDefinition


class RagSearchTool(AgentTool):
    def __init__(self, knowledge_base: SQLiteKnowledgeBase) -> None:
        self._knowledge_base = knowledge_base

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="rag_search",
            description="从内部知识库检索与问题相关的文档片段。",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索问题"},
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 8,
                        "default": 4,
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        query = str(arguments.get("query", "")).strip()
        if not query:
            raise ValueError("query 不能为空")
        limit_value = arguments.get("limit", 4)
        limit = min(max(int(limit_value), 1), 8)
        documents = await self._knowledge_base.search(query, limit=limit)
        return {
            "query": query,
            "documents": [document.model_dump(mode="json") for document in documents],
        }


class CalculatorTool(AgentTool):
    _binary_operators: dict[type[ast.operator], Callable[[float, float], float]] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    _unary_operators: dict[type[ast.unaryop], Callable[[float], float]] = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    @property
    def definition(self) -> ToolDefinition:
        return ToolDefinition(
            name="calculator",
            description="计算只包含数字、括号和基础算术运算符的表达式。",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "算术表达式"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, object]) -> dict[str, object]:
        expression = str(arguments.get("expression", "")).strip()
        if not expression or len(expression) > 256:
            raise ValueError("expression 不能为空且长度不能超过 256")
        tree = ast.parse(expression, mode="eval")
        result = self._evaluate(tree.body)
        return {"expression": expression, "result": result}

    def _evaluate(self, node: ast.expr) -> float:
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return float(node.value)
        if isinstance(node, ast.BinOp) and type(node.op) in self._binary_operators:
            left = self._evaluate(node.left)
            right = self._evaluate(node.right)
            if isinstance(node.op, ast.Pow) and (abs(left) > 1_000_000 or abs(right) > 12):
                raise ValueError("幂运算超出允许范围")
            result = self._binary_operators[type(node.op)](left, right)
            if not (-1e100 < result < 1e100):
                raise ValueError("计算结果超出允许范围")
            return result
        if isinstance(node, ast.UnaryOp) and type(node.op) in self._unary_operators:
            return self._unary_operators[type(node.op)](self._evaluate(node.operand))
        raise ValueError("表达式包含不支持的语法")
