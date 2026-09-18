import os
from pathlib import Path

from mcp.server import MCPServer

from backend_agent.commerce import CommerceRepository


mcp = MCPServer("e-commerce-product-context")


def _repository() -> CommerceRepository:
    return CommerceRepository(Path(os.getenv("COMMERCE_DB_PATH", "data/commerce.db")))


@mcp.tool()
def get_product_context(
    merchant_id: str,
    product_id: str,
    sections: list[str] | None = None,
) -> dict[str, object]:
    """读取模拟商品的详情、库存、价格、评论和竞品快照。"""
    if not merchant_id.strip() or not product_id.strip():
        raise ValueError("merchant_id 和 product_id 不能为空")
    allowed = {"product", "inventory", "price", "reviews", "competitors"}
    selected = sections or sorted(allowed)
    if not set(selected).issubset(allowed):
        raise ValueError("sections 包含不支持的字段")
    return _repository().get_product_context_sync(merchant_id, product_id, selected)


if __name__ == "__main__":
    _repository().initialize_sync()
    mcp.run(
        transport="streamable-http",
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "8000")),
    )
