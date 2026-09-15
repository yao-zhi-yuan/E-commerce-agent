import hashlib
import os
from datetime import UTC, datetime

from mcp.server import MCPServer


mcp = MCPServer("backend-agent-operations")


@mcp.tool()
def get_service_health(service: str) -> dict[str, object]:
    """返回演示服务的确定性健康状态与依赖状态。"""
    normalized = service.strip().lower()
    if not normalized:
        raise ValueError("service 不能为空")
    bucket = int(hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:2], 16)
    status = "healthy" if bucket % 5 else "degraded"
    return {
        "service": normalized,
        "status": status,
        "checked_at": datetime.now(UTC).isoformat(),
        "dependencies": {
            "database": "healthy",
            "cache": "healthy" if bucket % 3 else "degraded",
            "downstream_rpc": "healthy" if bucket % 7 else "degraded",
        },
    }


@mcp.tool()
def get_incident_runbook(service: str) -> dict[str, object]:
    """返回服务故障排查步骤。"""
    normalized = service.strip().lower()
    if not normalized:
        raise ValueError("service 不能为空")
    return {
        "service": normalized,
        "steps": [
            "确认错误率、P99 延迟和影响范围",
            "检查最近发布、配置变更与依赖健康度",
            "按 trace_id 定位失败链路并核对超时预算",
            "必要时执行限流、降级或回滚，并记录恢复时间",
        ],
    }


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host=os.getenv("MCP_HOST", "0.0.0.0"),
        port=int(os.getenv("MCP_PORT", "8000")),
    )
