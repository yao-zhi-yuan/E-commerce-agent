class AppError(Exception):
    def __init__(self, code: str, message: str, *, http_status: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


class TaskNotFoundError(AppError):
    def __init__(self, task_id: str) -> None:
        super().__init__("TASK_NOT_FOUND", f"任务不存在：{task_id}", http_status=404)


class TaskNotResumableError(AppError):
    def __init__(self, task_id: str) -> None:
        super().__init__("TASK_NOT_RESUMABLE", f"任务不可续跑：{task_id}", http_status=409)


class TaskStateConflictError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__("TASK_STATE_CONFLICT", message, http_status=409)


class MerchantForbiddenError(AppError):
    def __init__(self) -> None:
        super().__init__("MERCHANT_FORBIDDEN", "不能访问其他商家的任务", http_status=403)


class AgentLoopDetectedError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__("AGENT_LOOP_DETECTED", message, http_status=500)


class DependencyError(AppError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, http_status=503)
