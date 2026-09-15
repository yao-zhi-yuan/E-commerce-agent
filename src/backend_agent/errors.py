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


class AgentLoopDetectedError(AppError):
    def __init__(self, message: str) -> None:
        super().__init__("AGENT_LOOP_DETECTED", message, http_status=500)


class DependencyError(AppError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(code, message, http_status=503)
