from backend.app.config import get_settings

settings = get_settings()

try:
    from celery import Celery

    celery_app = Celery(
        "agentic_chemrag",
        broker=settings.redis_url,
        backend=settings.redis_url,
        include=["backend.app.tasks.ingestion_tasks"],
    )
    celery_app.conf.update(
        task_track_started=True,
        task_soft_time_limit=1700,
        task_time_limit=1800,
        result_expires=86400,
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        worker_prefetch_multiplier=1,
    )
except ModuleNotFoundError:
    from dataclasses import dataclass
    from uuid import uuid4

    @dataclass
    class InlineResult:
        id: str
        status: str
        result: object = None

        def successful(self) -> bool:
            return self.status == "SUCCESS"

        def failed(self) -> bool:
            return self.status == "FAILURE"

    class InlineTask:
        _results: dict[str, InlineResult] = {}

        def __init__(self, fn):
            self.fn = fn
            self.__name__ = fn.__name__

        def __call__(self, *args, **kwargs):
            return self.fn(*args, **kwargs)

        def delay(self, *args, **kwargs) -> InlineResult:
            task_id = uuid4().hex
            try:
                result = self.fn(*args, **kwargs)
                task_result = InlineResult(task_id, "SUCCESS", result)
            except Exception as exc:
                task_result = InlineResult(task_id, "FAILURE", exc)
            self._results[task_id] = task_result
            return task_result

        def AsyncResult(self, task_id: str) -> InlineResult:
            return self._results.get(task_id, InlineResult(task_id, "PENDING"))

    class InlineCelery:
        def task(self, name: str | None = None):
            def decorator(fn):
                return InlineTask(fn)

            return decorator

    celery_app = InlineCelery()
