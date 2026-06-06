from collections import defaultdict
from time import perf_counter

from starlette.requests import Request


_counters: dict[str, int] = defaultdict(int)
_durations: dict[str, list[float]] = defaultdict(list)


def increment(name: str, amount: int = 1) -> None:
    _counters[name] += amount


def observe_duration(name: str, value_seconds: float) -> None:
    _durations[name].append(value_seconds)


def render_metrics() -> str:
    lines: list[str] = []
    for name, value in sorted(_counters.items()):
        lines.append(f"{name} {value}")
    for name, values in sorted(_durations.items()):
        if not values:
            continue
        lines.append(f"{name}_count {len(values)}")
        lines.append(f"{name}_sum {sum(values):.6f}")
    return "\n".join(lines) + "\n"


class MetricsMiddleware:
    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started_at = perf_counter()
        status_code = 500

        async def send_with_metrics(message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_with_metrics)
        finally:
            request = Request(scope)
            route = request.url.path.replace("/", "_").strip("_") or "root"
            increment(f"http_requests_total{{path=\"{route}\",status=\"{status_code}\"}}")
            observe_duration("http_request_duration_seconds", perf_counter() - started_at)
