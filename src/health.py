"""Lightweight HTTP health-check server for 24/7 PaaS hosting."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from aiohttp import web


StatusProvider = Callable[[], dict[str, Any]]


def create_health_app(status_provider: StatusProvider) -> web.Application:
    """Create the aiohttp web application exposing `/` and `/health`."""
    app = web.Application()

    async def handle_health(_request: web.Request) -> web.Response:
        payload = status_provider()
        return web.json_response(payload, status=200)

    app.router.add_get("/", handle_health)
    app.router.add_get("/health", handle_health)
    return app


async def start_health_server(
    port: int,
    status_provider: StatusProvider,
    host: str = "0.0.0.0",
) -> web.AppRunner:
    """Start the background HTTP health-check server and return its runner."""
    app = create_health_app(status_provider)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host=host, port=port)
    await site.start()
    return runner
