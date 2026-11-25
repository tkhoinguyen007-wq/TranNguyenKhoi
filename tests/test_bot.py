"""Unit tests for message splitting, GeminiChatService, bot helpers, and health server."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.ai_service import GeminiChatService, GeminiNotConfiguredError
from src.bot import format_uptime, make_session_key, strip_bot_mention
from src.health import create_health_app
from src.utils import split_message


class FakeAsyncInteractions:
    """In-memory stub for `client.aio.interactions`."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._counter = 0

    async def create(self, **kwargs):
        self._counter += 1
        self.calls.append(kwargs)
        return SimpleNamespace(
            id=f"inter_{self._counter}",
            output_text=f"Reply #{self._counter}: {kwargs['input']}",
        )


class FakeGenAIClient:
    """Minimal fake matching `client.aio.interactions.create`."""

    def __init__(self) -> None:
        self.aio = SimpleNamespace(interactions=FakeAsyncInteractions())


def test_split_message_short_and_empty() -> None:
    assert split_message("   ") == []
    assert split_message("Xin chào") == ["Xin chào"]


def test_split_message_long_preserves_words() -> None:
    paragraph = " ".join(["hello"] * 600)
    chunks = split_message(paragraph, max_length=1000)
    assert len(chunks) > 1
    assert all(len(chunk) <= 1000 for chunk in chunks)
    assert " ".join(chunks) == paragraph


def test_strip_bot_mention_handles_nick_and_standard_tags() -> None:
    assert strip_bot_mention("<@123456> hỏi cái này", 123456) == "hỏi cái này"
    assert strip_bot_mention("<@!123456>   hello", 123456) == "hello"
    assert strip_bot_mention("<@123456>", 123456) == ""


def test_format_uptime() -> None:
    start = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    end = start + timedelta(days=1, hours=2, minutes=3, seconds=4)
    assert format_uptime(start, end) == "1d 2h 3m 4s"


@pytest.mark.asyncio
async def test_gemini_chat_service_stateful_and_reset() -> None:
    fake_client = FakeGenAIClient()
    service = GeminiChatService(
        api_key=None,
        model="gemini-4-hydrogen",
        system_instruction="Be helpful.",
        client=fake_client,
    )
    key = make_session_key(10, 20)

    first = await service.ask(key, "Chào bạn")
    second = await service.ask(key, "Tên tôi là An")

    assert first == "Reply #1: Chào bạn"
    assert second == "Reply #2: Tên tôi là An"
    assert fake_client.aio.interactions.calls[0]["system_instruction"] == "Be helpful."
    assert "previous_interaction_id" not in fake_client.aio.interactions.calls[0]
    assert fake_client.aio.interactions.calls[1]["system_instruction"] == "Be helpful."
    assert fake_client.aio.interactions.calls[1]["previous_interaction_id"] == "inter_1"
    assert service.active_session_count() == 1

    assert service.reset_session(key) is True
    assert service.reset_session(key) is False
    assert service.active_session_count() == 0

    await service.ask(key, "Câu mới")
    assert "previous_interaction_id" not in fake_client.aio.interactions.calls[2]


@pytest.mark.asyncio
async def test_gemini_chat_service_raises_when_unconfigured() -> None:
    service = GeminiChatService(api_key=None)
    assert service.is_configured is False
    with pytest.raises(GeminiNotConfiguredError):
        await service.ask("1:2", "hello")


@pytest.mark.asyncio
async def test_health_endpoints_return_ok() -> None:
    app = create_health_app(lambda: {"status": "ok", "bot_ready": True, "guild_count": 2})
    async with TestClient(TestServer(app)) as client:
        for path in ("/", "/health"):
            resp = await client.get(path)
            assert resp.status == 200
            data = await resp.json()
            assert data == {"status": "ok", "bot_ready": True, "guild_count": 2}
