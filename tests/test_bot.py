"""Unit tests for message splitting, GeminiChatService, bot streaming, and health server."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.ai_service import GeminiChatService, GeminiNotConfiguredError
from src.bot import (
    DiscordGeminiBot,
    build_stream_preview,
    format_uptime,
    make_session_key,
    strip_bot_mention,
)
from src.config import BotConfig
from src.health import create_health_app
from src.utils import split_message


class FakeAsyncInteractions:
    """In-memory stub for `client.aio.interactions` supporting sync & streaming."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self._counter = 0

    async def create(self, **kwargs):
        self._counter += 1
        self.calls.append(kwargs)
        interaction_id = f"inter_{self._counter}"

        if kwargs.get("stream"):
            prompt = kwargs["input"]

            async def event_stream():
                yield SimpleNamespace(
                    event_type="interaction.created",
                    interaction=SimpleNamespace(id=interaction_id),
                )
                yield SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="thought_summary", text="thinking..."),
                )
                yield SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text=f"Stream #{self._counter}: "),
                )
                yield SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text=prompt),
                )
                yield SimpleNamespace(
                    event_type="interaction.completed",
                    interaction=SimpleNamespace(id=interaction_id),
                )

            return event_stream()

        return SimpleNamespace(
            id=interaction_id,
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


def test_build_stream_preview_caps_length() -> None:
    long_text = "word " * 600
    preview = build_stream_preview(long_text, finished=False)
    finished = build_stream_preview(long_text, finished=True)
    assert len(preview) <= 2000
    assert preview.endswith(" ...")
    assert len(finished) <= 1880
    assert not finished.endswith(" ...")


@pytest.mark.asyncio
async def test_gemini_chat_service_stateful_and_reset() -> None:
    fake_client = FakeGenAIClient()
    service = GeminiChatService(
        api_key=None,
        model="gemini-4-hydrogen",
        system_instruction="Be helpful.",
        thinking_level="low",
        client=fake_client,
    )
    key = make_session_key(10, 20)

    first = await service.ask(key, "Chào bạn")
    second = await service.ask(key, "Tên tôi là An")

    assert first == "Reply #1: Chào bạn"
    assert second == "Reply #2: Tên tôi là An"
    assert fake_client.aio.interactions.calls[0]["system_instruction"] == "Be helpful."
    assert fake_client.aio.interactions.calls[0]["generation_config"] == {"thinking_level": "low"}
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
async def test_gemini_chat_service_streaming_tracks_session_and_ignores_thoughts() -> None:
    fake_client = FakeGenAIClient()
    service = GeminiChatService(
        api_key=None,
        model="gemini-4-hydrogen",
        system_instruction="Be fast.",
        thinking_level="low",
        client=fake_client,
    )
    key = make_session_key(99, 77)

    chunks = [chunk async for chunk in service.stream_ask(key, "Xin chào")]
    assert "".join(chunks) == "Stream #1: Xin chào"
    assert fake_client.aio.interactions.calls[0]["stream"] is True
    assert fake_client.aio.interactions.calls[0]["generation_config"] == {"thinking_level": "low"}

    follow_up = [chunk async for chunk in service.stream_ask(key, "Tiếp tục")]
    assert "".join(follow_up) == "Stream #2: Tiếp tục"
    assert fake_client.aio.interactions.calls[1]["previous_interaction_id"] == "inter_1"


@pytest.mark.asyncio
async def test_bot_stream_reply_edits_initial_and_sends_final_chunks() -> None:
    fake_client = FakeGenAIClient()
    config = BotConfig(
        discord_token="dummy",
        gemini_api_key=None,
        gemini_model="gemini-4-hydrogen",
        thinking_level="low",
        system_instruction="Be helpful.",
        guild_id=None,
        port=8080,
    )
    service = GeminiChatService(
        api_key=None,
        model=config.gemini_model,
        system_instruction=config.system_instruction,
        thinking_level=config.thinking_level,
        client=fake_client,
    )
    bot = DiscordGeminiBot(config=config, ai_service=service)

    events: list[tuple[str, str]] = []

    async def send_initial(text: str):
        events.append(("initial", text))
        return "msg-1"

    async def edit_message(msg, text: str):
        assert msg == "msg-1"
        events.append(("edit", text))

    async def send_extra(text: str):
        events.append(("extra", text))

    await bot._stream_reply(
        session_key="1:2",
        prompt="Test",
        send_initial=send_initial,
        edit_message=edit_message,
        send_extra=send_extra,
    )

    assert events[0][0] == "initial"
    assert events[-1] == ("edit", "Stream #1: Test")


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
