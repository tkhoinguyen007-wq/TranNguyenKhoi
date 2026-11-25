"""Unit tests for message splitting, GeminiChatService, multimodal images, and health server."""

from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer

from src.ai_service import DEFAULT_IMAGE_PROMPT, GeminiChatService, GeminiNotConfiguredError, ImageInput
from src.bot import (
    DiscordGeminiBot,
    MAX_IMAGE_BYTES,
    build_stream_preview,
    format_uptime,
    make_session_key,
    resolve_image_mime_type,
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
        raw_input = kwargs["input"]
        prompt_summary = (
            raw_input[0]["text"] if isinstance(raw_input, list) else raw_input
        )

        if kwargs.get("stream"):
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
                    delta=SimpleNamespace(type="text", text=prompt_summary),
                )
                yield SimpleNamespace(
                    event_type="interaction.completed",
                    interaction=SimpleNamespace(id=interaction_id),
                )

            return event_stream()

        return SimpleNamespace(
            id=interaction_id,
            output_text=f"Reply #{self._counter}: {prompt_summary}",
        )


class FakeGenAIClient:
    """Minimal fake matching `client.aio.interactions.create`."""

    def __init__(self) -> None:
        self.aio = SimpleNamespace(interactions=FakeAsyncInteractions())


class FakeAttachment:
    """Minimal stub for `discord.Attachment`."""

    def __init__(
        self,
        *,
        filename: str,
        content_type: str | None,
        payload: bytes,
        size: int | None = None,
    ) -> None:
        self.filename = filename
        self.content_type = content_type
        self._payload = payload
        self.size = len(payload) if size is None else size

    async def read(self) -> bytes:
        return self._payload


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


def test_resolve_image_mime_type() -> None:
    assert resolve_image_mime_type("image/png; charset=utf-8", "a.bin") == "image/png"
    assert resolve_image_mime_type("image/jpg", "photo.jpg") == "image/jpeg"
    assert resolve_image_mime_type(None, "screenshot.WEBP") == "image/webp"
    assert resolve_image_mime_type("application/pdf", "doc.pdf") is None


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
        model="gemini-3.5-flash-lite",
        system_instruction="Be helpful.",
        thinking_level="minimal",
        client=fake_client,
    )
    key = make_session_key(10, 20)

    first = await service.ask(key, "Chào bạn")
    second = await service.ask(key, "Tên tôi là An")

    assert first == "Reply #1: Chào bạn"
    assert second == "Reply #2: Tên tôi là An"
    assert fake_client.aio.interactions.calls[0]["system_instruction"] == "Be helpful."
    assert fake_client.aio.interactions.calls[0]["generation_config"] == {"thinking_level": "minimal"}
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
async def test_gemini_chat_service_multimodal_images_and_default_prompt() -> None:
    fake_client = FakeGenAIClient()
    service = GeminiChatService(
        api_key=None,
        model="gemini-3.5-flash-lite",
        thinking_level="minimal",
        client=fake_client,
    )
    img = ImageInput.from_bytes(b"png-bytes", "image/png")

    chunks = [chunk async for chunk in service.stream_ask("img:1", "", images=[img])]
    assert "".join(chunks) == f"Stream #1: {DEFAULT_IMAGE_PROMPT}"

    sent_input = fake_client.aio.interactions.calls[0]["input"]
    assert isinstance(sent_input, list)
    assert sent_input[0] == {"type": "text", "text": DEFAULT_IMAGE_PROMPT}
    assert sent_input[1] == {
        "type": "image",
        "data": base64.b64encode(b"png-bytes").decode("ascii"),
        "mime_type": "image/png",
    }


@pytest.mark.asyncio
async def test_bot_collect_message_images_reads_message_and_reply() -> None:
    config = BotConfig(
        discord_token="dummy",
        gemini_api_key=None,
        gemini_model="gemini-3.5-flash-lite",
        fallback_model="gemini-3.5-flash-lite",
        thinking_level="minimal",
        system_instruction="Be helpful.",
        guild_id=None,
        port=8080,
    )
    bot = DiscordGeminiBot(config=config, ai_service=GeminiChatService(api_key=None))

    replied_msg = SimpleNamespace(
        attachments=[
            FakeAttachment(filename="ref.jpg", content_type="image/jpeg", payload=b"ref-img"),
            FakeAttachment(filename="notes.txt", content_type="text/plain", payload=b"ignore"),
        ]
    )
    async def fetch_message(_msg_id: int):
        return replied_msg

    msg = SimpleNamespace(
        attachments=[
            FakeAttachment(filename="own.png", content_type="image/png", payload=b"own-img"),
            FakeAttachment(
                filename="too_big.png",
                content_type="image/png",
                payload=b"x",
                size=MAX_IMAGE_BYTES + 1,
            ),
        ],
        reference=SimpleNamespace(message_id=555, resolved=None),
        channel=SimpleNamespace(fetch_message=fetch_message),
    )

    collected = await bot._collect_message_images(msg)
    assert len(collected) == 2
    assert collected[0].mime_type == "image/png"
    assert base64.b64decode(collected[0].data_base64) == b"own-img"
    assert collected[1].mime_type == "image/jpeg"
    assert base64.b64decode(collected[1].data_base64) == b"ref-img"


@pytest.mark.asyncio
async def test_bot_stream_reply_edits_initial_and_sends_final_chunks() -> None:
    fake_client = FakeGenAIClient()
    config = BotConfig(
        discord_token="dummy",
        gemini_api_key=None,
        gemini_model="gemini-3.5-flash-lite",
        fallback_model="gemini-3.5-flash-lite",
        thinking_level="minimal",
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
async def test_gemini_chat_service_falls_back_on_primary_error() -> None:
    class FailingPrimaryInteractions:
        def __init__(self) -> None:
            self.models_tried: list[str] = []

        async def create(self, **kwargs):
            self.models_tried.append(kwargs["model"])
            if kwargs["model"] == "gemini-4-hydrogen":
                raise TimeoutError("simulated timeout")

            async def ok_stream():
                yield SimpleNamespace(
                    event_type="step.delta",
                    delta=SimpleNamespace(type="text", text="Fallback OK"),
                )

            return ok_stream()

    failing_interactions = FailingPrimaryInteractions()
    client = SimpleNamespace(aio=SimpleNamespace(interactions=failing_interactions))
    service = GeminiChatService(
        api_key=None,
        model="gemini-4-hydrogen",
        fallback_model="gemini-3.5-flash-lite",
        thinking_level="low",
        client=client,
    )

    chunks = [chunk async for chunk in service.stream_ask("fb:1", "hello")]
    assert "".join(chunks) == "Fallback OK"
    assert failing_interactions.models_tried == ["gemini-4-hydrogen", "gemini-3.5-flash-lite"]


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
