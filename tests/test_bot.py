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
    is_knowledge_manager,
    make_session_key,
    parse_dm_knowledge_command,
    resolve_image_mime_type,
    strip_bot_mention,
)
from src.config import BotConfig
from src.health import create_health_app
from src.knowledge import KnowledgeStore
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
async def test_knowledge_store_persists_and_injects_into_gemini(tmp_path) -> None:
    store_path = tmp_path / "knowledge.json"
    store = KnowledgeStore(file_path=store_path)
    entry = await store.add("Shop mở cửa từ 8h đến 22h mỗi ngày.", author="_ngkhoi.9999")
    await store.add("Không nhận giao dịch ngoài server.", author="ModA")

    reloaded = KnowledgeStore(file_path=store_path)
    entries = reloaded.list_entries()
    assert len(entries) == 2
    assert entries[0].id == entry.id
    assert "8h đến 22h" in reloaded.build_prompt_section()

    fake_client = FakeGenAIClient()
    service = GeminiChatService(
        api_key=None,
        model="gemini-3.5-flash-lite",
        system_instruction="Base prompt.",
        thinking_level="minimal",
        knowledge_provider=reloaded.build_prompt_section,
        client=fake_client,
    )
    await service.ask("k:1", "Mấy giờ mở cửa?")
    sent_instruction = fake_client.aio.interactions.calls[0]["system_instruction"]
    assert "Base prompt." in sent_instruction
    assert "Shop mở cửa từ 8h đến 22h mỗi ngày." in sent_instruction

    removed = await reloaded.remove(1)
    assert removed is not None
    assert len(reloaded.list_entries()) == 1
    assert await reloaded.clear() == 1
    assert reloaded.build_prompt_section() == ""


def test_is_knowledge_manager_role_and_username() -> None:
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

    owner = SimpleNamespace(name="_ngkhoi.9999", display_name="Khoi", roles=[])
    manager_member = SimpleNamespace(
        name="someone_else",
        display_name="Someone",
        roles=[SimpleNamespace(name="  Người quản lí ")],
    )
    impersonator = SimpleNamespace(
        name="random_user",
        display_name="_ngkhoi.9999",
        roles=[SimpleNamespace(name="Member")],
    )
    dm_stranger = SimpleNamespace(name="stranger", display_name="Stranger")

    assert is_knowledge_manager(owner, config) is True
    assert is_knowledge_manager(manager_member, config) is True
    assert is_knowledge_manager(impersonator, config) is False
    assert is_knowledge_manager(dm_stranger, config) is False


def test_parse_dm_knowledge_command() -> None:
    assert parse_dm_knowledge_command("dạy: Giá vé VIP là 50k") == ("add", "Giá vé VIP là 50k")
    assert parse_dm_knowledge_command("Nhớ: Link web là example.com") == (
        "add",
        "Link web là example.com",
    )
    assert parse_dm_knowledge_command("cập nhật: Bảo trì lúc 2h sáng") == (
        "add",
        "Bảo trì lúc 2h sáng",
    )
    assert parse_dm_knowledge_command("!teach Quy định mới") == ("add", "Quy định mới")
    assert parse_dm_knowledge_command("xem kiến thức") == ("list", "")
    assert parse_dm_knowledge_command("!knowledge") == ("list", "")
    assert parse_dm_knowledge_command("quên: 2") == ("remove", "2")
    assert parse_dm_knowledge_command("!forget all") == ("remove", "all")
    assert parse_dm_knowledge_command("Hôm nay trời đẹp quá") is None


@pytest.mark.asyncio
async def test_bot_execute_knowledge_command(tmp_path) -> None:
    config = BotConfig(
        discord_token="dummy",
        gemini_api_key=None,
        gemini_model="gemini-3.5-flash-lite",
        fallback_model="gemini-3.5-flash-lite",
        thinking_level="minimal",
        system_instruction="Be helpful.",
        guild_id=None,
        port=8080,
        knowledge_file=tmp_path / "bot_knowledge.json",
    )
    bot = DiscordGeminiBot(config=config)

    add_msg = await bot._execute_knowledge_command("add", "Luật 1: Tôn trọng nhau", author="_ngkhoi.9999")
    assert "#1" in add_msg
    list_msg = await bot._execute_knowledge_command("list", "", author="_ngkhoi.9999")
    assert "Luật 1: Tôn trọng nhau" in list_msg
    del_msg = await bot._execute_knowledge_command("remove", "1", author="_ngkhoi.9999")
    assert "Đã xóa kiến thức" in del_msg


@pytest.mark.asyncio
async def test_health_endpoints_return_ok() -> None:
    app = create_health_app(lambda: {"status": "ok", "bot_ready": True, "guild_count": 2})
    async with TestClient(TestServer(app)) as client:
        for path in ("/", "/health"):
            resp = await client.get(path)
            assert resp.status == 200
            data = await resp.json()
            assert data == {"status": "ok", "bot_ready": True, "guild_count": 2}
