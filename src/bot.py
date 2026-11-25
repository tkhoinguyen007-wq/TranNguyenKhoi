"""Discord bot implementation with slash commands, multimodal AI, and role/DM knowledge updates."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime, timezone
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from src.ai_service import GeminiChatService, GeminiNotConfiguredError, ImageInput
from src.config import BotConfig
from src.knowledge import KnowledgeStore
from src.utils import split_message


logger = logging.getLogger(__name__)

STREAM_EDIT_INTERVAL_SECONDS = 0.8
STREAM_TYPING_INDICATOR = " ..."
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB per image
MAX_IMAGES_PER_MESSAGE = 4

SUPPORTED_IMAGE_MIME_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
}

EXTENSION_TO_MIME = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
}

_DM_ADD_RE = re.compile(r"^(?:!teach|dạy:|nhớ:|cập nhật:)\s*(.*)$", re.IGNORECASE | re.DOTALL)
_DM_LIST_RE = re.compile(r"^(?:!knowledge|xem kiến thức|kiến thức)$", re.IGNORECASE)
_DM_REMOVE_RE = re.compile(r"^(?:!forget|quên:|xóa kiến thức)\s*(.*)$", re.IGNORECASE)


def make_session_key(channel_id: int, user_id: int) -> str:
    """Build a per-channel, per-user session key for stateful AI chat."""
    return f"{channel_id}:{user_id}"


def strip_bot_mention(content: str, bot_user_id: int) -> str:
    """Remove direct bot mention tags from a Discord message."""
    pattern = rf"<@!?{bot_user_id}>"
    return re.sub(pattern, "", content).strip()


def resolve_image_mime_type(content_type: str | None, filename: str | None) -> str | None:
    """Normalize and validate an attachment's image MIME type."""
    if content_type:
        base_type = content_type.split(";", 1)[0].strip().lower()
        if base_type == "image/jpg":
            base_type = "image/jpeg"
        if base_type in SUPPORTED_IMAGE_MIME_TYPES:
            return base_type

    if filename:
        lower_name = filename.lower()
        for ext, mime in EXTENSION_TO_MIME.items():
            if lower_name.endswith(ext):
                return mime

    return None


def is_knowledge_manager(user: Any, config: BotConfig) -> bool:
    """Return True if ``user`` is an authorized username or holds the manager role."""
    username = (getattr(user, "name", "") or "").strip().lower()
    allowed_usernames = {name.strip().lower() for name in config.manager_usernames if name.strip()}
    if username and username in allowed_usernames:
        return True

    target_role = config.manager_role_name.strip().lower()
    roles = getattr(user, "roles", None) or ()
    return any((getattr(role, "name", "") or "").strip().lower() == target_role for role in roles)


def parse_dm_knowledge_command(text: str) -> tuple[str, str] | None:
    """Parse DM live-update knowledge commands; return None for regular AI chat."""
    cleaned = text.strip()
    if not cleaned:
        return None

    if _DM_LIST_RE.match(cleaned):
        return ("list", "")

    add_match = _DM_ADD_RE.match(cleaned)
    if add_match:
        return ("add", add_match.group(1).strip())

    remove_match = _DM_REMOVE_RE.match(cleaned)
    if remove_match:
        return ("remove", remove_match.group(1).strip())

    return None


def format_uptime(started_at: datetime, now: datetime | None = None) -> str:
    """Format elapsed bot uptime as a readable string."""
    current = now or datetime.now(timezone.utc)
    total_seconds = max(0, int((current - started_at).total_seconds()))
    days, rem = divmod(total_seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)

    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours or parts:
        parts.append(f"{hours}h")
    if minutes or parts:
        parts.append(f"{minutes}m")
    parts.append(f"{seconds}s")
    return " ".join(parts)


def build_stream_preview(text: str, *, finished: bool = False) -> str:
    """Build a safe <=2000-character preview of the first message while streaming."""
    chunks = split_message(text, max_length=1880)
    if not chunks:
        return "(Mình không nhận được nội dung phản hồi từ mô hình.)" if finished else "..."

    first_chunk = chunks[0]
    if finished:
        return first_chunk
    return f"{first_chunk}{STREAM_TYPING_INDICATOR}"


class DiscordGeminiBot(commands.Bot):
    """Discord bot combining utility slash commands, multimodal AI, and managed knowledge."""

    def __init__(
        self,
        config: BotConfig,
        ai_service: GeminiChatService | None = None,
        knowledge_store: KnowledgeStore | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix="!", intents=intents, help_command=None)
        self.config = config
        self.started_at = datetime.now(timezone.utc)
        self.knowledge_store = knowledge_store or KnowledgeStore(
            file_path=config.knowledge_file,
            gist_id=config.knowledge_gist_id,
            github_token=config.knowledge_github_token,
        )
        self.ai_service = ai_service or GeminiChatService(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            system_instruction=config.system_instruction,
            thinking_level=config.thinking_level,
            fallback_model=config.fallback_model,
            knowledge_provider=self.knowledge_store.build_prompt_section,
        )
        self._commands_registered = False

    def get_status(self) -> dict[str, Any]:
        """Return runtime status payload for the HTTP health-check endpoint."""
        latency_ms = round(self.latency * 1000, 1) if self.is_ready() else None
        return {
            "status": "ok",
            "bot_ready": self.is_ready(),
            "bot_user": str(self.user) if self.user else None,
            "guild_count": len(self.guilds),
            "latency_ms": latency_ms,
            "uptime": format_uptime(self.started_at),
            "ai_enabled": self.ai_service.is_configured,
            "model": self.config.gemini_model,
            "thinking_level": self.config.thinking_level or "default",
            "knowledge_count": len(self.knowledge_store.list_entries()),
        }

    async def setup_hook(self) -> None:
        """Load persisted knowledge and sync application slash commands."""
        await self.knowledge_store.async_load()

        if not self._commands_registered:
            self._register_commands()
            self._commands_registered = True

        if self.config.guild_id is not None:
            guild = discord.Object(id=self.config.guild_id)
            self.tree.copy_global_to(guild=guild)
            synced = await self.tree.sync(guild=guild)
            logger.info("Synced %d slash commands to guild %s", len(synced), self.config.guild_id)
        else:
            synced = await self.tree.sync()
            logger.info("Synced %d global slash commands", len(synced))

    async def on_ready(self) -> None:
        """Log connection status and set bot presence."""
        logger.info(
            "Bot online as %s (ID: %s) in %d server(s)",
            self.user,
            self.user.id if self.user else "?",
            len(self.guilds),
        )
        activity = discord.Activity(
            type=discord.ActivityType.listening,
            name="/ask & @mention",
        )
        await self.change_presence(status=discord.Status.online, activity=activity)

    def _permission_denied_message(self) -> str:
        managers = ", ".join(self.config.manager_usernames) or "_ngkhoi.9999"
        return (
            f"Bạn không có quyền cập nhật kiến thức cho bot. "
            f"Cần có Role **`{self.config.manager_role_name}`** trong server "
            f"hoặc là tài khoản **`{managers}`**."
        )

    async def _execute_knowledge_command(
        self,
        action: str,
        arg: str,
        *,
        author: str,
    ) -> str:
        """Run add/list/remove knowledge actions shared by slash commands and DM updates."""
        if action == "add":
            if not arg.strip():
                return "Hãy nhập nội dung kiến thức cần dạy cho bot (ví dụ: `dạy: Giờ mở cửa là 8h-22h`)."
            entry = await self.knowledge_store.add(arg, author=author)
            total = len(self.knowledge_store.list_entries())
            return (
                f"Đã tiếp thu kiến thức **#{entry.id}** (tổng cộng đang nhớ **{total}** mục):\n"
                f"> {entry.content}"
            )

        if action == "list":
            entries = self.knowledge_store.list_entries()
            if not entries:
                return "Hiện bot chưa được dạy kiến thức bổ sung nào."
            lines = [f"**Kho kiến thức của bot ({len(entries)} mục):**"]
            for entry in entries:
                lines.append(f"- **#{entry.id}**: {entry.content} *(bởi `{entry.author}`)*")
            return "\n".join(lines)

        if action == "remove":
            cleaned = arg.strip().lower()
            if not cleaned:
                return "Hãy nhập số ID cần xóa (vd: `1`) hoặc `all` để xóa toàn bộ."
            if cleaned in ("all", "tất cả"):
                cleared = await self.knowledge_store.clear()
                return (
                    f"Đã xóa toàn bộ **{cleared}** kiến thức đã lưu."
                    if cleared
                    else "Kho kiến thức hiện đang trống."
                )
            try:
                entry_id = int(cleaned.lstrip("#"))
            except ValueError:
                return "ID không hợp lệ. Hãy nhập số thứ tự (vd: `1`) hoặc `all`."

            removed = await self.knowledge_store.remove(entry_id)
            if removed is None:
                return f"Không tìm thấy kiến thức **#{entry_id}**."
            return f"Đã xóa kiến thức **#{removed.id}**: {removed.content}"

        return "Lệnh kiến thức không hợp lệ."

    async def _extract_images(
        self,
        attachments: Sequence[discord.Attachment],
        *,
        limit: int = MAX_IMAGES_PER_MESSAGE,
    ) -> list[ImageInput]:
        """Download supported image attachments up to ``limit`` and ``MAX_IMAGE_BYTES``."""
        images: list[ImageInput] = []
        for attachment in attachments:
            if len(images) >= limit:
                break
            mime_type = resolve_image_mime_type(attachment.content_type, attachment.filename)
            if not mime_type:
                continue
            if attachment.size > MAX_IMAGE_BYTES:
                logger.info(
                    "Skipping oversized image attachment %s (%d bytes)",
                    attachment.filename,
                    attachment.size,
                )
                continue
            try:
                raw_bytes = await attachment.read()
            except Exception:
                logger.warning("Failed to download attachment %s", attachment.filename, exc_info=True)
                continue
            if len(raw_bytes) > MAX_IMAGE_BYTES:
                continue
            images.append(ImageInput.from_bytes(raw_bytes, mime_type))
        return images

    async def _collect_message_images(self, message: discord.Message) -> list[ImageInput]:
        """Collect images from the message itself and, if replying, the referenced message."""
        images = await self._extract_images(message.attachments, limit=MAX_IMAGES_PER_MESSAGE)
        remaining = MAX_IMAGES_PER_MESSAGE - len(images)
        if remaining <= 0 or not message.reference or not message.reference.message_id:
            return images

        ref_message: discord.Message | None = None
        if isinstance(message.reference.resolved, discord.Message):
            ref_message = message.reference.resolved
        else:
            try:
                ref_message = await message.channel.fetch_message(message.reference.message_id)
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                ref_message = None

        if ref_message and ref_message.attachments:
            images.extend(await self._extract_images(ref_message.attachments, limit=remaining))

        return images

    async def _stream_reply(
        self,
        session_key: str,
        prompt: str,
        send_initial: Callable[[str], Awaitable[Any]],
        edit_message: Callable[[Any, str], Awaitable[Any]],
        send_extra: Callable[[str], Awaitable[Any]],
        images: Sequence[ImageInput] | None = None,
    ) -> None:
        """Stream Gemini text deltas and progressively update the Discord response."""
        buffer: list[str] = []
        sent_message: Any | None = None
        last_rendered = ""
        last_edit_at = 0.0

        async for delta in self.ai_service.stream_ask(session_key, prompt, images=images):
            buffer.append(delta)
            accumulated = "".join(buffer)
            if not accumulated.strip():
                continue

            now = time.monotonic()
            if sent_message is None:
                preview = build_stream_preview(accumulated, finished=False)
                sent_message = await send_initial(preview)
                last_rendered = preview
                last_edit_at = now
            elif now - last_edit_at >= STREAM_EDIT_INTERVAL_SECONDS:
                preview = build_stream_preview(accumulated, finished=False)
                if preview != last_rendered:
                    await edit_message(sent_message, preview)
                    last_rendered = preview
                    last_edit_at = now

        final_text = "".join(buffer).strip()
        chunks = split_message(final_text) or [
            "(Mình không nhận được nội dung phản hồi từ mô hình.)"
        ]

        if sent_message is None:
            sent_message = await send_initial(chunks[0])
        elif chunks[0] != last_rendered:
            await edit_message(sent_message, chunks[0])

        for extra_chunk in chunks[1:]:
            await send_extra(extra_chunk)

    async def on_message(self, message: discord.Message) -> None:
        """Handle DM live knowledge updates and multimodal Gemini AI chat."""
        if message.author.bot:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)

        if is_dm:
            parsed_cmd = parse_dm_knowledge_command(message.content)
            if parsed_cmd is not None:
                if not is_knowledge_manager(message.author, self.config):
                    await message.reply(self._permission_denied_message(), mention_author=False)
                    return

                action, arg = parsed_cmd
                reply_text = await self._execute_knowledge_command(
                    action, arg, author=str(message.author)
                )
                for index, chunk in enumerate(split_message(reply_text)):
                    if index == 0:
                        await message.reply(chunk, mention_author=False)
                    else:
                        await message.channel.send(chunk)
                return

        is_mentioned = (
            self.user is not None
            and not message.mention_everyone
            and self.user.mentioned_in(message)
        )

        if not is_dm and not is_mentioned:
            await self.process_commands(message)
            return

        prompt = message.content.strip()
        if self.user is not None:
            prompt = strip_bot_mention(prompt, self.user.id)

        images = await self._collect_message_images(message)

        if not prompt and not images:
            await message.reply(
                "Chào bạn! Hãy nhắn câu hỏi (kèm ảnh nếu muốn) với `@mention` hoặc dùng lệnh `/ask` để trò chuyện với mình nhé.",
                mention_author=False,
            )
            return

        session_key = make_session_key(message.channel.id, message.author.id)
        try:
            async with message.channel.typing():
                await self._stream_reply(
                    session_key=session_key,
                    prompt=prompt,
                    images=images,
                    send_initial=lambda text: message.reply(text, mention_author=False),
                    edit_message=lambda msg, text: msg.edit(content=text),
                    send_extra=lambda text: message.channel.send(text),
                )
        except GeminiNotConfiguredError as exc:
            await message.reply(str(exc), mention_author=False)
        except Exception:
            logger.exception("Failed to generate Gemini response for message %s", message.id)
            await message.reply(
                "Xin lỗi, mình gặp lỗi khi gọi Gemini AI. Vui lòng thử lại sau ít phút.",
                mention_author=False,
            )

    def _register_commands(self) -> None:
        """Attach slash commands to the bot's CommandTree."""

        @self.tree.command(name="ping", description="Kiểm tra trạng thái online và độ trễ của bot")
        async def ping_command(interaction: discord.Interaction) -> None:
            latency_ms = round(self.latency * 1000)
            await interaction.response.send_message(
                f"Pong! Bot đang online — độ trễ **{latency_ms} ms**."
            )

        @self.tree.command(name="info", description="Xem thông tin hoạt động và cấu hình của bot")
        async def info_command(interaction: discord.Interaction) -> None:
            latency_ms = round(self.latency * 1000)
            ai_status = "Đã bật" if self.ai_service.is_configured else "Chưa có GEMINI_API_KEY"
            embed = discord.Embed(
                title="Thông tin Bot Discord",
                color=discord.Color.blurple(),
                timestamp=datetime.now(timezone.utc),
            )
            embed.add_field(name="Trạng thái", value="Online", inline=True)
            embed.add_field(name="Độ trễ", value=f"{latency_ms} ms", inline=True)
            embed.add_field(name="Uptime", value=format_uptime(self.started_at), inline=True)
            embed.add_field(name="Số server", value=str(len(self.guilds)), inline=True)
            embed.add_field(name="Gemini AI", value=ai_status, inline=True)
            embed.add_field(name="Model", value=f"`{self.config.gemini_model}`", inline=True)
            embed.add_field(
                name="Thinking",
                value=f"`{self.config.thinking_level or 'default'}`",
                inline=True,
            )
            embed.add_field(
                name="Kiến thức đã học",
                value=f"{len(self.knowledge_store.list_entries())} mục",
                inline=True,
            )
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="ask", description="Hỏi Gemini AI (có thể đính kèm ảnh)")
        @app_commands.describe(
            prompt="Câu hỏi hoặc yêu cầu bạn muốn gửi cho AI",
            image="Ảnh đính kèm để bot xem và phân tích (PNG/JPG/WEBP/GIF, tối đa 10 MB)",
            private="Chỉ mình bạn nhìn thấy câu trả lời",
        )
        async def ask_command(
            interaction: discord.Interaction,
            prompt: str,
            image: discord.Attachment | None = None,
            private: bool = False,
        ) -> None:
            try:
                await interaction.response.defer(ephemeral=private, thinking=True)
            except discord.NotFound:
                logger.warning("Interaction %s expired before defer()", interaction.id)
                return

            images: list[ImageInput] = []
            if image is not None:
                mime_type = resolve_image_mime_type(image.content_type, image.filename)
                if not mime_type:
                    await interaction.followup.send(
                        "File đính kèm chưa phải ảnh hợp lệ. Hãy gửi file PNG, JPG, WEBP hoặc GIF nhé.",
                        ephemeral=private,
                    )
                    return
                if image.size > MAX_IMAGE_BYTES:
                    await interaction.followup.send(
                        "Ảnh vượt quá giới hạn 10 MB. Vui lòng chọn ảnh nhỏ hơn nhé.",
                        ephemeral=private,
                    )
                    return
                images = await self._extract_images([image], limit=1)
                if not images:
                    await interaction.followup.send(
                        "Mình không tải được ảnh đính kèm, bạn thử gửi lại nhé.",
                        ephemeral=private,
                    )
                    return

            session_key = make_session_key(interaction.channel_id or 0, interaction.user.id)

            try:
                await self._stream_reply(
                    session_key=session_key,
                    prompt=prompt,
                    images=images,
                    send_initial=lambda text: interaction.followup.send(
                        text, ephemeral=private, wait=True
                    ),
                    edit_message=lambda msg, text: msg.edit(content=text),
                    send_extra=lambda text: interaction.followup.send(text, ephemeral=private),
                )
            except GeminiNotConfiguredError as exc:
                await interaction.followup.send(str(exc), ephemeral=private)
            except Exception:
                logger.exception("Failed to handle /ask command")
                await interaction.followup.send(
                    "Xin lỗi, mình gặp lỗi khi gọi Gemini AI. Vui lòng thử lại sau.",
                    ephemeral=private,
                )

        @self.tree.command(
            name="teach",
            description="Dạy kiến thức mới cho bot (chỉ Role Người quản lí hoặc _ngkhoi.9999)",
        )
        @app_commands.describe(content="Thông tin, quy định hoặc kiến thức bạn muốn bot ghi nhớ")
        async def teach_command(interaction: discord.Interaction, content: str) -> None:
            if not is_knowledge_manager(interaction.user, self.config):
                await interaction.response.send_message(
                    self._permission_denied_message(), ephemeral=True
                )
                return

            await interaction.response.defer(ephemeral=True)
            result = await self._execute_knowledge_command(
                "add", content, author=str(interaction.user)
            )
            for chunk in split_message(result):
                await interaction.followup.send(chunk, ephemeral=True)

        @self.tree.command(
            name="knowledge",
            description="Xem toàn bộ kiến thức bot đã học (chỉ Role Người quản lí hoặc _ngkhoi.9999)",
        )
        async def knowledge_command(interaction: discord.Interaction) -> None:
            if not is_knowledge_manager(interaction.user, self.config):
                await interaction.response.send_message(
                    self._permission_denied_message(), ephemeral=True
                )
                return

            result = await self._execute_knowledge_command(
                "list", "", author=str(interaction.user)
            )
            chunks = split_message(result)
            await interaction.response.send_message(chunks[0], ephemeral=True)
            for extra in chunks[1:]:
                await interaction.followup.send(extra, ephemeral=True)

        @self.tree.command(
            name="forget",
            description="Xóa kiến thức đã dạy cho bot (chỉ Role Người quản lí hoặc _ngkhoi.9999)",
        )
        @app_commands.describe(target="Số ID kiến thức cần xóa (vd: 1) hoặc 'all' để xóa hết")
        async def forget_command(interaction: discord.Interaction, target: str) -> None:
            if not is_knowledge_manager(interaction.user, self.config):
                await interaction.response.send_message(
                    self._permission_denied_message(), ephemeral=True
                )
                return

            await interaction.response.defer(ephemeral=True)
            result = await self._execute_knowledge_command(
                "remove", target, author=str(interaction.user)
            )
            await interaction.followup.send(result, ephemeral=True)

        @self.tree.command(name="reset", description="Xóa ngữ cảnh hội thoại AI của bạn ở kênh này")
        async def reset_command(interaction: discord.Interaction) -> None:
            session_key = make_session_key(interaction.channel_id or 0, interaction.user.id)
            cleared = self.ai_service.reset_session(session_key)
            if cleared:
                await interaction.response.send_message(
                    "Đã xóa trí nhớ hội thoại AI của bạn trong kênh này.",
                    ephemeral=True,
                )
            else:
                await interaction.response.send_message(
                    "Hiện chưa có hội thoại nào được lưu trong kênh này.",
                    ephemeral=True,
                )

        @self.tree.command(name="help", description="Xem danh sách lệnh và cách trò chuyện với bot")
        async def help_command(interaction: discord.Interaction) -> None:
            embed = discord.Embed(
                title="Hướng dẫn sử dụng Bot",
                description="Bot hỗ trợ **Slash Commands**, **xem ảnh**, **cập nhật kiến thức** và **chat trực tiếp với Gemini AI**.",
                color=discord.Color.green(),
            )
            embed.add_field(
                name="Lệnh cơ bản",
                value=(
                    "`/ping` — Kiểm tra độ trễ\n"
                    "`/info` — Xem thông tin & uptime\n"
                    "`/help` — Xem hướng dẫn này"
                ),
                inline=False,
            )
            embed.add_field(
                name="Trò chuyện & Xem ảnh với AI",
                value=(
                    "`/ask <câu hỏi> [image]` — Hỏi Gemini AI (có thể đính kèm ảnh & bật `private`)\n"
                    "`/reset` — Xóa ngữ cảnh hội thoại hiện tại\n"
                    "`@mention` (kèm ảnh hoặc Reply vào tin nhắn có ảnh) / **DM** — Chat liên tục nhiều lượt"
                ),
                inline=False,
            )
            embed.add_field(
                name=f"Dạy kiến thức (Role `{self.config.manager_role_name}` hoặc `_ngkhoi.9999`)",
                value=(
                    "`/teach <nội dung>` — Dạy kiến thức mới cho bot\n"
                    "`/knowledge` — Xem các kiến thức bot đang nhớ\n"
                    "`/forget <id | all>` — Xóa kiến thức theo ID hoặc xóa hết\n"
                    "**Chat DM trực tuyến (`_ngkhoi.9999`)**: `dạy: <nội dung>`, `nhớ: <nội dung>`, `xem kiến thức`, `quên: <id | all>`"
                ),
                inline=False,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
