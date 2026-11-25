"""Discord bot implementation with slash commands and streaming Gemini AI chat."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any

import discord
from discord import app_commands
from discord.ext import commands

from src.ai_service import GeminiChatService, GeminiNotConfiguredError
from src.config import BotConfig
from src.utils import split_message


logger = logging.getLogger(__name__)

STREAM_EDIT_INTERVAL_SECONDS = 0.8
STREAM_TYPING_INDICATOR = " ..."


def make_session_key(channel_id: int, user_id: int) -> str:
    """Build a per-channel, per-user session key for stateful AI chat."""
    return f"{channel_id}:{user_id}"


def strip_bot_mention(content: str, bot_user_id: int) -> str:
    """Remove direct bot mention tags from a Discord message."""
    pattern = rf"<@!?{bot_user_id}>"
    return re.sub(pattern, "", content).strip()


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
    """Discord bot combining utility slash commands and streaming Gemini AI chat."""

    def __init__(
        self,
        config: BotConfig,
        ai_service: GeminiChatService | None = None,
    ) -> None:
        intents = discord.Intents.default()
        intents.message_content = True

        super().__init__(command_prefix="!", intents=intents, help_command=None)
        self.config = config
        self.started_at = datetime.now(timezone.utc)
        self.ai_service = ai_service or GeminiChatService(
            api_key=config.gemini_api_key,
            model=config.gemini_model,
            system_instruction=config.system_instruction,
            thinking_level=config.thinking_level,
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
        }

    async def setup_hook(self) -> None:
        """Register and sync application slash commands before connecting."""
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

    async def _stream_reply(
        self,
        session_key: str,
        prompt: str,
        send_initial: Callable[[str], Awaitable[Any]],
        edit_message: Callable[[Any, str], Awaitable[Any]],
        send_extra: Callable[[str], Awaitable[Any]],
    ) -> None:
        """Stream Gemini text deltas and progressively update the Discord response."""
        buffer: list[str] = []
        sent_message: Any | None = None
        last_rendered = ""
        last_edit_at = 0.0

        async for delta in self.ai_service.stream_ask(session_key, prompt):
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
        """Reply with streaming Gemini AI when mentioned in a server or messaged in DM."""
        if message.author.bot:
            return

        is_dm = isinstance(message.channel, discord.DMChannel)
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

        if not prompt:
            await message.reply(
                "Chào bạn! Hãy nhắn câu hỏi kèm `@mention` hoặc dùng lệnh `/ask` để trò chuyện với mình nhé.",
                mention_author=False,
            )
            return

        session_key = make_session_key(message.channel.id, message.author.id)
        try:
            async with message.channel.typing():
                await self._stream_reply(
                    session_key=session_key,
                    prompt=prompt,
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
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="ask", description="Hỏi Gemini AI bất kỳ điều gì")
        @app_commands.describe(
            prompt="Câu hỏi hoặc yêu cầu bạn muốn gửi cho AI",
            private="Chỉ mình bạn nhìn thấy câu trả lời",
        )
        async def ask_command(
            interaction: discord.Interaction,
            prompt: str,
            private: bool = False,
        ) -> None:
            try:
                await interaction.response.defer(ephemeral=private, thinking=True)
            except discord.NotFound:
                logger.warning("Interaction %s expired before defer()", interaction.id)
                return

            session_key = make_session_key(interaction.channel_id or 0, interaction.user.id)

            try:
                await self._stream_reply(
                    session_key=session_key,
                    prompt=prompt,
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
                description="Bot hỗ trợ cả **Slash Commands** và **chat trực tiếp với Gemini AI** (phản hồi streaming).",
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
                name="Trò chuyện với AI",
                value=(
                    "`/ask <câu hỏi>` — Hỏi Gemini AI (có tùy chọn `private`)\n"
                    "`/reset` — Xóa ngữ cảnh hội thoại hiện tại\n"
                    "`@mention` hoặc **DM** — Tag tên bot hoặc nhắn riêng để chat liên tục nhiều lượt"
                ),
                inline=False,
            )
            await interaction.response.send_message(embed=embed, ephemeral=True)
