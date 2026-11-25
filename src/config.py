"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_SYSTEM_INSTRUCTION = (
    "Bạn là một trợ lý AI thân thiện, hữu ích trong server Discord. "
    "Hãy trả lời rõ ràng, tự nhiên và súc tích bằng ngôn ngữ của người dùng."
)


@dataclass(frozen=True)
class BotConfig:
    """Configuration values required to run the Discord bot and health server."""

    discord_token: str
    gemini_api_key: str | None
    gemini_model: str
    system_instruction: str
    guild_id: int | None
    port: int

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from `.env` and process environment variables."""
        load_dotenv()

        discord_token = os.getenv("DISCORD_TOKEN", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        gemini_model = os.getenv("GEMINI_MODEL", "gemini-4-hydrogen").strip() or "gemini-4-hydrogen"
        system_instruction = (
            os.getenv("BOT_SYSTEM_INSTRUCTION", DEFAULT_SYSTEM_INSTRUCTION).strip()
            or DEFAULT_SYSTEM_INSTRUCTION
        )

        raw_guild_id = os.getenv("DISCORD_GUILD_ID", "").strip()
        guild_id: int | None = None
        if raw_guild_id:
            try:
                guild_id = int(raw_guild_id)
            except ValueError as exc:
                raise ValueError("DISCORD_GUILD_ID must be a numeric Discord server ID.") from exc

        raw_port = os.getenv("PORT", "8080").strip()
        try:
            port = int(raw_port)
        except ValueError as exc:
            raise ValueError("PORT must be a valid integer.") from exc

        return cls(
            discord_token=discord_token,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            system_instruction=system_instruction,
            guild_id=guild_id,
            port=port,
        )
