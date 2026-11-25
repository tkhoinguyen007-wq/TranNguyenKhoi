"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

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
    fallback_model: str | None
    thinking_level: str | None
    system_instruction: str
    guild_id: int | None
    port: int
    manager_role_name: str = "Người quản lí"
    manager_usernames: tuple[str, ...] = ("_ngkhoi.9999",)
    knowledge_file: Path = Path("data/knowledge.json")
    knowledge_gist_id: str | None = None
    knowledge_github_token: str | None = None

    @classmethod
    def from_env(cls) -> "BotConfig":
        """Load configuration from `.env` and process environment variables."""
        load_dotenv()

        discord_token = os.getenv("DISCORD_TOKEN", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip() or None
        gemini_model = (
            os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip() or "gemini-3.5-flash-lite"
        )
        fallback_model = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite").strip() or None

        raw_thinking = os.getenv("GEMINI_THINKING_LEVEL", "minimal").strip().lower()
        thinking_level = None if raw_thinking in ("", "none", "default") else raw_thinking

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

        manager_role_name = os.getenv("MANAGER_ROLE_NAME", "Người quản lí").strip() or "Người quản lí"
        raw_managers = os.getenv("KNOWLEDGE_MANAGER_USERNAMES", "_ngkhoi.9999").strip()
        manager_usernames = tuple(
            name.strip().lower() for name in raw_managers.split(",") if name.strip()
        )

        knowledge_file = Path(
            os.getenv("KNOWLEDGE_FILE", "data/knowledge.json").strip() or "data/knowledge.json"
        )
        knowledge_gist_id = os.getenv("KNOWLEDGE_GIST_ID", "").strip() or None
        knowledge_github_token = os.getenv("KNOWLEDGE_GITHUB_TOKEN", "").strip() or None

        return cls(
            discord_token=discord_token,
            gemini_api_key=gemini_api_key,
            gemini_model=gemini_model,
            fallback_model=fallback_model,
            thinking_level=thinking_level,
            system_instruction=system_instruction,
            guild_id=guild_id,
            port=port,
            manager_role_name=manager_role_name,
            manager_usernames=manager_usernames,
            knowledge_file=knowledge_file,
            knowledge_gist_id=knowledge_gist_id,
            knowledge_github_token=knowledge_github_token,
        )
