"""Persistent knowledge store with optional GitHub Gist cloud backup."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import aiohttp


logger = logging.getLogger(__name__)

GIST_FILENAME = "discord_bot_knowledge.json"
GIST_API_URL = "https://api.github.com/gists/{gist_id}"


@dataclass(frozen=True)
class KnowledgeEntry:
    """Single piece of knowledge taught to the bot by a manager."""

    id: int
    content: str
    author: str
    created_at: str


class KnowledgeStore:
    """Manages bot knowledge persisted to JSON and optionally backed up to a GitHub Gist."""

    def __init__(
        self,
        file_path: Path | str = Path("data/knowledge.json"),
        gist_id: str | None = None,
        github_token: str | None = None,
    ) -> None:
        self.file_path = Path(file_path)
        self.gist_id = gist_id.strip() if gist_id else None
        self.github_token = github_token.strip() if github_token else None
        self._entries: list[KnowledgeEntry] = []
        self._next_id = 1
        self.load()

    @property
    def cloud_backup_enabled(self) -> bool:
        """Return True when both GitHub Gist ID and token are configured."""
        return bool(self.gist_id and self.github_token)

    def _deserialize(self, raw_data: dict[str, Any]) -> None:
        entries: list[KnowledgeEntry] = []
        max_id = 0
        for item in raw_data.get("entries", []):
            entry_id = int(item["id"])
            content = str(item["content"]).strip()
            if not content:
                continue
            entries.append(
                KnowledgeEntry(
                    id=entry_id,
                    content=content,
                    author=str(item.get("author", "unknown")),
                    created_at=str(item.get("created_at", "")),
                )
            )
            if entry_id > max_id:
                max_id = entry_id

        self._entries = entries
        self._next_id = max(int(raw_data.get("next_id", max_id + 1)), max_id + 1)

    def _serialize(self) -> dict[str, Any]:
        return {
            "next_id": self._next_id,
            "entries": [asdict(entry) for entry in self._entries],
        }

    def load(self) -> None:
        """Load knowledge entries from the local JSON file if it exists."""
        if not self.file_path.exists():
            self._entries = []
            self._next_id = 1
            return

        try:
            raw = json.loads(self.file_path.read_text(encoding="utf-8"))
            self._deserialize(raw)
        except Exception:
            logger.exception("Failed to read knowledge file %s; starting empty", self.file_path)
            self._entries = []
            self._next_id = 1

    def _save_local(self) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._serialize(), ensure_ascii=False, indent=2)
        self.file_path.write_text(payload, encoding="utf-8")

    async def async_load(self) -> None:
        """Load local entries and pull newer backup from GitHub Gist if configured."""
        self.load()
        if not self.cloud_backup_enabled:
            return

        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "discord-gemini-bot",
        }
        url = GIST_API_URL.format(gist_id=self.gist_id)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status != 200:
                        logger.warning("GitHub Gist load returned HTTP %s", resp.status)
                        return
                    data = await resp.json()
        except Exception:
            logger.warning("Failed to pull knowledge backup from GitHub Gist", exc_info=True)
            return

        files = data.get("files") or {}
        file_obj = files.get(GIST_FILENAME)
        if not file_obj or not file_obj.get("content"):
            return

        try:
            remote_data = json.loads(file_obj["content"])
            self._deserialize(remote_data)
            self._save_local()
            logger.info("Loaded %d knowledge entries from GitHub Gist", len(self._entries))
        except Exception:
            logger.warning("Invalid JSON in GitHub Gist knowledge backup", exc_info=True)

    async def _save_cloud(self) -> None:
        if not self.cloud_backup_enabled:
            return

        headers = {
            "Authorization": f"Bearer {self.github_token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "discord-gemini-bot",
        }
        url = GIST_API_URL.format(gist_id=self.gist_id)
        payload = {
            "files": {
                GIST_FILENAME: {
                    "content": json.dumps(self._serialize(), ensure_ascii=False, indent=2)
                }
            }
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.patch(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status >= 400:
                        logger.warning("GitHub Gist save returned HTTP %s", resp.status)
        except Exception:
            logger.warning("Failed to push knowledge backup to GitHub Gist", exc_info=True)

    async def add(self, content: str, author: str) -> KnowledgeEntry:
        """Add a new knowledge entry and persist it."""
        cleaned = content.strip()
        if not cleaned:
            raise ValueError("Nội dung kiến thức không được để trống.")

        entry = KnowledgeEntry(
            id=self._next_id,
            content=cleaned,
            author=author,
            created_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        )
        self._next_id += 1
        self._entries.append(entry)
        self._save_local()
        await self._save_cloud()
        return entry

    async def remove(self, entry_id: int) -> KnowledgeEntry | None:
        """Remove a knowledge entry by ID and persist the change."""
        for index, entry in enumerate(self._entries):
            if entry.id == entry_id:
                removed = self._entries.pop(index)
                self._save_local()
                await self._save_cloud()
                return removed
        return None

    async def clear(self) -> int:
        """Remove all stored knowledge entries and return how many were cleared."""
        count = len(self._entries)
        if count == 0:
            return 0
        self._entries.clear()
        self._save_local()
        await self._save_cloud()
        return count

    def list_entries(self) -> list[KnowledgeEntry]:
        """Return a shallow copy of all stored knowledge entries."""
        return list(self._entries)

    def build_prompt_section(self) -> str:
        """Format stored knowledge for injection into Gemini system_instruction."""
        if not self._entries:
            return ""

        lines = [
            "[KIẾN THỨC BỔ SUNG TỪ NGƯỜI QUẢN LÝ]",
            "Hãy luôn ghi nhớ và áp dụng chính xác các thông tin, quy định và kiến thức sau đây khi trả lời người dùng:",
        ]
        for entry in self._entries:
            lines.append(f"- (#{entry.id}) {entry.content}")
        return "\n".join(lines)
