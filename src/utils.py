"""Utility helpers for formatting and splitting Discord messages."""

from __future__ import annotations


DISCORD_MAX_MESSAGE_LENGTH = 2000


def split_message(text: str, max_length: int = 1900) -> list[str]:
    """Split a long response into chunks within Discord's 2000-character limit.

    Prefers splitting on paragraph or line boundaries, then whitespace, and falls
    back to a hard split only when a single token exceeds ``max_length``.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    if max_length <= 0 or max_length > DISCORD_MAX_MESSAGE_LENGTH:
        raise ValueError(f"max_length must be between 1 and {DISCORD_MAX_MESSAGE_LENGTH}.")

    if len(cleaned) <= max_length:
        return [cleaned]

    chunks: list[str] = []
    remaining = cleaned

    while len(remaining) > max_length:
        window = remaining[:max_length]
        split_at = window.rfind("\n\n")
        if split_at < max_length // 2:
            split_at = window.rfind("\n")
        if split_at < max_length // 2:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = max_length

        chunk = remaining[:split_at].rstrip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].lstrip()

    if remaining:
        chunks.append(remaining)

    return chunks
