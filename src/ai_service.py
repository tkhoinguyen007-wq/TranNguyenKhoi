"""Gemini AI chat service using the current google-genai Interactions API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from google import genai


class GeminiNotConfiguredError(RuntimeError):
    """Raised when Gemini AI is invoked without a configured GEMINI_API_KEY."""


class GeminiChatService:
    """Manages stateful multi-turn conversations with Gemini via Interactions API."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-4-hydrogen",
        system_instruction: str = "",
        thinking_level: str | None = "low",
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.system_instruction = system_instruction
        self.thinking_level = thinking_level
        self._sessions: dict[str, str] = {}

        if client is not None:
            self._client = client
        elif api_key:
            self._client = genai.Client(api_key=api_key)
        else:
            self._client = None

    @property
    def is_configured(self) -> bool:
        """Return True if a Gemini client is available."""
        return self._client is not None

    def _build_request_kwargs(
        self,
        session_key: str,
        cleaned_prompt: str,
        *,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build keyword arguments for `aio.interactions.create`."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": cleaned_prompt,
        }
        if self.system_instruction:
            kwargs["system_instruction"] = self.system_instruction
        if self.thinking_level:
            kwargs["generation_config"] = {"thinking_level": self.thinking_level}

        previous_id = self._sessions.get(session_key)
        if previous_id:
            kwargs["previous_interaction_id"] = previous_id

        if stream:
            kwargs["stream"] = True

        return kwargs

    def _ensure_configured(self) -> Any:
        if self._client is None:
            raise GeminiNotConfiguredError(
                "GEMINI_API_KEY chưa được cấu hình. Hãy thêm key vào biến môi trường hoặc file .env."
            )
        return self._client

    async def ask(self, session_key: str, prompt: str) -> str:
        """Send a prompt to Gemini and remember the interaction ID for follow-up turns."""
        client = self._ensure_configured()

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return "Bạn hãy nhập nội dung câu hỏi để mình trả lời nhé."

        kwargs = self._build_request_kwargs(session_key, cleaned_prompt, stream=False)
        interaction = await client.aio.interactions.create(**kwargs)

        interaction_id = getattr(interaction, "id", None)
        if interaction_id:
            self._sessions[session_key] = interaction_id

        output_text = (getattr(interaction, "output_text", None) or "").strip()
        return output_text or "(Mình không nhận được nội dung phản hồi từ mô hình.)"

    async def stream_ask(self, session_key: str, prompt: str) -> AsyncIterator[str]:
        """Stream text deltas from Gemini while keeping conversation state."""
        client = self._ensure_configured()

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            yield "Bạn hãy nhập nội dung câu hỏi để mình trả lời nhé."
            return

        kwargs = self._build_request_kwargs(session_key, cleaned_prompt, stream=True)
        stream = await client.aio.interactions.create(**kwargs)

        async for event in stream:
            event_type = getattr(event, "event_type", None)

            if event_type in ("interaction.created", "interaction.completed"):
                interaction = getattr(event, "interaction", None)
                interaction_id = getattr(interaction, "id", None)
                if interaction_id:
                    self._sessions[session_key] = interaction_id
            elif event_type == "step.delta":
                delta = getattr(event, "delta", None)
                if getattr(delta, "type", None) == "text":
                    text_chunk = getattr(delta, "text", "")
                    if text_chunk:
                        yield text_chunk
            elif event_type == "error":
                error_obj = getattr(event, "error", None)
                raise RuntimeError(f"Gemini streaming error: {error_obj}")

    def reset_session(self, session_key: str) -> bool:
        """Clear saved conversation state for the given session key."""
        return self._sessions.pop(session_key, None) is not None

    def active_session_count(self) -> int:
        """Return the number of active conversation sessions in memory."""
        return len(self._sessions)
