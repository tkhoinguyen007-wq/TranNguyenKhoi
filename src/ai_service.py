"""Gemini AI chat service using the current google-genai Interactions API."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

from google import genai


logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 25.0


class GeminiNotConfiguredError(RuntimeError):
    """Raised when Gemini AI is invoked without a configured GEMINI_API_KEY."""


class GeminiChatService:
    """Manages stateful multi-turn conversations with Gemini via Interactions API."""

    def __init__(
        self,
        api_key: str | None,
        model: str = "gemini-3.5-flash-lite",
        system_instruction: str = "",
        thinking_level: str | None = "minimal",
        fallback_model: str | None = "gemini-3.5-flash-lite",
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT_SECONDS,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self.system_instruction = system_instruction
        self.thinking_level = thinking_level
        self.fallback_model = fallback_model
        self.request_timeout = request_timeout
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
        model: str,
        thinking_level: str | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build keyword arguments for `aio.interactions.create`."""
        kwargs: dict[str, Any] = {
            "model": model,
            "input": cleaned_prompt,
            "timeout": self.request_timeout,
        }
        if self.system_instruction:
            kwargs["system_instruction"] = self.system_instruction
        if thinking_level:
            kwargs["generation_config"] = {"thinking_level": thinking_level}

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

    def _model_candidates(self) -> list[tuple[str, str | None]]:
        """Return primary and optional fallback `(model, thinking_level)` pairs."""
        candidates: list[tuple[str, str | None]] = [(self.model, self.thinking_level)]
        if self.fallback_model and self.fallback_model != self.model:
            fallback_thinking = "minimal" if "flash-lite" in self.fallback_model else "low"
            candidates.append((self.fallback_model, fallback_thinking))
        return candidates

    async def ask(self, session_key: str, prompt: str) -> str:
        """Send a prompt to Gemini and remember the interaction ID for follow-up turns."""
        client = self._ensure_configured()

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            return "Bạn hãy nhập nội dung câu hỏi để mình trả lời nhé."

        candidates = self._model_candidates()
        for idx, (model_name, thinking) in enumerate(candidates):
            kwargs = self._build_request_kwargs(
                session_key,
                cleaned_prompt,
                model=model_name,
                thinking_level=thinking,
                stream=False,
            )
            try:
                interaction = await client.aio.interactions.create(**kwargs)
                interaction_id = getattr(interaction, "id", None)
                if interaction_id:
                    self._sessions[session_key] = interaction_id
                output_text = (getattr(interaction, "output_text", None) or "").strip()
                return output_text or "(Mình không nhận được nội dung phản hồi từ mô hình.)"
            except Exception as exc:
                if idx + 1 >= len(candidates):
                    raise
                logger.warning(
                    "Primary model %s failed (%s); falling back to %s",
                    model_name,
                    exc,
                    candidates[idx + 1][0],
                )

        raise RuntimeError("No Gemini model candidates available.")

    async def stream_ask(self, session_key: str, prompt: str) -> AsyncIterator[str]:
        """Stream text deltas from Gemini with fallback if primary model stalls/fails."""
        client = self._ensure_configured()

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt:
            yield "Bạn hãy nhập nội dung câu hỏi để mình trả lời nhé."
            return

        candidates = self._model_candidates()
        for idx, (model_name, thinking) in enumerate(candidates):
            emitted_any = False
            kwargs = self._build_request_kwargs(
                session_key,
                cleaned_prompt,
                model=model_name,
                thinking_level=thinking,
                stream=True,
            )
            try:
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
                                emitted_any = True
                                yield text_chunk
                    elif event_type == "error":
                        error_obj = getattr(event, "error", None)
                        raise RuntimeError(f"Gemini streaming error: {error_obj}")
                return
            except Exception as exc:
                if emitted_any or idx + 1 >= len(candidates):
                    raise
                logger.warning(
                    "Primary streaming model %s failed (%s); falling back to %s",
                    model_name,
                    exc,
                    candidates[idx + 1][0],
                )

    def reset_session(self, session_key: str) -> bool:
        """Clear saved conversation state for the given session key."""
        return self._sessions.pop(session_key, None) is not None

    def active_session_count(self) -> int:
        """Return the number of active conversation sessions in memory."""
        return len(self._sessions)
