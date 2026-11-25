"""Gemini AI chat service using the current google-genai Interactions API."""

from __future__ import annotations

import base64
import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Any

from google import genai


logger = logging.getLogger(__name__)

DEFAULT_REQUEST_TIMEOUT_SECONDS = 25.0
DEFAULT_IMAGE_PROMPT = "Hãy xem và mô tả chi tiết hình ảnh này."


@dataclass(frozen=True)
class ImageInput:
    """Base64-encoded image payload for Gemini multimodal interactions."""

    data_base64: str
    mime_type: str

    @classmethod
    def from_bytes(cls, raw_bytes: bytes, mime_type: str) -> "ImageInput":
        """Encode raw image bytes to ASCII base64."""
        encoded = base64.b64encode(raw_bytes).decode("ascii")
        return cls(data_base64=encoded, mime_type=mime_type)


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

    def _build_input_payload(
        self,
        cleaned_prompt: str,
        images: Sequence[ImageInput] | None = None,
    ) -> str | list[dict[str, str]]:
        """Build either plain text or multimodal list input for Gemini."""
        if not images:
            return cleaned_prompt

        prompt_text = cleaned_prompt or DEFAULT_IMAGE_PROMPT
        blocks: list[dict[str, str]] = [{"type": "text", "text": prompt_text}]
        for image in images:
            blocks.append(
                {
                    "type": "image",
                    "data": image.data_base64,
                    "mime_type": image.mime_type,
                }
            )
        return blocks

    def _build_request_kwargs(
        self,
        session_key: str,
        cleaned_prompt: str,
        *,
        model: str,
        thinking_level: str | None,
        images: Sequence[ImageInput] | None = None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """Build keyword arguments for `aio.interactions.create`."""
        kwargs: dict[str, Any] = {
            "model": model,
            "input": self._build_input_payload(cleaned_prompt, images=images),
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

    async def ask(
        self,
        session_key: str,
        prompt: str,
        images: Sequence[ImageInput] | None = None,
    ) -> str:
        """Send a prompt (optionally with images) to Gemini and save conversation state."""
        client = self._ensure_configured()

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt and not images:
            return "Bạn hãy nhập nội dung câu hỏi để mình trả lời nhé."

        candidates = self._model_candidates()
        for idx, (model_name, thinking) in enumerate(candidates):
            kwargs = self._build_request_kwargs(
                session_key,
                cleaned_prompt,
                model=model_name,
                thinking_level=thinking,
                images=images,
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

    async def stream_ask(
        self,
        session_key: str,
        prompt: str,
        images: Sequence[ImageInput] | None = None,
    ) -> AsyncIterator[str]:
        """Stream text deltas from Gemini (optionally with images) with fallback."""
        client = self._ensure_configured()

        cleaned_prompt = prompt.strip()
        if not cleaned_prompt and not images:
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
                images=images,
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
