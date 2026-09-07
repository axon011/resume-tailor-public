"""Antigravity `agy` (Gemini) wrapper exposing an OpenAI-SDK-shaped interface.

Drop-in sibling of ClaudeCodeClient: returns objects with
`.chat.completions.create(...)` so tailoring.py / coverletter/generator.py don't
need to know the backend. Selected via LLM_PROVIDER=gemini (or =agy).

Routes through agy_delegate.ask_agy, which runs `agy -p` on Google's free
CloudCode quota and scrapes the reply from agy's transcript (Windows stdout bug
workaround). Use to conserve Claude/Anthropic tokens.

Note: resume rewrite + cover letter are high-stakes; Gemini is opt-in here, with
Claude/claude-code remaining the recommended quality default.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

# agy_delegate lives at the resume-tailor root (one level up from pipeline/).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agy_delegate import ask_agy  # noqa: E402


@dataclass
class _Message:
    content: str


@dataclass
class _Choice:
    message: _Message


@dataclass
class _Response:
    choices: list


class _CompletionsAPI:
    def __init__(self, client: "AgyClient"):
        self._client = client

    def create(self, model: str | None = None, messages: list[dict] | None = None,
               max_completion_tokens: int = 8192, **_kwargs) -> _Response:
        if not messages:
            raise ValueError("messages is required")
        prompt_parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                prompt_parts.append(
                    "## SYSTEM INSTRUCTIONS\n"
                    f"{content}\n"
                    "## END SYSTEM INSTRUCTIONS\n"
                )
            else:
                prompt_parts.append(content)
        prompt = "\n\n".join(prompt_parts)

        answer = ask_agy(prompt, timeout_sec=self._client.timeout,
                         model=self._client.model)
        if not answer:
            raise RuntimeError(
                "agy/Gemini returned no answer (check `agy` auth via "
                "gemini_status, or set LLM_PROVIDER=claude-code to fall back)."
            )
        return _Response(choices=[_Choice(message=_Message(content=answer))])


class _ChatAPI:
    def __init__(self, client: "AgyClient"):
        self.completions = _CompletionsAPI(client)


class AgyClient:
    """OpenAI-shaped client backed by the `agy` CLI (Gemini)."""

    def __init__(self, model: str | None = None, timeout: int = 300):
        # agy uses human-readable model labels, not API ids. None = agy default.
        self.model = model or os.getenv("AGY_MODEL") or None
        self.timeout = timeout
        self.chat = _ChatAPI(self)
