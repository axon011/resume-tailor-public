"""Claude Code wrapper that exposes an OpenAI-SDK-shaped interface.

Why this exists:
  - The pipeline was originally built against GLM-4.7 via z.ai's
    OpenAI-compatible endpoint. Calls go through `openai.OpenAI(...)`.
  - We want to migrate resume tailoring to Claude Haiku, but keep the
    rest of the pipeline (tailoring.py, coverletter/generator.py) intact
    so we're not rewriting call sites.
  - This client shells out to the `claude` CLI (which uses the user's
    existing Max subscription via OAuth), wraps the JSON output in an
    OpenAI-compatible response shape, and is drop-in for OpenAI client.

Selected via env var: set LLM_PROVIDER=claude-code in .env to activate.
Model selection: LLM_MODEL=haiku|sonnet|opus (default haiku).

Limitations:
  - One-shot prompt mode only (no streaming, no multi-turn within a call).
  - max_completion_tokens is accepted but informational; the CLI doesn't
    expose a hard output cap. Haiku 4.5 caps at 32K output anyway.
  - Cost is accumulated against the user's Max subscription. First call in
    a session pays the cache-creation cost (~$0.05); subsequent calls hit
    the cache and are much cheaper.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass


def _strip_code_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model wrapped its
    output in ```json ... ``` (common from agentic CLIs). Leaves inner content
    untouched so the existing JSON raw_decode parsing keeps working."""
    t = text.strip()
    m = re.match(r"^```[a-zA-Z0-9_-]*\s*\n(.*)\n```\s*$", t, re.DOTALL)
    if m:
        return m.group(1).strip()
    return t


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
    def __init__(self, client: "ClaudeCodeClient"):
        self._client = client

    def create(
        self,
        model: str | None = None,
        messages: list[dict] | None = None,
        max_completion_tokens: int = 8192,
        **_kwargs,
    ) -> _Response:
        """OpenAI-compatible chat completion via Claude Code CLI subprocess."""
        if not messages:
            raise ValueError("messages is required")

        # Split system from user messages. System content goes through the
        # CLI's real --append-system-prompt flag; only user turns become the
        # `-p` prompt.
        #
        # It used to be inlined into the prompt inside a "## SYSTEM
        # INSTRUCTIONS" block. Current Claude models correctly refuse that:
        # a block in the USER turn claiming system authority is the canonical
        # prompt-injection shape, so the model answered with a refusal instead
        # of JSON and every tailoring run died at "Could not parse LLM JSON"
        # (2026-08-15). The flag is the supported channel — no delimiter
        # spoofing, and the text lands where it actually belongs.
        system_parts: list[str] = []
        prompt_parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                system_parts.append(content)
            else:
                prompt_parts.append(content)
        prompt = "\n\n".join(prompt_parts)
        system_text = "\n\n".join(p for p in system_parts if p.strip())

        # Resolve model. Callers in this codebase pass `config.LLM_MODEL`
        # which may be set to a non-Claude string (e.g. "glm-4.7") if the
        # user only flipped LLM_PROVIDER without updating LLM_MODEL.
        # Whitelist known Claude aliases; otherwise silently use the client's
        # default so the call doesn't break.
        _CLAUDE_ALIASES = {
            "haiku", "sonnet", "opus",
            "claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7",
            "claude-haiku-4-5-20251001",
        }
        if model and model.lower() in _CLAUDE_ALIASES:
            chosen_model = model
        else:
            chosen_model = self._client.model

        cmd = [
            self._client.claude_path,
            "-p",
            "--model", chosen_model,
            "--output-format", "json",
            # Load ZERO user MCP servers: a bare `claude -p` otherwise inherits
            # the full interactive env (claude-mem ~34k-token injection + stuck
            # Google-auth / pending chrome-colab-ppt servers), adding ~8s+ init
            # that intermittently stalls past the timeout. See diag 2026-06-11.
            "--strict-mcp-config",
            "--mcp-config", '{"mcpServers":{}}',
        ]
        if system_text:
            cmd += ["--append-system-prompt", system_text]

        # Primary: Claude Code CLI. On timeout/error, fall back to Codex CLI
        # (the user's Codex subscription via `codex exec`) so a single hung
        # `claude -p` doesn't abort the whole pipeline. See 2026-06-18.
        try:
            text = self._run_claude(cmd, prompt)
        except RuntimeError as claude_err:
            if not self._client.codex_fallback:
                raise
            print(
                f"   [llm_client] claude path failed ({claude_err}); "
                f"falling back to codex exec...",
                file=sys.stderr,
            )
            try:
                text = self._client._run_codex(prompt)
            except RuntimeError as codex_err:
                raise RuntimeError(
                    f"both LLM paths failed -- claude: {claude_err} | "
                    f"codex: {codex_err}"
                )

        # Wrap in OpenAI response shape so callers using
        # `response.choices[0].message.content` keep working.
        return _Response(choices=[_Choice(message=_Message(content=text))])

    def _run_claude(self, cmd: list[str], prompt: str) -> str:
        """Run the claude CLI and return the assistant text. Raises RuntimeError."""
        # Run from a neutral scratch dir, NOT the repo. A `claude -p` started
        # inside this project inherits the job-search CLAUDE.md and starts
        # behaving like the job-search agent: it reads the hard-filter list,
        # decides the posting is out of scope, and returns a refusal essay
        # instead of the JSON it was asked for (2026-08-15: every --force
        # intern run died at "Could not parse LLM JSON"). This call is a pure
        # text transform — it has no business reading repo instructions.
        neutral_cwd = tempfile.mkdtemp(prefix="rt_llm_")
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=self._client.timeout_sec,
                encoding="utf-8",
                cwd=neutral_cwd,
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"claude CLI timed out after {self._client.timeout_sec}s"
            )

        if result.returncode != 0:
            raise RuntimeError(
                f"claude CLI exited {result.returncode}: "
                f"{(result.stderr or '')[:500]}"
            )

        # Parse the JSON envelope. Claude Code returns:
        #   {"type":"result","subtype":"success","is_error":false,
        #    "result":"<the assistant text>", ...usage..., "total_cost_usd":...}
        try:
            envelope = json.loads(result.stdout)
        except json.JSONDecodeError as e:
            raise RuntimeError(
                f"claude CLI returned non-JSON output: "
                f"{result.stdout[:300]}\nparse error: {e}"
            )

        if envelope.get("is_error"):
            raise RuntimeError(
                f"claude CLI reported error: "
                f"{envelope.get('result') or envelope}"
            )

        text = envelope.get("result")
        if text is None:
            raise RuntimeError(
                f"claude CLI envelope missing 'result' field: "
                f"{json.dumps(envelope)[:300]}"
            )
        # An empty/whitespace result is a degenerate success (seen when the
        # nested claude env returns nothing). Treat it as a failure so the
        # codex fallback takes over instead of handing empty text downstream.
        if not text.strip():
            raise RuntimeError("claude CLI returned an empty result")
        # The nested `claude -p` inherits the user's plugin hooks. A broken
        # plugin (e.g. claude-mem worker down) returns a hook-block message as
        # the "result" instead of the model output. Detect that and fail over
        # to codex rather than feeding the block notice downstream as JSON.
        head = text.lstrip()[:200].lower()
        if "operation blocked by hook" in head or "blocked by hook" in head:
            raise RuntimeError(
                "claude CLI prompt blocked by a plugin hook "
                "(likely claude-mem worker down)"
            )
        return text


class _Chat:
    def __init__(self, client: "ClaudeCodeClient"):
        self.completions = _CompletionsAPI(client)


class ClaudeCodeClient:
    """OpenAI-SDK-shaped client backed by the Claude Code CLI.

    Usage:
        client = ClaudeCodeClient(model="haiku")
        resp = client.chat.completions.create(
            model="haiku",
            messages=[
                {"role": "system", "content": "You are..."},
                {"role": "user", "content": "..."},
            ],
            max_completion_tokens=8192,
        )
        text = resp.choices[0].message.content
    """

    def __init__(
        self,
        model: str = "haiku",
        claude_path: str = "claude",
        timeout_sec: int = 300,
        codex_path: str = "codex",
        codex_timeout_sec: int = 420,
        codex_fallback: bool = True,
    ):
        self.model = model
        self.claude_path = claude_path
        # Allow overriding the per-call claude timeout via env for large prompts
        # (e.g. resume tailoring produces ~8K JSON tokens; haiku with cold cache
        # creation on a big input can exceed the 300s default in some envs).
        _env_timeout = os.environ.get("LLM_CLAUDE_TIMEOUT", "").strip()
        if _env_timeout.isdigit():
            timeout_sec = int(_env_timeout)
        self.timeout_sec = timeout_sec
        self.codex_path = codex_path
        self.codex_timeout_sec = codex_timeout_sec
        # Allow disabling via env (LLM_CODEX_FALLBACK=0) for envs without codex.
        env_flag = os.environ.get("LLM_CODEX_FALLBACK", "").strip().lower()
        if env_flag in {"0", "false", "no", "off"}:
            codex_fallback = False
        self.codex_fallback = codex_fallback
        self.chat = _Chat(self)

    def _run_codex(self, prompt: str) -> str:
        """Fallback completion via the Codex CLI (`codex exec`).

        Uses the user's Codex subscription (no separate API bill). We run in a
        read-only sandbox so the agent can't touch the filesystem, force it to
        behave as a plain text generator, and capture ONLY the final message via
        `-o <file>` (avoids parsing agent progress chatter off stdout).
        """
        guard = (
            "You are a text-generation API, not an interactive agent. "
            "Do NOT run shell commands, do NOT read or write files, do NOT "
            "explain your process. Respond with ONLY the requested output as "
            "your final message.\n\n"
        )
        full_prompt = guard + prompt

        # Resolve the codex executable. On Windows, npm installs `codex` as a
        # .cmd shim that bare-name subprocess can't find; shutil.which respects
        # PATHEXT and returns the real path.
        codex_exe = shutil.which(self.codex_path) or self.codex_path

        out_fd, out_path = tempfile.mkstemp(suffix=".txt", prefix="codex_out_")
        os.close(out_fd)
        try:
            cmd = [
                codex_exe,
                "exec",
                "--sandbox", "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--color", "never",
                "-o", out_path,
                "-",  # read prompt from stdin
            ]
            try:
                result = subprocess.run(
                    cmd,
                    input=full_prompt,
                    capture_output=True,
                    text=True,
                    timeout=self.codex_timeout_sec,
                    encoding="utf-8",
                )
            except FileNotFoundError:
                raise RuntimeError("codex CLI not found on PATH")
            except subprocess.TimeoutExpired:
                raise RuntimeError(
                    f"codex CLI timed out after {self.codex_timeout_sec}s"
                )

            try:
                with open(out_path, "r", encoding="utf-8") as fh:
                    text = fh.read().strip()
            except OSError:
                text = ""

            if not text:
                if result.returncode != 0:
                    raise RuntimeError(
                        f"codex CLI exited {result.returncode}: "
                        f"{(result.stderr or '')[:500]}"
                    )
                raise RuntimeError(
                    "codex CLI produced no final message "
                    f"(stderr: {(result.stderr or '')[:300]})"
                )

            return _strip_code_fences(text)
        finally:
            try:
                os.remove(out_path)
            except OSError:
                pass
