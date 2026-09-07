"""Delegate a prompt to Antigravity's `agy` CLI (Gemini 3.5 Flash, free Google
CloudCode quota) and return the answer text — to save Claude/Anthropic tokens.

Stdlib only (no fastmcp), so it can be imported by ats_check.py and the
resume-tailor pipeline without extra deps. Same transcript-scrape workaround as
the gemini-delegate MCP server: `agy -p` has a known bug (gemini-cli #27466,
antigravity-cli #76) where it writes nothing to stdout under non-TTY/Windows, so
we read the model reply straight from agy's own transcript JSONL.

Public API:
    ask_agy(prompt, timeout_sec=300, workdir=None, model=None) -> str | None
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

HOME = Path.home()
AGY_ROOT = HOME / ".gemini" / "antigravity-cli"
BRAIN = AGY_ROOT / "brain"
LAST_CONV = AGY_ROOT / "cache" / "last_conversations.json"

_CONV_RE = re.compile(r"conversation[=:]\s*([0-9a-fA-F-]{36})")


def agy_available() -> bool:
    return _agy_bin() is not None


def _agy_bin() -> str | None:
    found = shutil.which("agy")
    if found:
        return found
    fallback = HOME / "AppData" / "Local" / "agy" / "bin" / "agy.exe"
    return str(fallback) if fallback.exists() else None


def _conv_id_from_log(log_path: Path) -> str | None:
    try:
        text = log_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return None
    ids = _CONV_RE.findall(text)
    return ids[-1] if ids else None


def _conv_id_from_last(workdir: Path) -> str | None:
    try:
        data = json.loads(LAST_CONV.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    target = str(workdir).lower()
    for k, v in data.items():
        if k.lower() == target:
            return v
    return None


def _newest_brain_dir(since: float) -> str | None:
    if not BRAIN.is_dir():
        return None
    best, best_mtime = None, since
    for d in BRAIN.iterdir():
        if d.is_dir():
            m = d.stat().st_mtime
            if m >= best_mtime:
                best, best_mtime = d.name, m
    return best


def _transcript_paths(conv_id: str) -> list[Path]:
    """Prefer transcript_full.jsonl. transcript.jsonl is agy's *truncated* display
    view — it caps each content field at 2048 bytes and inserts a literal
    "<truncated N bytes>" marker, which corrupts any reply over ~2KB (breaks JSON).
    transcript_full.jsonl holds the complete, untruncated content."""
    logs = BRAIN / conv_id / ".system_generated" / "logs"
    return [logs / "transcript_full.jsonl", logs / "transcript.jsonl"]


def _parse_transcript(path: Path) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    steps = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            steps.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    last_user_idx = -1
    for i, s in enumerate(steps):
        if s.get("type") == "USER_INPUT":
            last_user_idx = i
    parts = []
    for s in steps[last_user_idx + 1:]:
        if (s.get("source") == "MODEL"
                and s.get("type") == "PLANNER_RESPONSE"
                and s.get("status") == "DONE"):
            content = s.get("content")
            if content:
                parts.append(content)
    return "\n".join(parts).strip() or None


def _extract_answer(conv_id: str, deadline: float) -> str | None:
    paths = _transcript_paths(conv_id)
    while time.time() < deadline:
        for path in paths:
            if path.exists():
                answer = _parse_transcript(path)
                if answer:
                    return answer
        time.sleep(0.4)
    for path in paths:
        if path.exists():
            answer = _parse_transcript(path)
            if answer:
                return answer
    return None


def ask_agy(prompt: str, timeout_sec: int = 300,
            workdir: str | None = None, model: str | None = None) -> str | None:
    """Run `agy -p <prompt>` and return Gemini's reply text, or None on failure.

    Self-contained: agy starts a fresh conversation seeing only this prompt.
    """
    agy = _agy_bin()
    if not agy:
        return None
    wd = Path(workdir).resolve() if workdir else Path.cwd()
    if not wd.is_dir():
        return None

    log_path = wd / f".agy_run_{int(time.time() * 1000)}_{os.getpid()}.log"
    args = [agy, "-p", prompt, "--dangerously-skip-permissions",
            "--print-timeout", f"{timeout_sec}s", "--log-file", str(log_path)]
    if model:
        args += ["--model", model]

    started = time.time()
    try:
        subprocess.run(
            args, cwd=str(wd),
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL, timeout=timeout_sec + 30,
        )
    except (subprocess.TimeoutExpired, OSError):
        try:
            log_path.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    conv_id = (_conv_id_from_log(log_path)
               or _conv_id_from_last(wd)
               or _newest_brain_dir(started - 1))
    try:
        log_path.unlink(missing_ok=True)
    except OSError:
        pass
    if not conv_id:
        return None
    return _extract_answer(conv_id, deadline=time.time() + 15)


if __name__ == "__main__":  # quick smoke test
    print(repr(ask_agy("Reply with exactly the single word PONG.", timeout_sec=120)))
