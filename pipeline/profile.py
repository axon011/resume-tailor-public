"""Candidate profile — everything about ONE person lives here, nowhere else in the code.

The pipeline ships with no personal data. Every gate, prompt rule, header field and
availability story is read from a profile directory (default `candidate/`, gitignored):

    candidate/
      gates.json   employers, personal-project vocabulary, banned phrases, real metric
                   values, protected clauses, tracks, home cities, publications
      rules.md     free-text rules injected into the tailoring and cover-letter prompts
      resume.tex   the base CV (path may instead be given by RESUME_TEX_PATH)

Copy `candidate.example/` to `candidate/` and edit. `CANDIDATE_DIR` overrides the path.
The loader fails loudly with the path it looked in — a silently empty profile would make
every gate pass, which is the worst possible failure for a tool whose job is to block.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

_HERE = Path(__file__).resolve().parent.parent
PROFILE_DIR = Path(os.environ.get("CANDIDATE_DIR") or _HERE / "candidate")
EXAMPLE_DIR = _HERE / "candidate.example"


class ProfileError(RuntimeError):
    pass


def _compile(pattern: str, flags=re.I) -> re.Pattern:
    try:
        return re.compile(pattern, flags)
    except re.error as e:
        raise ProfileError(f"bad regex in gates.json: {pattern!r} ({e})") from e


class Profile:
    """Typed view over gates.json + rules.md. Attributes are plain data or compiled regexes."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        gates = self.dir / "gates.json"
        rules = self.dir / "rules.md"
        if not gates.is_file():
            raise ProfileError(
                f"No candidate profile at {self.dir} (missing gates.json). "
                f"Copy {EXAMPLE_DIR} to {self.dir} and edit it, or set CANDIDATE_DIR.")
        try:
            g = json.loads(gates.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ProfileError(f"{gates}: invalid JSON ({e})") from e
        self.raw = g
        self.rules_text = rules.read_text(encoding="utf-8") if rules.is_file() else ""

        # --- employers -----------------------------------------------------------
        self.employers = g.get("employers", [])
        if not self.employers:
            raise ProfileError("gates.json: 'employers' must list at least one employer")
        self.primary_employer = g.get("primary_employer") or self.employers[0]["name"]
        names = [e["name"] for e in self.employers]
        self.employer_re = _compile("|".join(re.escape(n) for n in names))
        self.primary_employer_re = _compile(re.escape(self.primary_employer))
        prod_words = []
        for e in self.employers:
            if e.get("production_ok"):
                prod_words += [re.escape(e["name"])] + list(e.get("aliases", []))
        # Never an empty alternation: an empty pattern matches everything and would
        # disable the production rule wholesale.
        self.production_ok_re = _compile("|".join(prod_words)) if prod_words else _compile(r"(?!x)x")
        self.production_ok_names = [e["name"] for e in self.employers if e.get("production_ok")]
        self.protected_clauses = {e["name"].lower(): e.get("protected_clauses", [])
                                  for e in self.employers if e.get("protected_clauses")}
        self.banned_additions = {e["name"].lower(): e.get("banned_additions", [])
                                 for e in self.employers if e.get("banned_additions")}
        self.frozen_titles = {e["name"]: e["frozen_title"] for e in self.employers if e.get("frozen_title")}
        self.frozen_bullets = [e["name"] for e in self.employers if e.get("frozen_bullets")]

        # --- personal projects ---------------------------------------------------
        pp = g.get("personal_projects", {})
        vocab = pp.get("vocabulary", [])
        self.personal_repo_re = _compile("|".join(vocab)) if vocab else _compile(r"(?!x)x")
        self.claimgate_vocabulary = [(_compile(v["pattern"]), v["label"])
                                     for v in pp.get("claimgate_vocabulary", [])]
        self.cover_split_re = _compile(pp["cover_split_vocabulary"]) if pp.get("cover_split_vocabulary") else self.personal_repo_re
        self.github_marker = pp.get("github_marker", "")
        lp = pp.get("live_project") or {}
        self.live_project_re = _compile(lp["pattern"]) if lp.get("pattern") else None
        self.live_project_name = lp.get("name", "")

        # --- claims and metrics --------------------------------------------------
        self.realtime_ok_re = _compile(g["realtime_ok"]) if g.get("realtime_ok") else None
        self.real_metric_values = set(g.get("real_metric_values", []))
        self.banned_phrases = [(_compile(b["pattern"]), b["reason"]) for b in g.get("banned_phrases", [])]
        self.auto_strip_sentences = [_compile(p) for p in g.get("auto_strip_sentences", [])]
        self.never_claim_tools = g.get("never_claim_tools", [])

        # --- tracks / availability ----------------------------------------------
        self.tracks = g.get("tracks") or {"default": {"availability": ""}}
        self.default_track = g.get("default_track") or next(iter(self.tracks))
        if self.default_track not in self.tracks:
            raise ProfileError(f"gates.json: default_track {self.default_track!r} is not in tracks")

        # --- header / geography / publications ----------------------------------
        self.home_cities = {k.lower(): v for k, v in (g.get("home_cities") or {}).items()}
        self.publications = g.get("publications", [])
        self.header = g.get("header", {})          # name/phone/email/links may also come from env

    # ------------------------------------------------------------------ helpers
    def track_availability(self, track: str) -> str:
        if track not in self.tracks:
            raise ValueError(f"unknown track {track!r} (profile defines: {', '.join(self.tracks)})")
        return self.tracks[track].get("availability", "")

    def cover_directive(self, track: str) -> str:
        return self.tracks[track].get("cover_directive", "")

    def base_resume_env(self, track: str) -> str:
        """Env var naming the base CV for this track (falls back to RESUME_TEX_PATH)."""
        return self.tracks[track].get("base_resume_env", "RESUME_TEX_PATH")

    def header_field(self, key: str, default: str = "") -> str:
        """Header fields: env CANDIDATE_<KEY> wins, then gates.json 'header', then default."""
        return os.environ.get(f"CANDIDATE_{key.upper()}") or self.header.get(key) or default

    def rules_for(self, kind: str) -> str:
        """Sections of rules.md tagged for `kind` ("tailoring" or "cover").

        A heading like `## [tailoring] Title` routes its section; `[both]` or no tag goes
        to both. The HTML comment at the top of the file is dropped.
        """
        text = re.sub(r"<!--.*?-->", "", self.rules_text, flags=re.S)
        out = []
        for chunk in re.split(r"(?m)^(?=## )", text):
            if not chunk.strip():
                continue
            m = re.match(r"## \[(\w+)\]\s*(.*)", chunk)
            tag = m.group(1).lower() if m else "both"
            if tag in (kind, "both"):
                out.append(re.sub(r"^## \[\w+\]\s*", "## ", chunk, count=1).rstrip())
        return "\n\n".join(out).strip()


@lru_cache(maxsize=1)
def load() -> Profile:
    return Profile(PROFILE_DIR)
