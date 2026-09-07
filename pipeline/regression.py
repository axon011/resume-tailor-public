"""Deterministic pre-ship regression gate.

Scans the FINAL rendered LaTeX (resume and cover letter) for banned strings and
attribution leaks the prompts are supposed to prevent but the LLM still emits.
Hard-fails the run before compile, so a banned claim never reaches a shippable PDF.

Every candidate-specific fact comes from the profile (candidate/gates.json):
  banned_phrases      phrases banned regardless of context (pattern + reason)
  employers           which employers had real production deploys ("production_ok")
  personal_projects   vocabulary that must never share a sentence with the employer
  realtime_ok         where "real-time" is a true claim (e.g. MQTT sensor streams)
  real_metric_values  the only evaluation scores that exist

The RULES are generic and live here; the FACTS are yours and live in the profile.
"""

import re

from . import profile as _profile


class RegressionError(Exception):
    """Raised when the rendered LaTeX contains a banned claim. Aborts the run."""


_P = _profile.load()

# "production" belongs only to employers the profile marks production_ok. Personal
# repos are not production, and the puffery variants (production-grade / -quality /
# -rigor) follow the same allowlist rather than an unconditional ban: they are true of
# a real deploy and false of a hobby repo, exactly like the bare word.
_PRODUCTION = re.compile(r"\bproduction\w*", re.I)
_PRODUCTION_PUFFERY = re.compile(r"production[\s-]?(?:grade|quality|rigor)", re.I)
_PRODUCTION_OK = _P.production_ok_re
_PRODUCTION_NAMES = " or ".join(_P.production_ok_names) or "no employer"

# Attribution leak: claiming a PERSONAL project as employer work. Checked per
# SENTENCE, not per line: a summary that says "two years at <employer> ... on
# personal projects I build multi-agent orchestration" is correctly attributed, and a
# per-line rule fails the build on it.
_EMPLOYER = _P.primary_employer_re
_PERSONAL_REPO = _P.personal_repo_re
# Phrases that explicitly assign the work to the candidate personally. Their presence
# in the same sentence is what makes the employer/project co-occurrence correct.
# Kept IDENTICAL to claimgate.py's copy so the two gates agree on what "personal" means.
_PERSONAL_MARKER = re.compile(
    r"(?:my\s+own|personal|side|own)\s+(?:project|repo|time)\w*"
    r"|personally\s+(?:built|build|wrote|developed)"
    r"|\bindependently\b|\bon\s+my\s+own\b|\boutside\s+(?:of\s+)?work\b"
    + (r"|\b" + _P.github_marker + r"\b" if _P.github_marker else "")
    # German markers: a German-language letter states ownership as "eigene", "selbst
    # gebaut", "in meiner Freizeit" — without these every German cover letter false-fails.
    + r"|\beigene[nrsm]?\b|\beigenständig\b|\bprivat(?:es|en|e)?\b"
    r"|\bin\s+meiner\s+Freizeit\b|\bneben\s+(?:der\s+)?Arbeit\b"
    r"|\bselbst\s+(?:gebaut|entwickelt|geschrieben|implementiert)\b", re.I
)

# "real time" in all three spellings. Scoped, not banned outright: the profile names
# the context where it is true (sensor streams, brokers); anywhere else it is a
# latency overclaim (a dashboard that polls and batches is not real-time).
_REALTIME = re.compile(r"\breal[\s-]?time\b", re.I)
_REALTIME_OK = _P.realtime_ok_re

# Section tracking: bullets under \section{Projects} are personal by construction,
# so the attribution rule does not apply inside that block.
_SECTION = re.compile(r"\\section\{([^}]*)\}")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# A retrieval metric must name its retrieval context: "0.94 hit@5" relabelled as generic
# "ML evaluation" on a computer-vision resume is true-but-misleading and collapses the
# moment an interviewer asks what model scored hit@5.
_RETRIEVAL_METRIC = re.compile(r"hit@\d|presence\s+accuracy|citation\s+presence|mrr@|ndcg", re.I)
_RETRIEVAL_CONTEXT = re.compile(r"\bRAG\b|retriev|search|rerank|BM25|embedding|vector", re.I)

# Metric INVENTION. The profile lists the only evaluation scores that exist; any other
# decimal in [0,1] next to an eval-metric word is fabricated (the "faithfulness 0.82,
# relevance 0.89" pair is the classic).
_REAL_METRIC_VALUES = _P.real_metric_values
_METRIC_WORD = (
    r"faithful\w*|answer\s+relevanc\w*|relevanc\w*|groundedness|hallucination"
    r"|context\s+(?:precision|recall)|precision|recall|\bf1\b|accuracy|hit@\d|mrr@|ndcg"
    r"|ragas|correctness|score"
)
_METRIC_VALUE = re.compile(
    r"(?:" + _METRIC_WORD + r")[^.\n]{0,30}?\(?\b(0\.\d{1,3})\b"
    r"|\b(0\.\d{1,3})\b[^.\n]{0,30}?(?:" + _METRIC_WORD + r")",
    re.I,
)


def check_regressions(latex: str, label: str = "document", dump_path: str | None = None) -> None:
    """Scan rendered LaTeX; raise RegressionError if any banned claim is present.

    Comment lines (LaTeX '%' preamble) are skipped so template comments never
    false-positive. On failure, optionally dump the offending .tex to dump_path.
    """
    violations = []
    in_projects = False
    for raw_line in latex.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        section = _SECTION.search(line)
        if section:
            in_projects = "project" in section.group(1).lower()
        for pat, reason in _P.banned_phrases:
            if pat.search(line):
                violations.append((reason, line))
        # Scoped per sentence so one employer mention cannot launder a claim made
        # elsewhere in the same paragraph, and so a correctly attributed clause is
        # not condemned by its neighbour.
        for sentence in _SENTENCE_SPLIT.split(line):
            if not sentence.strip():
                continue
            if _PRODUCTION.search(sentence) and not _PRODUCTION_OK.search(sentence):
                _what = ("'production-grade'/'production quality'/'production rigor'"
                         if _PRODUCTION_PUFFERY.search(sentence) else "'production'")
                violations.append(
                    (f"{_what} outside the {_PRODUCTION_NAMES} roles — those had real "
                     "production deploys, the personal repos did not", sentence)
                )
            if _REALTIME.search(sentence) and not (_REALTIME_OK and _REALTIME_OK.search(sentence)):
                violations.append(
                    ("'real-time' / 'real time' / 'realtime' claimed outside the profile's "
                     "real-time context — latency overclaim", sentence)
                )
            if (_EMPLOYER.search(sentence) and _PERSONAL_REPO.search(sentence)
                    and not _PERSONAL_MARKER.search(sentence) and not in_projects):
                violations.append(
                    (f"personal project claimed as {_P.primary_employer} work (attribution slip)", sentence)
                )
        for _m in _METRIC_VALUE.finditer(line):
            _val = _m.group(1) or _m.group(2)
            if _val not in _REAL_METRIC_VALUES:
                violations.append(
                    ("invented evaluation metric value " + _val + " -- the only real eval numbers "
                     "are " + (", ".join(sorted(_REAL_METRIC_VALUES)) or "none") +
                     " (profile real_metric_values); every other score is fabricated", line)
                )
        if _RETRIEVAL_METRIC.search(line) and not _RETRIEVAL_CONTEXT.search(line):
            violations.append(
                ("retrieval metric (hit@k / presence accuracy) presented without its "
                 "RAG/retrieval context — metric relabeling", line)
            )

    if not violations:
        return

    seen, unique = set(), []
    for v in violations:
        if v in seen:
            continue
        seen.add(v)
        unique.append(v)

    if dump_path:
        try:
            with open(dump_path, "w", encoding="utf-8") as f:
                f.write(latex)
        except OSError:
            dump_path = None

    raise RegressionError(_format(label, unique, dump_path))


def _format(label: str, violations: list, dump_path: str | None) -> str:
    out = [
        f"REGRESSION GATE FAILED for {label}: {len(violations)} banned claim(s) "
        f"would ship. Nothing was compiled.",
        "",
    ]
    for i, (reason, ctx) in enumerate(violations, 1):
        snippet = ctx if len(ctx) <= 160 else ctx[:157] + "..."
        out.append(f"  {i}. {reason}")
        out.append(f"     -> {snippet}")
    out.append("")
    if dump_path:
        out.append(f"Offending LaTeX dumped to: {dump_path}")
    out.append("Fix: re-run to re-roll the LLM, or edit the dumped .tex and compile manually.")
    return "\n".join(out)
