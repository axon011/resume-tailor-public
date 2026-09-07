import os
import datetime as _dt
from dotenv import load_dotenv

load_dotenv()

from pipeline import profile as _profile  # noqa: E402  (after load_dotenv: CANDIDATE_DIR may come from .env)

_P = _profile.load()

LLM_API_KEY = os.getenv("LLM_API_KEY")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.z.ai/api/coding/paas/v4")
LLM_MODEL = os.getenv("LLM_MODEL", "glm-4.7")
PDFLATEX_PATH = os.getenv(
    "PDFLATEX_PATH",
    "pdflatex",  # Override with PDFLATEX_PATH env var for your system
)
GITHUB_USERNAME = os.getenv("GITHUB_USERNAME", "your-github-username")
RESUME_TEX_PATH = os.getenv(
    "RESUME_TEX_PATH",
    os.path.join(os.path.dirname(__file__), "templates", "resume.tex.j2"),  # Override with RESUME_TEX_PATH env var
)


def base_resume_path(track: str) -> str:
    """Base CV for a track. A track may name its own env var (profile `base_resume_env`),
    e.g. a part-time variant whose tagline and section order differ; unset falls back
    to RESUME_TEX_PATH so every track always has a base."""
    return os.getenv(_P.base_resume_env(track)) or RESUME_TEX_PATH

# Candidate details (used in cover letter generation). Env wins, then the profile's
# "header" block. Hard-fail rather than silently producing a generic letter.
CANDIDATE_NAME = _P.header_field("name")
CANDIDATE_LOCATION = _P.header_field("location")
CANDIDATE_DEGREE = _P.header_field("degree")
_required = {"CANDIDATE_NAME": CANDIDATE_NAME, "CANDIDATE_LOCATION": CANDIDATE_LOCATION,
             "CANDIDATE_DEGREE": CANDIDATE_DEGREE, "LLM_API_KEY": LLM_API_KEY}
_missing = [k for k, v in _required.items() if not v]
if _missing:
    raise RuntimeError(
        f"Missing required env vars: {', '.join(_missing)}. "
        "Copy .env.example to .env and fill them in (or set them under \"header\" in candidate/gates.json)."
    )

# Optional personal context injected into the tailoring system prompt.
# Use this to give the LLM concrete, specific anchors (tech stack, tone, domain)
# so tailored summaries sound like the actual candidate, not a generic AI Engineer.
# Keep it short (under 500 chars). Loaded from CANDIDATE_CONTEXT env var.
CANDIDATE_CONTEXT = os.getenv("CANDIDATE_CONTEXT", "").strip()

# Prefix for generated resume/cover letter filenames, e.g. "Jane_Doe" -> Jane_Doe_acme_20260417.pdf
OUTPUT_FILENAME_PREFIX = os.getenv("OUTPUT_FILENAME_PREFIX", "Tailored_Resume")

# ---------------------------------------------------------------------------
# TRACKS
# ---------------------------------------------------------------------------
# A profile may run several job searches in parallel that need DIFFERENT availability
# stories (the original case: a part-time student track selling remaining runway and a
# full-time track selling availability from graduation). Saying both in one document is
# the conflict this split exists to remove. The stories live in the profile under
# `tracks.<name>.availability` and may use two placeholders:
#
#   {thesis_submission}  the THESIS_SUBMISSION env var, verbatim (the only place a
#                        completion date is stated — never write a longer runway than
#                        it supports; change the env var instead)
#   {in_two_months}      month + year two calendar months from TODAY, computed at
#                        generation time. A stored month goes stale within weeks and
#                        understates the runway; this never does.
THESIS_SUBMISSION = os.getenv("THESIS_SUBMISSION", "").strip()

_MONTHS = ("January", "February", "March", "April", "May", "June",
           "July", "August", "September", "October", "November", "December")


def _in_two_months(today=None) -> str:
    """Month + year, two calendar months from today."""
    d = today or _dt.date.today()
    m0 = d.month - 1 + 2
    return f"{_MONTHS[m0 % 12]} {d.year + m0 // 12}"


FULLTIME_AVAILABLE_FROM = _in_two_months()


class _KeepUnknown(dict):
    """str.format_map helper: an unknown {placeholder} is left as written, not an error."""
    def __missing__(self, key):
        return "{" + key + "}"


_FMT = _KeepUnknown(
    thesis_submission=THESIS_SUBMISSION or "(set THESIS_SUBMISSION in .env)",
    in_two_months=FULLTIME_AVAILABLE_FROM,
)

AVAILABILITY = {name: _P.track_availability(name).format_map(_FMT) for name in _P.tracks}
DEFAULT_TRACK = _P.default_track


def candidate_context(track: str = DEFAULT_TRACK) -> str:
    """CANDIDATE_CONTEXT with exactly one availability story — the track's.

    CANDIDATE_CONTEXT itself must carry NO availability sentence: that is how two
    contradictory stories end up in one document. Keep availability in the profile.
    """
    if track not in AVAILABILITY:
        raise ValueError(f"unknown track {track!r} (expected one of {', '.join(AVAILABILITY)})")
    story = AVAILABILITY[track]
    if not story:
        return CANDIDATE_CONTEXT
    return f"{CANDIDATE_CONTEXT.strip()}\n\nAVAILABILITY ({track}): {story}".strip()
