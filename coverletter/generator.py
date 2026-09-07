"""Generate humanized cover letters using LLM."""

import json
import re
from datetime import datetime

import config
from pipeline.tailoring import get_llm_client
from pipeline.humanizer import (
    humanize_auto_fix,
    _check_patterns,
    _check_em_dashes,
    AI_VOCAB,
    FILLER_PHRASES,
    HYPHENATED_OVERUSE,
)
from ingest.company_research import fetch_company_context
from . import prompts as cl_prompts
from pipeline import profile as _profile

_P = _profile.load()


def generate_cover_letter(tailored_resume: dict, job_info: dict, track: str = None) -> dict:
    """Generate a cover letter based on tailored resume and job info.

    ``track`` selects the availability story from the profile's ``tracks``. The
    stories must never be mixed in one letter.

    Returns dict with 'date', 'greeting', 'paragraphs' keys.
    """
    track = track or config.DEFAULT_TRACK
    client = get_llm_client()

    # Extract top experience bullets and projects for the prompt
    exp_bullets = ""
    if tailored_resume["experience"]:
        bullets = tailored_resume["experience"][0].get("bullets", [])[:3]
        exp_bullets = "\n".join(f"- {b}" for b in bullets)

    projects_str = ""
    for proj in tailored_resume["projects"][:2]:
        projects_str += f"\n{proj['title']} ({proj['tech_stack']})\n"
        for b in proj["bullets"][:2]:
            projects_str += f"  - {b}\n"

    # Pull a lightweight snippet from the company's homepage so the letter can
    # cite something specific instead of parroting the JD. Caller can override
    # by passing job_info["company_url"]; otherwise we try a best-effort guess.
    company_name = job_info.get("company", "")
    company_url = job_info.get("company_url") or _guess_company_url(company_name)
    company_research = fetch_company_context(company_url) if company_url else ""
    if not company_research:
        company_research = "(no additional company research available — use JD only)"
    else:
        print(f"   Company research: fetched {len(company_research)} chars from {company_url}")

    prompt = cl_prompts.COVER_LETTER_PROMPT.format(
        job_title=job_info["title"],
        company=job_info.get("company", "the company"),
        job_description=job_info["description"][:3000],
        summary=tailored_resume["summary"],
        experience_bullets=exp_bullets,
        projects=projects_str,
        candidate_name=config.CANDIDATE_NAME,
        candidate_location=config.CANDIDATE_LOCATION,
        candidate_degree=config.CANDIDATE_DEGREE,
        company_research=company_research,
    )

    # The cover-letter model call intermittently returns malformed JSON on the
    # claude-code backend (e.g. "Expecting ',' delimiter" mid-object) — this
    # crashed the whole Lampenwelt #316 run TWICE on 2026-07-04, discarding an
    # already-clean resume. Retry up to 3x and repair-parse before giving up so
    # a single flaky response no longer throws away the run.
    data = None
    last_err = None
    for attempt in range(1, 4):
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": cl_prompts.COVER_LETTER_SYSTEM
                                              + cl_prompts.AVAILABILITY_DIRECTIVE[track]},
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=4096,
        )

        raw = (response.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1]
            if raw.endswith("```"):
                raw = raw[:-3]
            raw = raw.strip()
        # Trim to the first "{" so leading commentary can't break the decode.
        start = raw.find("{")
        if start > 0:
            raw = raw[start:]

        # A trailing comma before "}" / "]" is the claude backend's most common
        # defect (tailoring.py, 2026-09-06); here it would burn one of three LLM
        # retries. Same regex as pipeline.tailoring._strip_trailing_commas.
        raw = re.sub(r",(\s*[}\]])", r"\1", raw)
        try:
            # raw_decode tolerates trailing commentary after the first valid
            # JSON value (GLM-4.7 "Extra data"); the retry handles mid-object
            # corruption the claude backend produces intermittently.
            data, _ = json.JSONDecoder().raw_decode(raw)
            break
        except json.JSONDecodeError as e:
            last_err = e
            print(f"   Cover-letter JSON parse failed (attempt {attempt}/3): {e}; retrying...")

    if data is None:
        raise RuntimeError(
            f"Cover-letter generation returned unparseable JSON after 3 attempts "
            f"(last error: {last_err}). The resume itself was fine — re-run with "
            f"--cover-letter, or generate resume-only and write the email body by hand."
        )

    paragraphs = data.get("paragraphs", [])
    paragraphs = [_humanize_paragraph(p) for p in paragraphs]
    paragraphs = _split_employer_attribution(paragraphs)
    _warn_on_ai_patterns(paragraphs)

    return {
        "date": datetime.now().strftime("%B %d, %Y"),
        "greeting": data.get("greeting", "Dear Hiring Team,"),
        "paragraphs": paragraphs,
    }


# Attribution-slip auto-split: the regression gate bans an employer mention and a
# personal-repo mention (the profile's cover_split_vocabulary) sharing ONE line —
# the .tex puts each cover paragraph on its own line, so the fix is to move the two
# mentions into SEPARATE paragraphs. The LLM re-injected this on nearly every cover
# (hand-split on a dozen letters before this existed). This does the split
# structurally so the gate passes first-pass. If a single SENTENCE contains both
# tokens it can't be split cleanly, so it's left for the gate to flag (rare; needs a
# human rephrase).
_EMPLOYER_RE = _P.primary_employer_re
_PERSONAL_REPO_RE = _P.cover_split_re


def _split_employer_attribution(paragraphs: list) -> list:
    out = []
    for p in paragraphs:
        if not (_EMPLOYER_RE.search(p) and _PERSONAL_REPO_RE.search(p)):
            out.append(p)
            continue
        # Promote an inner ';' between the two topics to a sentence boundary so
        # "...personal work; at the employer..." can be split, then capitalize.
        candidate = re.sub(r";\s+(?=\w)", ". ", p)
        candidate = re.sub(r"(?<=\.\s)([a-z])", lambda m: m.group(1).upper(), candidate)
        sents = re.split(r"(?<=[.!?])\s+", candidate)
        # If any single sentence still carries BOTH tokens, we can't safely split.
        if any(_EMPLOYER_RE.search(s) and _PERSONAL_REPO_RE.search(s) for s in sents):
            out.append(p)
            continue
        # Group consecutive sentences into chunks that never mix the two topics.
        chunks, cur = [], []

        def _tag(s):
            if _EMPLOYER_RE.search(s):
                return "employer"
            if _PERSONAL_REPO_RE.search(s):
                return "personal"
            return None
        cur_tag = None
        for s in sents:
            t = _tag(s)
            if t and cur_tag and t != cur_tag:
                chunks.append(" ".join(cur).strip())
                cur = []
            if t:
                cur_tag = t
            cur.append(s)
        if cur:
            chunks.append(" ".join(cur).strip())
        out.extend(c for c in chunks if c)
    return out


# Contractions to enforce a human voice. Only apply when the formal form is
# clearly being used as the subject-verb pair; avoid touching quoted material.
# Negative lookahead on "I have/am" guards against breaking idioms like
# "I have to", "I am to" which would otherwise contract to broken English.
_CONTRACTIONS = [
    (r"\bI have(?!\s+to\b)\b", "I've"),
    (r"\bI am(?!\s+to\b)\b", "I'm"),
    (r"\bI will\b", "I'll"),
    (r"\bI would\b", "I'd"),
    (r"\bit is\b", "it's"),
    (r"\bthat is\b", "that's"),
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bcannot\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bwould not\b", "wouldn't"),
]

# Template closers that signal AI — replace with something conversational.
_CLOSER_REPLACEMENTS = [
    (r"I'?d love to discuss how this (experience|background) fits your roadmap\.?",
     "Happy to walk through any of this."),
    (r"I look forward to (the opportunity to )?discuss(ing)?[^.]*\.",
     "Happy to chat whenever works."),
    (r"I look forward to hearing (back )?from you\.?",
     "Let me know if the fit makes sense."),
]


def _humanize_paragraph(text: str) -> str:
    """Humanize a single paragraph: auto-fix AI patterns, add contractions,
    replace template closers."""
    import re
    text = humanize_auto_fix(text)
    for pattern, replacement in _CONTRACTIONS:
        text = re.sub(pattern, replacement, text)
    for pattern, replacement in _CLOSER_REPLACEMENTS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def _warn_on_ai_patterns(paragraphs: list) -> None:
    """Print warnings for any remaining AI-tell patterns in the paragraphs."""
    joined = " ".join(paragraphs)
    findings = []
    findings += _check_patterns(joined, AI_VOCAB, "AI vocabulary")
    findings += _check_patterns(joined, FILLER_PHRASES, "Filler phrase")
    findings += _check_patterns(joined, HYPHENATED_OVERUSE, "Hyphenated overuse")
    findings += _check_em_dashes(joined)
    if findings:
        print(f"   Cover letter AI patterns flagged ({len(findings)}):")
        for f in findings[:5]:
            print(f"   - {f}")


def _guess_company_url(company: str) -> str:
    """Cheap best-effort guess at the company's homepage from its name.

    Returns the .de TLD when the company name has a German legal suffix
    (GmbH, AG, UG, SE), otherwise .com. fetch_company_context follows
    redirects so this only needs to be roughly right.
    """
    if not company:
        return ""
    slug = company.lower().strip()
    # Detect German legal suffix BEFORE stripping it — drives TLD choice.
    de_suffixes = (" gmbh", " gmbh & co. kg", " ag", " ug", " se")
    is_german = slug.endswith(de_suffixes)
    # Strip common suffixes that rarely appear in domains
    for suffix in list(de_suffixes) + [" ltd", " inc", " llc"]:
        if slug.endswith(suffix):
            slug = slug[: -len(suffix)].strip()
    # Domain can't have spaces / punctuation
    slug = "".join(c for c in slug if c.isalnum() or c == "-")
    if not slug:
        return ""
    tld = "de" if is_german else "com"
    return f"https://{slug}.{tld}"
