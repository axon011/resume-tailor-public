"""Post-processing humanizer for resume text.

Scans LLM-generated resume content for AI writing patterns and flags them.
Based on Wikipedia's "Signs of AI writing" guide.
Focuses on patterns most relevant to resume/CV text.
"""

import re

from . import profile as _profile

_P = _profile.load()

# Pattern 1: Significance inflation
SIGNIFICANCE_WORDS = [
    "pivotal", "crucial", "vital", "key role", "testament", "underscores",
    "highlights its", "reflects broader", "symbolizing", "setting the stage",
    "marking a", "shaping the", "represents a shift", "key turning point",
    "evolving landscape", "indelible mark", "deeply rooted", "at the forefront",
]

# Pattern 3: Superficial -ing endings.
# Restricted to genuine AI tells — verbs like "demonstrating", "spearheading",
# "championing", "pioneering" are recruiter-favored action verbs in the Google XYZ
# format and should not be flagged as AI patterns.
ING_FILLERS = [
    "highlighting", "underscoring", "emphasizing",
    "reflecting", "symbolizing", "contributing to", "cultivating",
    "fostering", "encompassing", "showcasing",
]

# Pattern 4: Promotional language
PROMOTIONAL = [
    "boasts", "vibrant", "profound", "groundbreaking", "renowned",
    "breathtaking", "stunning", "cutting-edge", "state-of-the-art",
    "world-class", "best-in-class", "unparalleled", "exceptional",
    "outstanding", "remarkable", "transformative", "revolutionary",
    "innovative", "next-generation",
]

# Pattern 7: AI vocabulary
AI_VOCAB = [
    "delve", "tapestry", "landscape", "interplay", "intricate",
    "intricacies", "garner", "foster", "holistic", "synergy",
    "paradigm", "leverage", "utilize", "facilitate", "endeavor",
    "comprehensive", "robust", "seamless", "streamline",
    "empower", "elevate", "unlock", "harness",
    "proven track record", "results-driven", "deep expertise",
    "best practices", "best-in-class", "passionate",
    "good communicator", "self-motivated", "dynamic",
    "pitched as", "rigorously", "meticulously", "diligently",
    "positioning", "spearhead", "orchestrated a", "crafted",
    "ensuring", "enabling", "allowing for",
]

# Pattern 8: Copula avoidance
COPULA_AVOIDANCE = [
    "serves as", "stands as", "functions as", "acts as a",
    "marks a", "represents a", "operates as",
]

# Pattern 9: Negative parallelisms
NEGATIVE_PARALLELISMS = [
    r"not just.*?but also",
    r"not merely.*?but",
    r"it's not about.*?it's about",
    r"not only.*?but",
    r"no guessing",
    r"no wasted",
]

# Pattern 14: Em dash overuse (more than 2 per text block)
EM_DASH = "—"

# Pattern 23: Filler phrases
FILLER_PHRASES = [
    "in order to", "due to the fact that", "at this point in time",
    "in the event that", "has the ability to", "it is important to note",
    "it is worth mentioning", "it should be noted", "plays a key role",
    "with a focus on", "with a strong focus", "hands-on experience",
    "hands-on production", "in a fast-paced",
]

# Pattern 26: Overused hyphenated pairs
HYPHENATED_OVERUSE = [
    "end-to-end", "cross-functional", "client-facing", "data-driven",
    "decision-making", "high-quality", "real-time", "long-term",
    "cloud-native", "production-grade", "mission-critical",
]

# Pattern 27: Persuasive authority
AUTHORITY_TROPES = [
    "the real question is", "at its core", "in reality",
    "what really matters", "fundamentally", "the heart of",
    "the deeper issue",
]


def _check_patterns(text: str, patterns: list, pattern_name: str) -> list:
    """Check text against a list of string patterns."""
    findings = []
    text_lower = text.lower()
    for pattern in patterns:
        if pattern.lower() in text_lower:
            findings.append(f"[{pattern_name}] Found: \"{pattern}\"")
    return findings


def _check_regex(text: str, patterns: list, pattern_name: str) -> list:
    """Check text against a list of regex patterns."""
    findings = []
    for pattern in patterns:
        if re.search(pattern, text, re.IGNORECASE):
            match = re.search(pattern, text, re.IGNORECASE)
            findings.append(f"[{pattern_name}] Found: \"{match.group()}\"")
    return findings


def _check_em_dashes(text: str) -> list:
    """Flag em dash overuse (>2 per text block)."""
    count = text.count(EM_DASH)
    if count > 2:
        return [f"[Em dash overuse] Found {count} em dashes — consider using commas or periods"]
    return []


def _check_rule_of_three(text: str) -> list:
    """Detect forced triples like 'X, Y, and Z' repeated patterns."""
    # Simple heuristic: count comma-separated triples ending with "and"
    triples = re.findall(r'\w+,\s+\w+,\s+and\s+\w+', text)
    if len(triples) > 2:
        return [f"[Rule of three] Found {len(triples)} triple patterns — vary grouping sizes"]
    return []


# Trailing participle codas (", driving X", ", ensuring Y") — the single most
# detectable AI fingerprint in this pipeline's output per the 2026-07-19 recruiter
# audit: near-every bullet ended in the same rhythmic clause, duplicated across
# documents. Document-level check: >1/3 of bullets ending this way, or any two
# bullets sharing the same coda wording, gets flagged.
_TRAILING_PARTICIPLE = re.compile(
    r",\s+(driving|enabling|ensuring|demonstrating|reducing|improving|streamlining|"
    r"accelerating|supporting|allowing|providing|delivering|boosting|empowering|"
    r"enhancing|increasing|optimizing|facilitating|strengthening)\s+([^.]*?)\.?\s*$",
    re.IGNORECASE,
)


def _check_participle_cadence(bullets: list) -> list:
    """Document-level check for uniform trailing-participle bullet endings."""
    if not bullets:
        return []
    codas = []  # (location, verb, full coda text)
    for location, text in bullets:
        m = _TRAILING_PARTICIPLE.search(text)
        if m:
            codas.append((location, m.group(1).lower(), m.group(0).strip().rstrip(".").lower()))
    findings = []
    if len(codas) > max(1, len(bullets) // 3):
        verbs = ", ".join(sorted({c[1] for c in codas}))
        findings.append(
            f"[Participle cadence] {len(codas)}/{len(bullets)} bullets end in a "
            f"trailing participle clause ({verbs}) — max 1/3 allowed; vary bullet "
            f"shapes (lead with the number, end on the artifact, plain S-V-O)"
        )
    seen = {}
    for location, _verb, coda in codas:
        key = coda[:60]
        if key in seen:
            findings.append(
                f"[Participle cadence] Duplicate coda in {seen[key]} and {location}: \"{coda[:60]}\""
            )
        else:
            seen[key] = location
    return findings


def humanize_check(tailored: dict) -> list:
    """Run humanizer checks on all tailored text fields.

    Returns list of warning strings. Empty list = clean.
    """
    warnings = []

    # Collect all text to check
    texts = []
    if "summary" in tailored:
        texts.append(("Summary", tailored["summary"]))

    for i, exp in enumerate(tailored.get("experience", [])):
        for j, bullet in enumerate(exp.get("bullets", [])):
            texts.append((f"Experience[{i}].bullet[{j}]", bullet))

    for i, proj in enumerate(tailored.get("projects", [])):
        for j, bullet in enumerate(proj.get("bullets", [])):
            texts.append((f"Project[{i}].bullet[{j}]", bullet))

    for location, text in texts:
        findings = []
        findings += _check_patterns(text, SIGNIFICANCE_WORDS, "Significance inflation")
        findings += _check_patterns(text, ING_FILLERS, "Superficial -ing")
        findings += _check_patterns(text, PROMOTIONAL, "Promotional language")
        findings += _check_patterns(text, AI_VOCAB, "AI vocabulary")
        findings += _check_patterns(text, COPULA_AVOIDANCE, "Copula avoidance")
        findings += _check_regex(text, NEGATIVE_PARALLELISMS, "Negative parallelism")
        findings += _check_em_dashes(text)
        findings += _check_patterns(text, FILLER_PHRASES, "Filler phrase")
        findings += _check_patterns(text, HYPHENATED_OVERUSE, "Hyphenated overuse")
        findings += _check_patterns(text, AUTHORITY_TROPES, "Authority trope")
        findings += _check_rule_of_three(text)

        for f in findings:
            warnings.append(f"{location}: {f}")

    # Document-level cadence check across all bullets (not per-bullet)
    bullet_texts = [(loc, t) for loc, t in texts if ".bullet[" in loc]
    warnings.extend(_check_participle_cadence(bullet_texts))

    return warnings


def humanize_auto_fix(text: str) -> str:
    """Auto-fix the most common AI patterns in a text string.

    Only fixes patterns that have safe, mechanical replacements.
    More nuanced issues are left for manual review.
    """
    # Fix copula avoidance
    text = re.sub(r'\bserves as\b', 'is', text, flags=re.IGNORECASE)
    text = re.sub(r'\bstands as\b', 'is', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfunctions as\b', 'is', text, flags=re.IGNORECASE)

    # Fix filler phrases
    text = re.sub(r'\bin order to\b', 'to', text, flags=re.IGNORECASE)
    text = re.sub(r'\bdue to the fact that\b', 'because', text, flags=re.IGNORECASE)
    text = re.sub(r'\bhas the ability to\b', 'can', text, flags=re.IGNORECASE)
    text = re.sub(r'\bit is important to note that\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bit is worth mentioning that\b', '', text, flags=re.IGNORECASE)

    # Fix AI vocabulary (safe replacements only)
    text = re.sub(r'\butilize[sd]?\b', 'use', text, flags=re.IGNORECASE)
    text = re.sub(r'\bleverag(e[sd]?|ing)\b', 'use', text, flags=re.IGNORECASE)
    text = re.sub(r'\bfacilitat(e[sd]?|ing)\b', 'enable', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpitched as\b', 'built for', text, flags=re.IGNORECASE)
    text = re.sub(r'\brigorously\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\bmeticulously\b', 'carefully', text, flags=re.IGNORECASE)
    text = re.sub(r'\bpositioning (.+?) as\b', r'building \1 as', text, flags=re.IGNORECASE)
    # Note: 'ensuring' and 'excellence' intentionally NOT auto-replaced.
    # 'ensuring X' often means 'so that X holds' — replacing with 'to keep' breaks meaning.
    # 'excellence' has legitimate uses (e.g. 'Center of Excellence'). Both stay flagged via humanize_check.

    # Em dash auto-strip (2026-06-24): the LLM re-injects em dashes on nearly every
    # cover letter and many resume bullets; the check used to only FLAG them, so they
    # shipped and were hand-removed every time. Collapse any spacing around an em dash
    # (U+2014) or en dash (U+2013) into a comma so it can never reach a PDF.
    text = re.sub(r'\s*[—–]\s*', ', ', text)
    # Collapse an accidental ", ," and space-before-comma from the above.
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s+,', ',', text)

    # Sentence auto-strip: any sentence carrying a phrase the profile lists under
    # auto_strip_sentences is dropped whole. The regression gate hard-bans those
    # phrases, but the cover LLM keeps re-adding them (the original case was an
    # immigration-status claim the candidate must never make), so the sentence goes
    # here and never trips the gate or ships.
    for _pat in _P.auto_strip_sentences:
        _p = "(?:" + _pat.pattern + ")"
        text = re.sub(r'(?<![.!?])\s*[^.!?]*' + _p + r'[^.!?]*[.!?]', '', text, flags=re.IGNORECASE)
        text = re.sub(_p + r'[^.!?]*[.!?]', '', text, flags=re.IGNORECASE)

    # Production-framing auto-fix: the LLM re-injects "production" framing on a
    # personal-project RAG bullet or cover paragraph on nearly every generation, and
    # the regression gate blocked the compile every time, so it is auto-fixed here to
    # let runs pass first-pass. It mirrors regression.py's allowlist exactly: a RAG
    # sentence is rewritten ONLY when no production_ok employer marker (from the
    # profile) is present, i.e. when it reads as a personal repo. A true claim about a
    # real employer deploy is left alone.
    if (re.search(r'\bRAG\b', text, re.IGNORECASE)
            and not _P.production_ok_re.search(text)):
        # 'robust' was the old replacement and is itself a banned AI tell — use 'reliable'.
        text = re.sub(r'\bproduction[- ]grade\b', 'reliable', text, flags=re.IGNORECASE)
        text = re.sub(r'\bproduction[- ]ready\b', 'deployment-ready', text, flags=re.IGNORECASE)
        text = re.sub(r'\bproduction readiness\b', 'internal release readiness', text, flags=re.IGNORECASE)
        text = re.sub(r'\bproduction reliability\b', 'deployment reliability', text, flags=re.IGNORECASE)
        text = re.sub(r'\bproduction (anomal\w+)\b', r'process \1', text, flags=re.IGNORECASE)   # manufacturing sense
        text = re.sub(r'\bproduction (quality )?monitoring\b', r'\1monitoring', text, flags=re.IGNORECASE)
        # Phrase swaps that stay grammatical whether "production" reads as a noun
        # ("deployed to production") or gets a noun after it ("in production with
        # guardrails"): "to production" -> "to release", "in production" -> "in
        # live use". Run these before the catch-all.
        text = re.sub(r'\bto production\b', 'to release', text, flags=re.IGNORECASE)
        text = re.sub(r'\bin production\b', 'in live use', text, flags=re.IGNORECASE)
        # residual adjectival "production X" (services/pipelines/systems/AI ...) -> "internal X"
        text = re.sub(r'\bproduction\b', 'internal', text, flags=re.IGNORECASE)

    # NOTE: the fabricated "reducing ... 60% of manual reporting" metric the cover
    # LLM keeps inventing is handled by a GATE BLOCK in regression.py, not here —
    # auto-deleting it mid-sentence left broken grammar ("...pipeline that."), and
    # a fabrication deserves a proper human rephrase, not a lossy strip.

    # Clean up double spaces
    text = re.sub(r'  +', ' ', text).strip()

    return text
