"""Omission gate — reports what the tailored resume DROPPED from the base CV.

Every other gate in this pipeline checks for OVERCLAIMING: regression.py catches
banned framing and invented metrics, claimgate.py catches skills with no backing in
the base. Nothing checked the opposite direction, so content could silently vanish
between the base CV and the shipped PDF and no run would say a word.

That blind spot has cost real applications:
  * the renderer's escaped-dict-key bug blanked every project's tech stack for
    ~2.5 months (327 of 338 resumes, 2026-06-13 .. 08-29) and only surfaced when a
    keyword score looked oddly weak — see feedback_renderer_escaped_dict_keys;
  * one package dropped the candidate's only live deployment while covering RAG
    twice. Caught by hand. A later measurement showed that was not a one-off: the
    live project shipped in 0 of 108 resumes in one month.

DESIGN NOTE — why this WARNS and does not BLOCK.
Dropping content is often correct: rule 3 of the system prompt explicitly licenses
dropping irrelevant projects, and bullets are meant to be rephrased. A hard failure
here would fail the build on good tailoring. So this gate prints and returns; only
the caller decides what to do. It is a spotlight, not a wall.

DESIGN NOTE — why matching is fuzzy.
An earlier hand-audit of this exact question reported "the employer's RAG-eval bullet
only survives in 5% of resumes". That was WRONG, and wrong in an instructive way: it
matched the base bullet text verbatim, but rule 1 of the system prompt explicitly
permits rephrasing. Measured on meaning instead, the same bullet survives in 94-100%.
So bullet matching here is by DISTINCTIVE CONTENT TOKENS (proper nouns, tech, numbers),
never by string equality. Anything stricter manufactures false alarms.
"""

from __future__ import annotations

import re

from . import profile as _profile

_P = _profile.load()

# Tokens that carry identity: capitalised words, tech spellings, numbers, acronyms.
# Deliberately excludes ordinary verbs/adjectives, which rephrasing churns freely.
_DISTINCTIVE = re.compile(
    r"[A-Z][a-zA-Z0-9+#.]{2,}"          # Acme, FastAPI, Go, MQTT, GA4
    r"|\b\d+[\w%./]*"                    # 0.94, 23, 4-bit, 60%
    r"|\b(?:[a-z]+[A-Z][a-zA-Z]*)\b"     # camelCase spellings
)

# Stopwords among capitalised tokens — common sentence-initial words carry no identity.
_NOT_IDENTIFYING = {
    "Built", "Developed", "Owned", "Led", "Designed", "Implemented", "Created",
    "Measured", "Delivered", "Managed", "Maintained", "Engineered", "Architected",
    "Deployed", "Automated", "Integrated", "Migrated", "Orchestrated", "Reduced",
    "Optimized", "Optimised", "The", "This", "That", "With", "From", "Into", "And",
    "For", "Their", "These", "Its", "Also", "Which", "While", "When", "Using",
}

# The live deployment, and the JD signals that make omitting it a real mistake.
_LIVE_PROJECT = _P.live_project_re          # None when the profile names no live project
# 2026-09-01: `production` and `deployed` were REMOVED from this list. They fired on the
# one run — a JD about agent memory and retrieval that says "production" a dozen
# times in the sense of "production AI-agent systems", with no frontend signal anywhere.
# Nearly every AI JD says "production", so those two words made this warning fire almost
# always, and a warning that always fires is one nobody reads. The generator's own
# reasoning was correct there ("pin rule not triggered: JD has no TypeScript/frontend/
# web-app/streaming signal") and this gate contradicted it. The pin is about FRONTEND and
# DEPLOYED-PRODUCT evidence, so trigger only on signals that actually imply those.
_LIVE_TRIGGERS = re.compile(
    r"\btypescript\b|\bjavascript\b|\breact\b|\bnext\.?js\b|\bfrontend\b|\bfront[\s-]end\b"
    r"|\bfull[\s-]?stack\b|\bweb app\b|\bweb application\b|\bUI/UX\b|\bSSE\b"
    r"|\bserver[\s-]sent\b|\bchat (?:interface|app|application|ui)\b"
    r"|\buser[\s-]facing\b|\bcustomer[\s-]facing\b|\blive (?:service|app|product)\b",
    re.I,
)

_PROJECT_RX = re.compile(r"\\projecttitle\{(.+?)\}\{(.*?)\}", re.S)
_ITEM_RX = re.compile(r"\\item\s+(.+?)(?=\n\s*\\item|\n\s*\\end\{itemize\})", re.S)
_SKILLCAT_RX = re.compile(r"\\skillcat\{(.+?)\}\{(.*?)\}", re.S)
# Mirrors parser.py::_SECTION_RX so both agree on where a section ends.
_SECTION_RX = re.compile(r"\\section\{([^}]*)\}(.*?)(?=\\section\{|\\end\{document\})", re.S)


def _section(tex: str, name: str) -> str:
    """Body of the section whose heading CONTAINS `name` (loose, like parser.py).

    Loose on purpose: the base CV titles its sections "Professional Experience" and
    "Technical Skills", and exact matching is what broke parse_resume on 2026-08-28.
    """
    want = name.lower()
    for m in _SECTION_RX.finditer(tex):
        if want in m.group(1).lower():
            return m.group(2)
    return ""


def _strip_tex(s: str) -> str:
    """Flatten LaTeX to comparable prose."""
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = s.replace("\\", " ").replace("{", " ").replace("}", " ")
    s = s.replace("~", " ").replace("&", "and")
    return re.sub(r"\s+", " ", s).strip()


def _identity_tokens(text: str) -> set[str]:
    toks = set()
    for m in _DISTINCTIVE.finditer(text):
        t = m.group(0).strip(".,;:")
        if len(t) > 1 and t not in _NOT_IDENTIFYING:
            toks.add(t.lower())
    return toks


def _project_key(title_tex: str) -> str:
    """Stable short key for a project: text before any ' --- ' or link decoration."""
    t = _strip_tex(title_tex)
    t = re.split(r"\s+---\s+|\s*\|\s*", t)[0]
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


# --- employer-attribution integrity (added 2026-09-06) -------------------------------
# The employer block is rephrased on nearly every generation and NOTHING caught it.
# Six consecutive packages once shipped an altered version of a load-bearing bullet,
# and one silently INSERTED "agentic" into the employer's product description. This
# check is phrase-exact and direction-aware instead - it asks whether specific clauses
# SURVIVED, and whether specific words were ADDED.
_JOBTITLE_RX = re.compile(
    r"\\jobtitle\{[^}]*\}\{([^}]*)\}(.*?)(?=\\jobtitle\{|\\section\{|\\end\{document\})",
    re.S)

# Clauses that must survive verbatim inside a given employer's block. Keyed by a lowercase
# substring of the employer name as it appears in \jobtitle{...}{EMPLOYER}.
_PROTECTED_CLAUSES = _P.protected_clauses

# Words that must never be ADDED to an employer's block when the base does not have them.
# Keyed like protected_clauses; both come from the profile's employers[].
_BANNED_ADDITIONS = _P.banned_additions


def _employer_blocks(tex: str) -> dict:
    """Map a lowercase employer key -> the raw tex of that employer's experience block."""
    out = {}
    for m in _JOBTITLE_RX.finditer(_section(tex, "experience") or tex):
        employer = _strip_tex(m.group(1)).lower()
        for key in set(list(_PROTECTED_CLAUSES) + list(_BANNED_ADDITIONS)):
            if key in employer:
                out[key] = out.get(key, "") + " " + m.group(2)
    return out


def _parse(tex: str):
    projects, bullets, skills = {}, [], {}
    for m in _PROJECT_RX.finditer(tex):
        projects[_project_key(m.group(1))] = (_strip_tex(m.group(1)), _strip_tex(m.group(2)))
    # Bullets are read from the EXPERIENCE section ONLY. Reading every \item in the
    # document made a legitimately-dropped project report its bullets a second time as
    # "experience bullet missing" — the project drop is already reported on its own.
    for m in _ITEM_RX.finditer(_section(tex, "experience")):
        b = _strip_tex(m.group(1))
        if len(b) > 25:
            bullets.append(b)
    for m in _SKILLCAT_RX.finditer(tex):
        cat = _strip_tex(m.group(1))
        vals = [v.strip() for v in re.split(r",(?![^(]*\))", _strip_tex(m.group(2))) if v.strip()]
        skills[cat.lower()] = vals
    return projects, bullets, skills


def check_omissions(tailored_tex: str, base_tex_path: str, label: str = "",
                    jd_text: str = "") -> list[str]:
    """Compare a rendered resume against the base CV and report dropped content.

    Returns a list of human-readable warnings (empty means nothing notable was lost).
    Never raises on content grounds and never exits — see the module docstring.
    """
    try:
        with open(base_tex_path, encoding="utf-8", errors="replace") as f:
            base_tex = f.read()
    except OSError as e:
        return [f"omission gate could not read the base CV ({e}) — skipped"]

    b_proj, b_bullets, b_skills = _parse(base_tex)
    t_proj, t_bullets, t_skills = _parse(tailored_tex)
    warnings: list[str] = []

    # --- 1. the live deployment, when the JD asks for exactly what it demonstrates ---
    if (jd_text and _LIVE_PROJECT is not None and _LIVE_TRIGGERS.search(jd_text)
            and not _LIVE_PROJECT.search(tailored_tex)):
        hits = sorted({m.group(0).lower() for m in _LIVE_TRIGGERS.finditer(jd_text)})
        warnings.append(
            "LIVE DEPLOYMENT DROPPED: " + _P.live_project_name + " is absent, but the "
            "JD asks for " + ", ".join(hits[:6]) + ". It is the only personal project that is "
            "genuinely deployed and the strongest evidence of a running system in the base. "
            "See system-prompt rule 3a."
        )

    # --- 2. projects that vanished, and whether their tech went with them ---
    dropped = [k for k in b_proj if k not in t_proj]
    if dropped:
        names = [b_proj[k][0][:46] for k in dropped]
        warnings.append(
            f"projects dropped ({len(dropped)}/{len(b_proj)}): " + "; ".join(names) +
            "  [legitimate if genuinely off-lane — rule 3 permits it; flagged so the choice is visible]"
        )

    # --- 3. tech-stack blanking (the 2026-06-13 renderer bug's signature) ---
    blanked = [t_proj[k][0][:46] for k in t_proj if not t_proj[k][1].strip()]
    if blanked:
        warnings.append(
            "TECH STACK EMPTY on " + str(len(blanked)) + " shipped project(s): " +
            "; ".join(blanked) + ".  This is the signature of the renderer bug fixed "
            "2026-08-30 (escaped dict keys -> proj.tech_stack undefined). If it is back, "
            "check pipeline/renderer.py::_escape_tree before shipping."
        )

    # --- 4. experience bullets whose distinctive content did not survive anywhere ---
    tailored_tokens = _identity_tokens(" ".join(t_bullets))
    lost = []
    for b in b_bullets:
        bt = _identity_tokens(b)
        if len(bt) < 3:
            continue
        # rephrasing is expected; treat a bullet as lost only when almost none of its
        # identifying content appears anywhere in the tailored bullets.
        kept = len(bt & tailored_tokens) / len(bt)
        if kept < 0.34:
            lost.append((round(kept * 100), b[:88]))
    for pct, b in lost:
        warnings.append(f"experience bullet largely absent ({pct}% of its distinctive terms survive): {b}...")

    # --- 5. base skills that did not make it through ---
    b_all = {s.lower(): s for cat in b_skills.values() for s in cat}
    t_blob = _strip_tex(tailored_tex).lower()
    missing = [orig for low, orig in sorted(b_all.items()) if low not in t_blob]
    if missing:
        warnings.append(
            f"base skills not present anywhere in the output ({len(missing)}/{len(b_all)}): " +
            ", ".join(missing[:14]) + ("..." if len(missing) > 14 else "")
        )


    # --- 6. employer-attribution integrity (added 2026-09-06) ---
    b_emp = _employer_blocks(base_tex)
    t_emp = _employer_blocks(tailored_tex)
    for key, clauses in _PROTECTED_CLAUSES.items():
        base_block = b_emp.get(key, "")
        tail_block = t_emp.get(key, "")
        if not base_block or not tail_block:
            continue
        bl, tl = base_block.lower(), tail_block.lower()
        for clause in clauses:
            c = clause.lower()
            if c in bl and c not in tl:
                warnings.append(
                    "ATTRIBUTION: protected clause missing from the " + key.upper() +
                    " block -- \"" + clause + "\". It is in the base and did not survive "
                    "tailoring. This bullet has been silently rewritten on six consecutive "
                    "packages; restore the base wording by hand before compiling."
                )
    for key, banned in _BANNED_ADDITIONS.items():
        base_block = b_emp.get(key, "").lower()
        tail_block = t_emp.get(key, "").lower()
        if not tail_block:
            continue
        for word in banned:
            if word in tail_block and word not in base_block:
                warnings.append(
                    "ATTRIBUTION: \"" + word + "\" was ADDED to the " + key.upper() +
                    " block and is not in the base. The profile lists it under "
                    "banned_additions for this employer: personal-project framing that must "
                    "not be attributed to the employer. Remove it."
                )

    return warnings


def report(tailored_tex: str, base_tex_path: str, label: str = "", jd_text: str = "") -> int:
    """Print the omission report. Returns the number of warnings (0 == clean)."""
    w = check_omissions(tailored_tex, base_tex_path, label, jd_text)
    tag = f" ({label})" if label else ""
    if not w:
        print(f"   Omission gate:   clean{tag}")
        return 0
    print(f"   Omission gate:   {len(w)} note(s){tag}")
    for line in w:
        print(f"      - {line}")
    return len(w)
