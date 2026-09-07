"""Core tailoring engine: LLM calls to analyze JD and reorder/rewrite resume."""

import json
import copy
import re
from collections import OrderedDict

from openai import OpenAI

import config
from . import prompts
from . import profile as _profile

_P = _profile.load()


# Fix #3 (Apr 28): Resume header location is now LOCKED to the candidate's
# actual location (config.CANDIDATE_LOCATION), with a stable openness suffix.
#
# Why: prior behavior derived the location from the JD ("Berlin, Germany
# (relocating)" for a Berlin role, "Munich, Germany (relocating)" for Munich,
# etc). This created a per-app inconsistency — LinkedIn says one city, the
# resume says a different city per submission. Recruiters who cross-checked
# saw a credibility gap. Honest answer is "I'm in X, open to relocating",
# said the same way every time.
#
# DE_CITIES + JD scanning is kept below for cover-letter usage / future
# location-aware features, but the resume-header value is now stable.

DE_CITIES = [
    "Berlin", "Munich", "München", "Hamburg", "Frankfurt", "Cologne", "Köln",
    "Stuttgart", "Düsseldorf", "Dusseldorf", "Leipzig", "Dortmund", "Essen",
    "Bremen", "Dresden", "Hanover", "Hannover", "Nuremberg", "Nürnberg",
    "Duisburg", "Bochum", "Wuppertal", "Bielefeld", "Bonn", "Karlsruhe",
    "Heidelberg", "Freiburg", "Heilbronn", "Cottbus", "Potsdam", "Augsburg",
    "Mannheim", "Wiesbaden", "Metzingen", "Paderborn", "Oranienburg",
]


# ── Geography-aware header (tracks with `geo_header: true`) ─────────────────────
# For a part-time or on-site role, geography is a screening question: a 20 h/week job
# cannot be worked from 600 km away. On a track the profile marks `geo_header`, the
# header answers that question instead of ignoring it.
#
# The profile's `home_cities` maps a job city (lowercase) to the REAL address that may
# be printed for it — the candidate's own addresses and the commuter belt around them.
# Anything else is stated as an explicit relocation, never as an address the candidate
# does not have.
_HOME_CITIES = _P.home_cities

_JOB_CITY_RE = re.compile(
    r"\b(Berlin|Potsdam|Cottbus|Senftenberg|M[uü]nchen|Munich|Hamburg|K[oö]ln|Cologne|"
    r"Frankfurt am Main|Frankfurt|Stuttgart|D[uü]sseldorf|Dortmund|Essen|Leipzig|Dresden|"
    r"Hannover|N[uü]rnberg|Nuremberg|Bremen|Karlsruhe|Mannheim|Augsburg|Aachen|"
    r"Braunschweig|M[uü]nster|Bonn|Bielefeld|Darmstadt|Ulm|W[uü]rzburg|Heidelberg|Kiel|"
    r"Regensburg|Ingolstadt|W[uü]rselen|Herzogenrath|Paderborn|H[uü]rth|Weinheim)\b")


def _job_city(jd_text: str):
    """First German city named in the JD, or None. Order of appearance wins."""
    m = _JOB_CITY_RE.search(jd_text or "")
    return m.group(1) if m else None


def _ws_location(jd_text: str, override: str = None) -> str:
    """Header location for a geo_header track.

    Never invents an address. If the job is in or near a home city, that real address is
    printed plainly. Otherwise the real address is printed alongside an explicit
    relocation offer naming the city, which is what a part-time screener is filtering on.
    """
    if override:
        loc = override.strip()
        return loc if "(" in loc or "relocate" in loc.lower() else f"{loc} (open to relocate)"
    city = _job_city(jd_text)
    if not city:
        return f"{config.CANDIDATE_LOCATION} (open to relocate)"
    home = _HOME_CITIES.get(city.lower())
    if home:
        return home
    return f"{config.CANDIDATE_LOCATION} \\,|\\, open to relocate to {city}"


def _extract_job_location(jd_text: str, override: str = None, track: str = None) -> str:
    """Return the resume-header location string.

    Defaults to "{CANDIDATE_LOCATION} (open to relocate)" — one stable string,
    which keeps the header consistent with LinkedIn and avoids a per-app
    credibility gap. The JD city is NEVER used: a Munich posting must not make
    him claim to live in Munich.

    `override` exists for a candidate with more than one real address, where which
    one to state depends on where the job is. Pass it explicitly via
    `main.py --location`; there is no automatic inference.
    """
    if track and _P.tracks.get(track, {}).get("geo_header"):
        return _ws_location(jd_text, override)
    if override:
        loc = override.strip()
        return loc if "(" in loc else f"{loc} (open to relocate)"
    return f"{config.CANDIDATE_LOCATION} (open to relocate)"


def get_llm_client():
    """Create the LLM client. Provider chosen via LLM_PROVIDER env var.

    Supported providers:
      - "claude-code"  → shell out to the local `claude` CLI, using the user's
                         Claude Max OAuth session. No HTTP API key needed.
                         Default model: haiku (override via LLM_MODEL).
      - "gemini"/"agy" → shell out to Antigravity's `agy` CLI (Gemini, free
                         Google CloudCode quota). Conserves Claude tokens.
                         High-stakes work, so opt-in only. Model via AGY_MODEL.
      - default        → OpenAI-compatible HTTP client (GLM via z.ai, OpenAI,
                         OpenRouter, anything with /v1/chat/completions).

    Both return objects exposing `.chat.completions.create(...)` so callers
    don't need to know which backend they're talking to.
    """
    import os
    provider = os.getenv("LLM_PROVIDER", "").lower()
    if provider == "claude-code":
        from .llm_client import ClaudeCodeClient
        model = os.getenv("LLM_MODEL", "haiku")
        return ClaudeCodeClient(model=model)
    if provider in ("gemini", "agy"):
        from .agy_client import AgyClient
        return AgyClient(model=os.getenv("AGY_MODEL") or None)
    return OpenAI(
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_BASE_URL,
    )


_TRAILING_COMMA = re.compile(r",(\s*[}\]])")


def _strip_trailing_commas(s: str) -> str:
    """Drop a comma that directly precedes "}" or "]" — legal in JavaScript, fatal in JSON.

    Runs over the whole text, strings included, so a literal ", }" inside a bullet would
    lose its comma too; no field carries JSON-shaped prose, and a cosmetic comma is cheap
    next to a whole generation thrown away.
    """
    return _TRAILING_COMMA.sub(r"\1", s)


def tailor_resume(resume_data: dict, job_info: dict, github_context: str, github_repos: list[dict] = None, instructions: str = None, track: str = None, location: str = None) -> dict:
    """Tailor resume data to match a job description.

    Returns new resume_data dict with restructured sections.
    """
    client = get_llm_client()

    resume_for_prompt = {
        "summary": resume_data["summary"],
        "experience": resume_data["experience"],
        "projects": resume_data["projects"],
        "skills": dict(resume_data["skills"]),
    }

    # Format GitHub repos as available project pool
    github_projects_str = ""
    if github_repos:
        repo_lines = []
        for repo in github_repos:
            topics = ", ".join(repo.get("topics", []))
            line = f"- {repo['name']} ({repo['language']}): {repo['description']}"
            if topics:
                line += f" [topics: {topics}]"
            repo_lines.append(line)
        github_projects_str = "\n".join(repo_lines)

    prompt = prompts.TAILORING_PROMPT.format(
        job_title=job_info["title"],
        job_description=job_info["description"][:4000],
        github_context=github_context[:1500],
        github_projects=github_projects_str,
        resume_json=json.dumps(resume_for_prompt, indent=2),
    )

    if instructions:
        prompt += f"\n\n## Tailoring Instructions (from evaluator)\n{instructions[:1500]}"

    system_prompt = prompts.SYSTEM_PROMPT
    ctx = config.candidate_context(track or config.DEFAULT_TRACK)
    if ctx:
        system_prompt += f"\n\nCANDIDATE CONTEXT (use to make tailored output specific and credible, not generic):\n{ctx}"

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_completion_tokens=8192,
    )

    raw = response.choices[0].message.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[:-3]
        raw = raw.strip()

    # A model sometimes prefixes the JSON with a sentence of prose and only THEN
    # opens the fence, so the startswith(fence) strip above never fires and
    # raw_decode dies at char 0. Slice to the first fenced block, else to the first
    # brace. 2026-09-02: this failed 5/5 attempts on the ActAI run, where the model
    # opened with "That tool search wasn't needed for this task".
    if not raw.startswith('{'):
        _m = re.search(r'```(?:json)?\s*\n(.*?)(?:\n```|\Z)', raw, re.S)
        if _m:
            raw = _m.group(1).strip()
        else:
            _b = raw.find('{')
            if _b > 0:
                raw = raw[_b:]

    # Fix LLM JSON issues: LaTeX escapes (\&, \%, \,) break JSON parsing.
    # Also handles GLM-4.7 "Extra data" trailing commentary via raw_decode
    # (stops at first valid JSON value, ignores trailing text).
    try:
        tailoring, _ = json.JSONDecoder().raw_decode(raw)
    except json.JSONDecodeError:
        # Iteratively fix invalid escapes until JSON parses.
        # 2026-09-06: a trailing comma before "}" ('"...",\n    },') failed 5/5 attempts on
        # one run with an otherwise perfect response — every attempt below
        # repairs a BACKSLASH, so a comma defect could never converge. Strip those first.
        fixed = _strip_trailing_commas(raw)
        max_attempts = 5
        tailoring = None
        for attempt in range(max_attempts):
            try:
                tailoring, _ = json.JSONDecoder().raw_decode(fixed)
                break
            except json.JSONDecodeError as e:
                # Find the problematic position and fix the escape there
                pos = e.pos
                if pos and pos < len(fixed) and fixed[pos - 1] == '\\':
                    # Double-escape this specific backslash
                    fixed = fixed[:pos - 1] + '\\\\' + fixed[pos:]
                else:
                    # Fallback: replace ALL non-JSON backslash escapes
                    fixed = re.sub(r'\\(?!["\\/bfnrtu\\])', r'\\\\', fixed)
                    try:
                        tailoring, _ = json.JSONDecoder().raw_decode(fixed)
                    except json.JSONDecodeError:
                        # Global fix didn't help; let the loop try again or exit.
                        continue
                    break
        if tailoring is None:
            # Dump the raw response. A parse failure at char 0 is almost always
            # prose (a refusal), not malformed JSON — and the message alone
            # cannot tell you which. See the 2026-08-15 diagnostic order.
            try:
                import pathlib
                _dbg = pathlib.Path(__file__).resolve().parent.parent / "last_llm_raw.txt"
                _dbg.write_text(raw, encoding="utf-8")
                print(f"   [tailoring] raw LLM response dumped -> {_dbg}")
                print(f"   [tailoring] first 400 chars: {raw[:400]!r}")
            except Exception:
                pass
            raise ValueError(f"Could not parse LLM JSON after {max_attempts} attempts")

    # Print changes for transparency
    if "tagline" in tailoring and tailoring["tagline"]:
        print(f"   Tagline: {tailoring['tagline']}")
    if "added_keywords" in tailoring and tailoring["added_keywords"]:
        # Normalize: LLM sometimes returns list[str], sometimes list[dict]
        kws = []
        for k in tailoring["added_keywords"]:
            if isinstance(k, str):
                kws.append(k)
            elif isinstance(k, dict):
                kws.append(str(k.get("keyword") or k.get("term") or k))
        if kws:
            print(f"   Added keywords: {', '.join(kws)}")
    if "project_changes" in tailoring and tailoring["project_changes"]:
        print(f"   Projects: {tailoring['project_changes']}")

    result = _apply_tailoring(resume_data, tailoring)

    # Header location: stable by default, explicit --location to override.
    _loc = _extract_job_location(job_info.get("description", ""), override=location, track=track)
    result["location"] = _loc
    print(f"   Location: {_loc}" + ("   [--location override]" if location else ""))

    return result


def _apply_tailoring(resume_data: dict, tailoring: dict) -> dict:
    """Apply the LLM's tailoring instructions to the resume data."""
    result = copy.deepcopy(resume_data)

    # 0. Apply tagline (LLM returns plain text, we convert to LaTeX)
    if "tagline" in tailoring and tailoring["tagline"]:
        tagline = tailoring["tagline"]
        # Convert plain text to LaTeX format
        # Replace & with \& for LaTeX
        tagline = tagline.replace("&", r"\&")
        # Replace | separators with \,|\, for LaTeX spacing
        if r"\,|\," not in tagline and "|" in tagline:
            parts = [p.strip() for p in tagline.split("|")]
            tagline = r" \,|\, ".join(parts)
        result["tagline"] = tagline
    else:
        result["tagline"] = r"AI Engineer \,|\, Agentic Systems \& Production RAG \,|\, Full-Stack GenAI"

    # 1. Apply rephrased summary
    if "summary" in tailoring:
        result["summary"] = tailoring["summary"]

    # 2. Reorder + enhance skills
    if "skill_order" in tailoring:
        orig_keys = {}
        for key in resume_data["skills"]:
            normalized = key.replace("\\&", "&").replace("\\", "").strip()
            orig_keys[normalized] = key
            orig_keys[key] = key

        new_skills = OrderedDict()
        used_orig_keys = set()
        for entry in tailoring["skill_order"]:
            cat_name = entry["category"]
            orig_key = orig_keys.get(cat_name)
            if orig_key:
                new_skills[orig_key] = entry["items"]
                used_orig_keys.add(orig_key)
            else:
                new_skills[cat_name] = entry["items"]
        # Only add back original categories that weren't renamed/replaced
        # Skip if the LLM already covered this content under a new name
        new_skills_lower = " ".join(str(v) for v in new_skills.values()).lower()
        for cat_name in resume_data["skills"]:
            if cat_name in used_orig_keys:
                continue
            # Check if this category's content is already covered by a new category
            orig_items = resume_data["skills"][cat_name].lower()
            top_keywords = [w.strip().rstrip(",") for w in orig_items.split(",")[:3]]
            already_covered = sum(1 for kw in top_keywords if kw.lower() in new_skills_lower)
            if already_covered >= 2:
                continue
            new_skills[cat_name] = resume_data["skills"][cat_name]
        result["skills"] = new_skills

    # 3. Apply restructured projects (full objects from LLM)
    if "projects" in tailoring and isinstance(tailoring["projects"], list):
        # The prompt asks for tech_stack back, but the LLM routinely omits it. The old
        # .get("tech_stack", "") then rendered \projecttitle{Name}{} — the PDF silently
        # lost every project's tech tags (Qdrant, LangChain, RAGAs, MLflow, BM25, networkx,
        # PyTorch, PEFT, CrewAI...), which no gate checks and which costs real ATS keywords.
        # It hit three packages on 2026-08-29 alone. The stack is factual data straight from
        # the base resume, so fall back to it by title instead of trusting the LLM to echo it.
        def _norm_title(t):
            return re.sub(r"[^a-z0-9]+", "", re.sub(r"\[a-zA-Z]+|---", " ", str(t)).lower())

        base_stacks = {}
        for orig in resume_data.get("projects", []):
            key = _norm_title(orig.get("title", ""))
            if key and orig.get("tech_stack", "").strip():
                base_stacks[key] = orig["tech_stack"].strip()

        new_projects = []
        backfilled, unmatched = [], []
        for proj in tailoring["projects"]:
            if isinstance(proj, dict) and "title" in proj and "bullets" in proj:
                stack = str(proj.get("tech_stack", "") or "").strip()
                if not stack:
                    stack = base_stacks.get(_norm_title(proj["title"]), "")
                    if stack:
                        backfilled.append(proj["title"])
                    else:
                        unmatched.append(proj["title"])
                new_projects.append({
                    "title": proj["title"],
                    "tech_stack": stack,
                    "bullets": proj["bullets"],
                })
        if backfilled:
            print(f"   Tech stacks restored from base for {len(backfilled)} project(s): "
                  f"{', '.join(t[:34] for t in backfilled)}")
        if unmatched:
            print(f"   WARNING: no tech stack for {len(unmatched)} project(s) and no base "
                  f"match by title: {', '.join(t[:34] for t in unmatched)}")
        if len(new_projects) >= 2:
            result["projects"] = new_projects
            dropped = len(resume_data["projects"]) - len(new_projects)
            if dropped > 0:
                print(f"   Dropped {dropped} irrelevant project(s)")
    # Fallback: old index-based reordering
    elif "project_order" in tailoring:
        original_projects = resume_data["projects"]
        new_projects = []
        for idx in tailoring["project_order"]:
            if 0 <= idx < len(original_projects):
                new_projects.append(original_projects[idx])
        for proj in original_projects:
            if proj not in new_projects:
                new_projects.append(proj)
        result["projects"] = new_projects

    # 4. Apply restructured experience (full objects from LLM)
    if "experience" in tailoring and isinstance(tailoring["experience"], list):
        new_experience = []
        for i, exp in enumerate(tailoring["experience"]):
            if isinstance(exp, dict) and "bullets" in exp:
                # Use original metadata, take rephrased bullets from LLM
                orig = resume_data["experience"][i] if i < len(resume_data["experience"]) else {}
                new_experience.append({
                    "title": exp.get("title", orig.get("title", "")),
                    "company": exp.get("company", orig.get("company", "")),
                    "location": exp.get("location", orig.get("location", "")),
                    "dates": exp.get("dates", orig.get("dates", "")),
                    "bullets": exp["bullets"],
                })
        if new_experience:
            result["experience"] = new_experience
    # Fallback: old index-based reordering
    elif "experience_bullet_orders" in tailoring:
        for exp_idx_str, bullet_order in tailoring["experience_bullet_orders"].items():
            exp_idx = int(exp_idx_str)
            if 0 <= exp_idx < len(result["experience"]):
                original_bullets = resume_data["experience"][exp_idx]["bullets"]
                new_bullets = []
                for b_idx in bullet_order:
                    if 0 <= b_idx < len(original_bullets):
                        new_bullets.append(original_bullets[b_idx])
                for b in original_bullets:
                    if b not in new_bullets:
                        new_bullets.append(b)
                result["experience"][exp_idx]["bullets"] = new_bullets

    return result
