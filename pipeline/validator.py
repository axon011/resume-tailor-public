"""Validate tailored resume has no fabricated content."""

import re


def validate(original: dict, tailored: dict) -> list[str]:
    """Check that tailored resume doesn't introduce fabricated content.

    Returns a list of warnings. Empty list means validation passed.
    """
    warnings = []

    # 1. Check skills — new keywords are now ALLOWED for ATS boost
    # Only flag new skill categories (not new items within existing categories)
    for cat, items in tailored["skills"].items():
        if cat not in original["skills"]:
            warnings.append(f"New skill category added: {cat}")

    # 2. Check project count is reasonable (titles can be rephrased)
    if len(tailored["projects"]) > len(original["projects"]):
        warnings.append(f"More projects than original: {len(original['projects'])} -> {len(tailored['projects'])}")
    elif len(tailored["projects"]) < 2:
        warnings.append(f"Too few projects: {len(tailored['projects'])} (minimum 2)")

    # 3. Check experience entry count matches (bullets can be rephrased)
    if len(tailored["experience"]) != len(original["experience"]):
        warnings.append(f"Experience count changed: {len(original['experience'])} -> {len(tailored['experience'])}")
    for i, exp in enumerate(tailored["experience"]):
        if i >= len(original["experience"]):
            warnings.append(f"New experience entry at index {i}")
            continue
        if len(exp["bullets"]) != len(original["experience"][i]["bullets"]):
            warnings.append(f"Experience[{i}] bullet count changed: {len(original['experience'][i]['bullets'])} -> {len(exp['bullets'])}")

    # 4a. Summary bounded-adaptation check (revised 2026-07-19; supersedes the Apr 28
    # verbatim-lock). The prompt now allows reweighting/trimming the summary's TRUE
    # specialization clauses per JD. What must still hold:
    #   - no NEW technical claims (enforced by 4b below: every technical term in the
    #     tailored summary must already exist somewhere in the base resume), and
    #   - the summary stays compact (<= ~3 rendered lines).
    tailored_summary = tailored["summary"].strip()
    if len(tailored_summary) > 480:
        warnings.append(
            f"SUMMARY too long ({len(tailored_summary)} chars, cap ~480 / 3 rendered "
            f"lines). Bounded adaptation allows reweighting, not expansion."
        )

    # 4b. Check summary doesn't introduce new technical terms
    # We check against ALL resume text (not just original summary)
    # because the humanized summary may reference tech from skills/experience
    original_full_text = _get_full_text(original).lower()
    # Also normalize LaTeX escapes for matching
    original_full_normalized = original_full_text.replace("\\&", "&").replace("\\%", "%")
    summary_terms = _extract_technical_terms(tailored["summary"])
    for term in summary_terms:
        term_lower = term.lower()
        if term_lower not in original_full_text and term_lower not in original_full_normalized:
            # For multi-word terms, check if all individual words exist
            words = term_lower.split()
            if len(words) > 1 and all(_word_in_text(w, original_full_normalized) for w in words):
                continue  # compound of existing terms, not fabrication
            # Single word: check with stem matching (e.g. "develops" matches "develop")
            if len(words) == 1 and _word_in_text(term_lower, original_full_normalized):
                continue
            warnings.append(f"New term in summary: '{term}'")

    # 5. Check LaTeX escaping in metrics (critical ATS issue)
    all_bullets = []
    for exp in tailored["experience"]:
        all_bullets.extend(exp.get("bullets", []))
    for proj in tailored["projects"]:
        all_bullets.extend(proj.get("bullets", []))

    for bullet in all_bullets:
        # Check for unescaped percent signs (breaks LaTeX and corrupts PDF metrics)
        if re.search(r'\d+%(?!\\)', bullet) and '\\%' not in bullet:
            warnings.append(f"Unescaped percent in bullet: '{bullet[:60]}...'")
        # Check for unescaped ampersands
        if '&' in bullet and '\\&' not in bullet:
            if not re.search(r'\\&', bullet):
                warnings.append(f"Unescaped ampersand in bullet: '{bullet[:60]}...'")

    # 6. Check tagline format
    tagline = tailored.get("tagline", "")
    if tagline and '|' in tagline and r'\,|\,' not in tagline:
        warnings.append(f"Tagline has plain separators (should use \\,|\\,): '{tagline[:50]}'")

    # 7. Check skill count (max 5 new per category)
    for cat, items in tailored["skills"].items():
        if cat in original["skills"]:
            orig_count = len(_split_skills(original["skills"][cat]))
            new_count = len(_split_skills(items))
            added = new_count - orig_count
            if added > 5:
                warnings.append(f"Too many new skills in '{cat}': {added} added (max 5)")

    # 8. Check for duplicate skill categories (same content under different names)
    skill_values = list(tailored["skills"].items())
    for i in range(len(skill_values)):
        for j in range(i + 1, len(skill_values)):
            cat_a, items_a = skill_values[i]
            cat_b, items_b = skill_values[j]
            keywords_a = set(w.strip().lower().rstrip(",") for w in items_a.split(","))
            keywords_b = set(w.strip().lower().rstrip(",") for w in items_b.split(","))
            overlap = keywords_a & keywords_b
            if len(overlap) >= 3:
                warnings.append(f"DUPLICATE SKILLS: '{cat_a}' and '{cat_b}' share {len(overlap)} items: {', '.join(list(overlap)[:4])}")

    # 9. Check for experience-project bullet duplication
    exp_bullets = []
    for exp in tailored["experience"]:
        exp_bullets.extend(exp.get("bullets", []))
    proj_bullets = []
    for proj in tailored["projects"]:
        proj_bullets.extend(proj.get("bullets", []))

    for eb in exp_bullets:
        eb_words = set(eb.lower().split())
        for pb in proj_bullets:
            pb_words = set(pb.lower().split())
            if len(eb_words) > 5 and len(pb_words) > 5:
                common = eb_words & pb_words
                similarity = len(common) / min(len(eb_words), len(pb_words))
                if similarity > 0.7:
                    warnings.append(f"DUPLICATE BULLET: experience and project bullets are >70% similar: '{eb[:50]}...'")
                    break

    return warnings


def _word_in_text(word: str, text: str) -> bool:
    """Check if a word exists in text, with basic stem matching."""
    if word in text:
        return True
    # Try without trailing 's' (plural)
    if word.endswith("s") and word[:-1] in text:
        return True
    # Try stem: match if the first 5+ chars of the word appear as a word prefix in text
    stem = word[:min(len(word), 5)]
    if len(stem) >= 4 and re.search(rf"\b{re.escape(stem)}", text):
        return True
    return False


def _split_skills(skills_str: str) -> list[str]:
    """Split a skills string on commas, handling parenthetical groups."""
    # Simple split - items like "Langfuse (Tracing, Evals)" should stay together
    result = []
    depth = 0
    current = []
    for char in skills_str:
        if char == "(":
            depth += 1
            current.append(char)
        elif char == ")":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            result.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        result.append("".join(current).strip())
    return result


def _extract_technical_terms(text: str) -> list[str]:
    """Extract likely technical terms from text (capitalized words, acronyms, known patterns)."""
    # Find capitalized words, acronyms, and tech-looking terms
    terms = re.findall(r"\b[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*\b", text)
    # Also find acronyms
    terms += re.findall(r"\b[A-Z]{2,}\b", text)
    # Filter out common English words
    common = {"The", "This", "That", "With", "From", "Into", "Using", "And", "For",
              "Ships", "End", "Through", "Including", "Combined", "Hands", "Production",
              "Applications", "Prototype", "Deployment", "Maturity", "Capability",
              "Engineer", "Experience", "Building", "Integrated", "Products", "Full",
              "Stack", "Cloud", "Native", "Differentiates", "Evaluation", "Tracking",
              "Tracing", "AI", "M", "Who", "Builds", "Currently", "Running", "Works",
              "Across", "Pipeline", "Backends", "Frontends", "Deployments", "Keeps",
              "Honest", "Rather", "Than", "Guessing", "Quality", "Comfortable",
              "Sc", "Currently", "Runs", "Also", "Both", "Well", "Can", "Has",
              "Does", "Ensures", "Experienced", "Focused", "Based", "Skilled",
              "Develops", "Manages", "Connects", "Serves", "While", "Like",
              "Robust", "Stable", "Operations", "Currently", "These",}
    return [t for t in terms if t not in common and len(t) > 1]


def _get_full_text(resume_data: dict) -> str:
    """Get all text content from resume data for term checking."""
    parts = [resume_data["summary"]]
    for exp in resume_data["experience"]:
        parts.append(exp["title"])
        parts.extend(exp["bullets"])
    for proj in resume_data["projects"]:
        parts.append(proj["title"])
        parts.append(proj["tech_stack"])
        parts.extend(proj["bullets"])
    for items in resume_data["skills"].values():
        parts.append(items)
    return " ".join(parts)
