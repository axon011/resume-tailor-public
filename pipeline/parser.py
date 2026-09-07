"""Parse LaTeX resume into structured Python dict."""

import re
from collections import OrderedDict


def parse_resume(tex_path: str) -> dict:
    """Parse the .tex resume file into a structured dict."""
    with open(tex_path, "r", encoding="utf-8") as f:
        content = f.read()

    return {
        "header": _parse_header(content),
        "summary": _parse_summary(content),
        "experience": _parse_experience(content),
        "projects": _parse_projects(content),
        "skills": _parse_skills(content),
        "education": _parse_education(content),
    }


def _parse_header(content: str) -> dict:
    """Extract header info (name, tagline, contact, links)."""
    header_match = re.search(
        r"\\begin\{center\}(.*?)\\end\{center\}", content, re.DOTALL
    )
    if not header_match:
        raise ValueError("Could not parse header")

    block = header_match.group(1)

    name_match = re.search(r"\\LARGE\\bfseries\\color\{accent\}\s*(.+?)\}", block)
    name = name_match.group(1).strip() if name_match else ""

    tagline_match = re.search(r"\\color\{graytext\}\s*(.+?)\}", block)
    tagline = tagline_match.group(1).strip() if tagline_match else ""
    # Clean LaTeX formatting
    tagline = tagline.replace(r"\,|\,", "|").replace("\\&", "&").strip()

    return {"name": name, "tagline": tagline}


def _parse_summary(content: str) -> str:
    """Extract professional summary paragraph."""
    match = re.search(
        r"\\section\{Professional Summary\}\s*\n(.+?)(?=\n\n|\\section)",
        content,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Could not parse summary")
    return match.group(1).strip()


def _parse_experience(content: str) -> list[dict]:
    """Extract experience entries with their bullets."""
    exp_section = _extract_section(content, "Professional Experience")
    if not exp_section:
        raise ValueError("Could not parse experience section")

    entries = []
    # Find all \jobtitle{...}{...}{...}{...} followed by \begin{itemize}...\end{itemize}
    pattern = re.compile(
        r"\\jobtitle\{(.+?)\}\{(.+?)\}\{(.+?)\}\{(.+?)\}\s*"
        r"\\begin\{itemize\}(.*?)\\end\{itemize\}",
        re.DOTALL,
    )
    for m in pattern.finditer(exp_section):
        bullets = _extract_bullets(m.group(5))
        entries.append({
            "title": m.group(1).strip(),
            "company": m.group(2).strip(),
            "location": m.group(3).strip(),
            "dates": m.group(4).strip(),
            "bullets": bullets,
        })

    return entries


def _parse_projects(content: str) -> list[dict]:
    """Extract project entries with their bullets."""
    proj_section = _extract_section(content, "Projects")
    if not proj_section:
        raise ValueError("Could not parse projects section")

    entries = []
    # 2026-08-28: the old pattern was
    #     r"\\projecttitle\{(.+?)\}\{(.+?)\}\s*\\begin\{itemize\}(.*?)\\end\{itemize\}"
    # Non-greedy \{(.+?)\} stops at the FIRST closing brace, so any nested braces inside
    # an argument silently truncate it. The maintained Master CV puts a clickable link in
    # the project TITLE — "Live App ... \href{https://app...}{app...}" — and the
    # generator emitted "\href{https://tutor...}{}" with the second argument and the whole
    # tech-stack argument lost, producing LaTeX that died with "File ended while scanning
    # use of \projecttitle" and no PDF at all (Cohere #577). Brace-balanced scanning
    # instead, so nested \href / \textbf inside either argument survive.
    pattern = re.compile(
        r"\\projecttitle\s*(?=\{)",
    )
    def _braced(s, i):
        """Read one brace-balanced {...} starting at s[i] == '{'. Returns (inner, next_i)."""
        assert s[i] == "{"
        depth, j = 0, i
        while j < len(s):
            if s[j] == "{" and (j == 0 or s[j - 1] != chr(92)):
                depth += 1
            elif s[j] == "}" and s[j - 1] != chr(92):
                depth -= 1
                if depth == 0:
                    return s[i + 1:j], j + 1
            j += 1
        return None, len(s)

    _matches = []
    for _m in pattern.finditer(proj_section):
        _i = _m.end()
        _title, _i = _braced(proj_section, _i)
        if _title is None:
            continue
        while _i < len(proj_section) and proj_section[_i].isspace():
            _i += 1
        if _i >= len(proj_section) or proj_section[_i] != "{":
            continue
        _stack, _i = _braced(proj_section, _i)
        _b0 = proj_section.find(chr(92) + "begin{itemize}", _i)
        _b1 = proj_section.find(chr(92) + "end{itemize}", _b0 + 1) if _b0 >= 0 else -1
        if _b0 < 0 or _b1 < 0:
            continue
        _body = proj_section[_b0 + len(chr(92) + "begin{itemize}"):_b1]
        _matches.append((_title, _stack or "", _body))

    for m in _matches:
        bullets = _extract_bullets(m[2])
        entries.append({
            "title": m[0].strip(),
            "tech_stack": m[1].strip(),
            "bullets": bullets,
        })

    return entries


def _parse_skills(content: str) -> OrderedDict:
    """Extract skill categories as an ordered dict."""
    skills_section = _extract_section(content, "Skills")
    if not skills_section:
        raise ValueError("Could not parse skills section")

    skills = OrderedDict()
    pattern = re.compile(r"\\skillcat\{(.+?)\}\{(.+?)\}\s*$", re.MULTILINE)
    for m in pattern.finditer(skills_section):
        category = m.group(1).strip()
        items = m.group(2).strip()
        skills[category] = items

    return skills


def _parse_education(content: str) -> list[dict]:
    """Extract education entries."""
    edu_section = _extract_section(content, "Education")
    if not edu_section:
        raise ValueError("Could not parse education section")

    entries = []
    # Match \edutitle with optional itemize block
    pattern = re.compile(
        r"\\edutitle\{(.+?)\}\{(.+?)\}\{(.+?)\}\{(.+?)\}"
        r"(?:\s*\\begin\{itemize\}(.*?)\\end\{itemize\})?",
        re.DOTALL,
    )
    for m in pattern.finditer(edu_section):
        bullets = []
        if m.group(5):
            bullets = _extract_bullets(m.group(5))
        entries.append({
            "title": m.group(1).strip(),
            "institution": m.group(2).strip(),
            "location": m.group(3).strip(),
            "dates": m.group(4).strip(),
            "bullets": bullets,
        })

    return entries


def _extract_bullets(itemize_body: str) -> list[str]:
    """Extract all \\item entries from an itemize body."""
    # Split on \item, skip the first empty part before the first \item
    parts = re.split(r"\\item\s+", itemize_body)
    bullets = []
    for part in parts[1:]:  # skip first empty split
        text = part.strip()
        if text:
            bullets.append(text)
    return bullets


_SECTION_RX = re.compile(r"\\section\{([^}]*)\}(.*?)(?=\\section\{|\\end\{document\})", re.DOTALL)


def _extract_section(content: str, section_name: str) -> str | None:
    """Content between a \\section{...} heading and the next \\section or \\end{document}.

    2026-08-28: the heading is matched LOOSELY — `section_name` only has to appear inside
    the heading, case-insensitively. Exact matching broke the entire pipeline the moment
    RESUME_TEX_PATH was pointed at the maintained Master CV, which titles the section
    "Technical Skills" rather than "Skills": parse_resume raised "Could not parse skills
    section" and no package could be generated at all. A section heading is a human-facing
    label that will keep drifting, and the parser should not be what breaks when it does.
    """
    target = (section_name or "").strip().lower()
    for m in _SECTION_RX.finditer(content):
        if target in m.group(1).strip().lower():
            return m.group(2)
    return None
