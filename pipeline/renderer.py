"""Render tailored resume data into LaTeX and compile to PDF."""

import os
import re
import subprocess
import tempfile
import shutil

import jinja2

import config
from . import profile as _profile

_P = _profile.load()


def get_jinja_env() -> jinja2.Environment:
    """Create Jinja2 environment with custom delimiters for LaTeX compatibility."""
    # Templates live at the project root, one level up from the pipeline package.
    template_dir = os.path.join(os.path.dirname(__file__), "..", "templates")
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir),
        block_start_string="<%",
        block_end_string="%>",
        variable_start_string="<<",
        variable_end_string=">>",
        comment_start_string="<#",
        comment_end_string="#>",
    )


def render_latex(resume_data: dict) -> str:
    """Render resume data dict into a LaTeX string."""
    env = get_jinja_env()
    template = env.get_template("resume.tex.j2")
    # The LLM and the base-resume parser routinely emit LaTeX specials in any
    # free-text field — most often '&' (e.g. "MLOps & Monitoring" skill category
    # or "Multi-Agent Research & Report Pipeline" project title). A raw '&' is a
    # LaTeX alignment tab and silently corrupts the PDF: nonstopmode still writes
    # a partial file, so the pipeline reports success while sections are mangled.
    # Deep-escape &/%/# across every string leaf, skipping anything already
    # escaped. (Caught 2026-05-27: first a skill category, then a project title.)
    return template.render(**_profile_context(), **_escape_tree(resume_data))


# snake_case only, and nothing that would ever need escaping. A printed label that
# happens to be lowercase ("ml_ops & tools") still carries a LaTeX special, so it fails
# this test and gets escaped like the content it is.
_STRUCTURAL_KEY = re.compile(r"^[a-z][a-z0-9_]*$")


def _is_structural_key(key) -> bool:
    """True for snake_case field names the templates address by name."""
    return isinstance(key, str) and bool(_STRUCTURAL_KEY.match(key))


def _escape_tree(value):
    """Recursively escape LaTeX specials in all string leaves; no double-escape.

    Escapes the full special set (parity with cover-letter _escape_latex). Before
    2026-06-13 this only handled & % #, so an underscore or '$' in a tailored
    resume bullet (e.g. a snake_case metric name like 'context_recall') would
    break LaTeX or silently vanish — and nonstopmode still emitted a PDF the
    pipeline reported as success.
    """
    if isinstance(value, str):
        # Backslash-escaped specials: & % # _ $
        for ch in ("&", "%", "#", "_", "$"):
            value = re.sub(r"(?<!\\)" + re.escape(ch), "\\\\" + ch, value)
        # ~ and ^ are active/superscript chars; backslash-escaping makes accents,
        # so map them to the text commands instead.
        value = re.sub(r"(?<!\\)~", r"\\textasciitilde{}", value)
        value = re.sub(r"(?<!\\)\^", r"\\textasciicircum{}", value)
        return value
    if isinstance(value, dict):
        # Keys are two different things here, and they need opposite treatment:
        #   - STRUCTURAL field names the templates look up by name (tech_stack, bullets,
        #     title...). Escaping these renamed "tech_stack" to "tech\_stack" once '_'
        #     joined the escape set (2026-06-13), so proj.tech_stack resolved to Undefined
        #     and every project rendered as \projecttitle{Name}{} — tech tags dropped from
        #     the PDF with no error and no gate to catch it. Three packages shipped
        #     degraded on 2026-08-29 before this was traced.
        #   - CONTENT keys that are printed, i.e. the skills-category labels
        #     ("Backend & Infra"), which still need escaping or a raw '&' corrupts the
        #     PDF (the 2026-05-27 fix).
        # snake_case identifiers are structural; anything else is a printed label.
        return {(k if _is_structural_key(k) else _escape_tree(k)): _escape_tree(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [_escape_tree(v) for v in value]
    return value


def render_cover_letter(cover_data: dict) -> str:
    """Render cover letter data into a LaTeX string."""
    env = get_jinja_env()
    template = env.get_template("cover_letter.tex.j2")
    # Escape LaTeX special chars in paragraphs
    escaped = dict(cover_data)
    escaped["paragraphs"] = [_escape_latex(p) for p in cover_data["paragraphs"]]
    escaped["greeting"] = _escape_latex(cover_data.get("greeting", "Dear Hiring Team,"))
    return template.render(**_profile_context(), **escaped)


def _profile_context() -> dict:
    """Header, links and publications for the templates, from the profile.

    Header TEXT is LaTeX-escaped (an underscore in an email must not break the build);
    URLs are not (an escaped underscore breaks a link). Publications are raw LaTeX by
    design — they carry \\underline, \\% and math already.
    """
    h = {k: _P.header_field(k) for k in
         ("name", "phone", "email", "linkedin", "github", "website", "location", "pdf_title", "languages")}
    # languages is raw LaTeX by design ("English (C1) \,|\, German (B1)")
    header = {k: (v if k == "languages" else _escape_latex(v)) for k, v in h.items()}
    header["pdf_title"] = h["pdf_title"] or f"{h['name']} - Resume"
    links = []
    for key in ("linkedin", "github", "website"):
        if h[key]:
            url = h[key] if h[key].startswith(("http://", "https://")) else "https://" + h[key]
            links.append((url, _escape_latex(h[key].replace("https://", "").replace("http://", ""))))
    return {"cand": header, "cand_links": links, "publications": _P.publications}


def _escape_latex(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    # Don't double-escape already escaped chars
    replacements = [
        ("\\", "\\textbackslash{}"),  # must be first
        ("&", "\\&"),
        ("%", "\\%"),
        ("$", "\\$"),
        ("#", "\\#"),
        ("_", "\\_"),
        ("{", "\\{"),
        ("}", "\\}"),
        ("~", "\\textasciitilde{}"),
        ("^", "\\textasciicircum{}"),
    ]
    for old, new in replacements:
        # Skip if already escaped
        if old == "\\":
            continue  # backslash handling is tricky, skip for cover letters
        text = text.replace(old, new)
    return text


def compile_pdf(tex_content: str, output_dir: str, filename: str) -> str:
    """Compile LaTeX content to PDF. Returns path to generated PDF."""
    os.makedirs(output_dir, exist_ok=True)

    tex_path = os.path.join(output_dir, f"{filename}.tex")
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")

    # Remove any STALE PDF from an earlier same-day run before compiling. The
    # filename is {prefix}_{company}_{YYYYMMDD}, so without this a failed
    # recompile would leave the previous PDF in place and os.path.exists() below
    # would report it as a fresh success (silent stale-PDF bug, 2026-06-13).
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tex_content)

    # Run pdflatex twice for references
    result = None
    for _ in range(2):
        result = subprocess.run(
            [config.PDFLATEX_PATH, "-interaction=nonstopmode", f"{filename}.tex"],
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=60,
        )

    # Judge success on the compiler, not just file presence: under nonstopmode
    # pdflatex writes a partial/garbled PDF on a fatal error and exits, so also
    # require that no fatal "! ..." LaTeX error appears in the log.
    log_text = (result.stdout or "") + "\n" + (result.stderr or "")
    fatal = re.search(r"(?m)^! ", log_text)
    if not os.path.exists(pdf_path) or fatal:
        detail = fatal.string[fatal.start():fatal.start() + 200] if fatal else ""
        raise RuntimeError(
            f"PDF compilation failed for {filename} "
            f"(fatal LaTeX error: {'yes' if fatal else 'no'}).\n"
            f"{detail}\nstdout: {(result.stdout or '')[-500:]}"
        )

    # A heading alone at the foot of a page (H.E.A.T. 2026-09-09: project title + tech
    # line on page 1, bullets on page 2) is a layout defect the template's needspace
    # should prevent; this check makes sure it never ships if it slips through.
    _check_orphaned_headings(pdf_path, tex_content, filename)

    # Clean aux files
    for ext in [".aux", ".log", ".out"]:
        aux_file = os.path.join(output_dir, f"{filename}{ext}")
        if os.path.exists(aux_file):
            os.remove(aux_file)

    return pdf_path


# ── orphaned-heading check ──────────────────────────────────────────────────

_HEADING_CMDS = re.compile(r"\\(?:section|jobtitle|projecttitle|edutitle)\s*\{")


def _tex_headings(tex: str) -> set:
    """Plain-text forms of every heading the document declares (first brace argument)."""
    out = set()
    for m in _HEADING_CMDS.finditer(tex):
        i, depth, j = m.end(), 1, m.end()
        while j < len(tex) and depth:
            depth += {"{": 1, "}": -1}.get(tex[j], 0)
            j += 1
        out.add(_plain(tex[i:j - 1]))
    return {h for h in out if len(h) > 3}


def _plain(s: str) -> str:
    s = re.sub(r"\\href\{[^}]*\}\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    s = re.sub(r"[{}$~|,]", " ", s)
    return re.sub(r"\s+", " ", s).strip().lower()


def _check_orphaned_headings(pdf_path: str, tex_content: str, filename: str) -> None:
    """Raise if any page except the last ends on a heading or a tech-stack line."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        try:
            from pypdf import PdfReader
            pages = [p.extract_text() or "" for p in PdfReader(pdf_path).pages]
        except ImportError:
            print("   [warn] orphan check skipped: install PyMuPDF or pypdf")
            return
    else:
        doc = fitz.open(pdf_path)
        pages = [p.get_text() for p in doc]
        doc.close()
    if len(pages) < 2:
        return
    headings = _tex_headings(tex_content)
    if not headings:
        return
    for idx, text in enumerate(pages[:-1], start=1):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            continue
        tail = [_plain(l) for l in lines[-2:]]
        hit = next((h for h in headings if any(t == h or (len(h) > 12 and t.startswith(h[:40])) for t in tail)), None)
        if hit:
            raise RuntimeError(
                f"Orphaned heading in {filename}.pdf: page {idx} ends on '{hit}' with its body on the "
                f"next page. The template's needspace should have prevented this; add a "
                f"\\needspace before that heading in output/{filename}.tex and recompile.")
