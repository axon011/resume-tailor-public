"""CLI entry point for resume-tailor."""

import argparse
import json
import os
import re
import sys
from datetime import datetime

# Windows consoles default to cp1252, which crashes print() on characters the
# LLM routinely emits in diagnostics (the arrow glyph, em-dash, smart quotes).
# A single logging line should never abort a generation run, so force UTF-8 on
# stdout/stderr with replacement. Caught after a tailoring run crashed on '→'
# inside project_changes (2026-05-27).
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

import config
from pipeline.parser import parse_resume
from ingest.scraper import scrape_job, parse_jd_text
from ingest.github_fetcher import fetch_github_context, get_github_repos
from pipeline.tailoring import tailor_resume
from pipeline.validator import validate
from pipeline.humanizer import humanize_check, humanize_auto_fix
from pipeline.renderer import render_latex, render_cover_letter, compile_pdf
from pipeline.regression import check_regressions, RegressionError
from pipeline.claimgate import check_claims, ClaimError
from pipeline.omissions import report as omission_report
from coverletter.generator import generate_cover_letter

# Optional pre-flight JD blocker gate. Point JD_GATE_PATH at a directory containing a
# `jd_blocker_check.py` that exposes `check(text, company, track) -> (hard, soft)`, each
# a list of (reason, matched_text). A hard hit stops the run before any LLM call; soft
# hits print and proceed. Unset = no gate. Never blocks a run on an import failure.
blocker_check = None
_gate_dir = os.getenv("JD_GATE_PATH", "").strip()
if _gate_dir:
    sys.path.insert(0, _gate_dir)
    try:
        from jd_blocker_check import check as blocker_check
    except Exception as _e:
        print(f"[warn] JD_GATE_PATH={_gate_dir!r} but jd_blocker_check could not be imported: {_e}",
              file=sys.stderr)


def main():
    args = parse_args()

    # 1. Get job description
    print("[1/6] Fetching job description...")
    # --company is a filename slug ("ml-reply-aiapp"); de-slug it so the ledger
    # dedup and alias gate inside blocker_check actually receive a company name.
    # Before 2026-09-04 the --jd-file path passed no company at all, so neither
    # check ever ran from main.py (only the manual step-2 gate run protected).
    _cli_company = (args.company or "").replace("-", " ").replace("_", " ").strip()
    if args.url:
        job_info = scrape_job(args.url)
    elif args.jd_text:
        job_info = parse_jd_text(args.jd_text, company=_cli_company)
    elif args.jd_file:
        with open(args.jd_file, "r", encoding="utf-8") as f:
            job_info = parse_jd_text(f.read(), company=_cli_company)
    else:
        print("Error: Provide --url, --jd-text, or --jd-file")
        sys.exit(2)

    # -- ARCHIVE THE POSTING, ALWAYS -------------------------------------------
    # A rejection-cause audit once could not verify 10 of 13 "language wall" tags
    # because the posting had never been saved anywhere, and the 3 that COULD be
    # checked did not support the tag. A cause tag without the employer text behind it
    # is a hypothesis, not evidence. This runs before any LLM call, so the posting
    # survives even when generation fails. data/ is gitignored.
    try:
        _arch = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "jd-archive")
        os.makedirs(_arch, exist_ok=True)
        _slug = args.company or job_info.get("company") or job_info.get("title") or "unknown"
        _slug = re.sub(r"[^A-Za-z0-9]+", "-", str(_slug)).strip("-").lower()[:48]
        _stamp = datetime.now().strftime("%Y-%m-%d")
        _dest = os.path.join(_arch, _stamp + "_" + _slug + ".txt")
        _body = ""
        if args.jd_file:
            _body = open(args.jd_file, encoding="utf-8").read()
        elif args.jd_text:
            _body = args.jd_text
        else:
            _body = job_info.get("description", "") or ""
        _hdr = chr(10).join([
            "# archived: " + _stamp,
            "# company: " + str(job_info.get("company") or args.company),
            "# title: " + str(job_info.get("title")),
            "# source: " + str(args.url or args.jd_file or "inline --jd-text"),
            "# track: " + str(getattr(args, "track", None)),
            "", "",
        ])
        with open(_dest, "w", encoding="utf-8") as _fh:
            _fh.write(_hdr + _body)
        print("   JD archived -> " + os.path.relpath(_dest))
    except Exception as _e:                      # never block a generation on this
        print("   [warn] JD archive failed: " + str(_e))

    company_name = job_info.get("company", "")
    if args.company_url:
        job_info["company_url"] = args.company_url
    print(f"   Job: {job_info['title']}")
    if company_name:
        print(f"   Company: {company_name}")

    # 1b. Pre-flight blocker gate — refuse to burn a tailor cycle on a known kill
    # criterion (C1 German, 3+yr floor, staff/lead, full-stack-required, intern/thesis,
    # freelance, already-rejected company). BLOCK exits before tailoring; --force
    # overrides for a deliberate longshot; soft flags print but proceed.
    if blocker_check is not None:
        gate_text = " ".join(str(job_info.get(k, "")) for k in ("title", "company", "description"))
        hard, soft = blocker_check(gate_text, company_name or None, args.track)
        if hard and not args.force:
            print("   [BLOCKED] pre-flight JD gate — not generating (no cycle spent):")
            for why, hit in hard:
                print(f"      - {why}  <- \"{hit}\"")
            print("   Override with --force if this is a deliberate longshot.")
            sys.exit(2)
        if hard:  # --force
            print(f"   [FORCED] {len(hard)} pre-flight blocker(s) overridden via --force:")
            for why, _ in hard:
                print(f"      - {why}")
        if soft:
            print(f"   [pre-flight] {len(soft)} soft flag(s) (proceeding):")
            for why, _ in soft:
                print(f"      - {why}")

    # 2. Parse current resume
    print("[2/6] Parsing resume...")
    # A track may use its own base CV (tagline, summary and section order differ).
    # Resolved ONCE here so the parser, the claim gate and the omission gate all
    # compare against the same file.
    BASE_CV = config.base_resume_path(args.track)
    if BASE_CV != config.RESUME_TEX_PATH:
        print(f"   Base CV: {args.track} variant ({os.path.basename(BASE_CV)})")
    resume_data = parse_resume(BASE_CV)
    print(f"   Parsed: {len(resume_data['experience'])} experiences, "
          f"{len(resume_data['projects'])} projects, "
          f"{len(resume_data['skills'])} skill categories")

    # 3. Fetch GitHub context
    if args.refresh_github:
        print("[3/6] Fetching GitHub context (fresh from API)...")
    else:
        print("[3/6] Loading GitHub context (cached)...")
    github_context = fetch_github_context(refresh=args.refresh_github)
    github_repos = get_github_repos(refresh=args.refresh_github)
    print(f"   {github_context.split(chr(10))[0]} ({len(github_repos)} repos available)")

    # 4. Tailor with LLM
    print("[4/6] Tailoring resume with LLM...")
    tailored = tailor_resume(resume_data, job_info, github_context, github_repos, instructions=args.instructions, track=args.track, location=args.location)

    # 5a. Humanizer auto-fix
    print("[5/7] Humanizing (removing AI writing patterns)...")
    if "summary" in tailored:
        tailored["summary"] = humanize_auto_fix(tailored["summary"])
    for exp in tailored.get("experience", []):
        exp["bullets"] = [humanize_auto_fix(b) for b in exp.get("bullets", [])]
    for proj in tailored.get("projects", []):
        proj["bullets"] = [humanize_auto_fix(b) for b in proj.get("bullets", [])]

    # 5b. Humanizer check (flag remaining issues)
    ai_warnings = humanize_check(tailored)
    if ai_warnings:
        print(f"   AI patterns flagged ({len(ai_warnings)}):")
        for w in ai_warnings[:5]:
            print(f"   - {w}")
        if len(ai_warnings) > 5:
            print(f"   ... and {len(ai_warnings) - 5} more")
    else:
        print("   Clean — no AI patterns detected")

    # 5c. Validate (no fabrication check)
    print("[6/7] Validating (no fabrication check)...")
    warnings = validate(resume_data, tailored)
    if warnings:
        print("   Warnings (review recommended):")
        for w in warnings:
            print(f"   - {w}")
    else:
        print("   Validation passed!")

    # 7. Render and compile
    if args.dry_run:
        print("[7/7] Dry run - showing changes:")
        _show_diff(resume_data, tailored)
        return

    steps = "8" if args.cover_letter else "7"
    company = args.company or _slugify(job_info.get("company") or job_info["title"])
    date_str = datetime.now().strftime("%Y%m%d")
    output_dir = os.path.join(os.path.dirname(__file__), "output")

    # Render every artifact and gate them ALL before compiling any. A banned
    # claim in the resume must not stop the cover letter from being generated
    # for inspection (and vice versa); and nothing compiles unless every
    # artifact is clean.
    artifacts = []  # list of (tex, label, filename)

    print(f"[7/{steps}] Rendering resume...")
    resume_tex = render_latex(tailored)
    resume_filename = f"{config.OUTPUT_FILENAME_PREFIX}_{company}_{date_str}"
    artifacts.append((resume_tex, "resume", resume_filename))

    if args.cover_letter:
        print(f"[8/{steps}] Generating cover letter...")
        cl_data = generate_cover_letter(tailored, job_info, track=args.track)
        cl_data["location"] = tailored.get("location", "Germany")
        cl_tex = render_cover_letter(cl_data)
        cl_filename = f"{config.OUTPUT_FILENAME_PREFIX}_CoverLetter_{company}_{date_str}"
        artifacts.append((cl_tex, "cover letter", cl_filename))

    _gate_all(artifacts, output_dir,
              jd_text=" ".join(str(job_info.get(k, "")) for k in ("title", "description")),
              base_cv=BASE_CV)

    for tex, label, fname in artifacts:
        path = compile_pdf(tex, output_dir, fname)
        print(f"   {label.capitalize()}: {path}")
        # Belt-and-braces against a silent success: a run that prints its steps
        # and exits 0 having produced no PDF is the worst outcome, because it
        # reads as done. compile_pdf already raises on a fatal LaTeX error, so
        # this only fires if it ever returns a path that is not on disk.
        if not os.path.isfile(path):
            print(f"\nFATAL: {label} reported success but no PDF exists at {path}.")
            sys.exit(1)


def _show_diff(original: dict, tailored: dict):
    """Show what changed between original and tailored resume."""
    # Summary
    if original["summary"] != tailored["summary"]:
        print("\n   SUMMARY: Rephrased")
        print(f"   Original: {original['summary'][:100]}...")
        print(f"   Tailored: {tailored['summary'][:100]}...")

    # Project order
    orig_titles = [p["title"] for p in original["projects"]]
    tail_titles = [p["title"] for p in tailored["projects"]]
    if orig_titles != tail_titles:
        print(f"\n   PROJECTS reordered:")
        for i, t in enumerate(tail_titles):
            marker = " *" if t != orig_titles[i] else ""
            print(f"   {i+1}. {t}{marker}")

    # Skill order
    orig_cats = list(original["skills"].keys())
    tail_cats = list(tailored["skills"].keys())
    if orig_cats != tail_cats:
        print(f"\n   SKILLS reordered:")
        for i, c in enumerate(tail_cats):
            marker = " *" if i < len(orig_cats) and c != orig_cats[i] else ""
            print(f"   {i+1}. {c}{marker}")

    # Experience bullet order
    for i, exp in enumerate(tailored["experience"]):
        if i < len(original["experience"]):
            if exp["bullets"] != original["experience"][i]["bullets"]:
                print(f"\n   EXPERIENCE[{i}] bullets reordered")


def _gate_all(artifacts: list, output_dir: str, jd_text: str = "", base_cv: str = None):
    """Hard-block compilation if ANY rendered artifact contains a banned claim.

    Blocking (not advisory) is the point: the old validator/grep warnings printed
    to stdout and were ignored, so banned strings shipped at a 55% rate. Every
    artifact is gated before any is compiled; on a hit the offending .tex is
    dumped for hand-fixing and the run exits non-zero with NO PDFs produced.
    """
    base_cv = base_cv or config.RESUME_TEX_PATH
    os.makedirs(output_dir, exist_ok=True)
    failures = []
    for tex, label, fname in artifacts:
        # Persist every rendered .tex up front, BEFORE gating, so a banned claim
        # in one artifact never discards a clean sibling's content (the gate
        # aborts before any compile). The flagged artifact is left on disk under
        # its real name for hand-fixing in place.
        tex_path = os.path.join(output_dir, f"{fname}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tex)
        try:
            check_regressions(tex, label)
            print(f"   Regression gate: clean ({label})")
        except RegressionError as e:
            failures.append((tex_path, str(e)))
        # Claim-backing gate — the checks regression.py structurally cannot make, because
        # they need the base resume as ground truth. Runs even when the regression gate
        # failed: seeing every defect in one pass beats fixing them one re-run at a time.
        try:
            check_claims(tex, base_cv, label)
            print(f"   Claim gate:      clean ({label})")
        except ClaimError as e:
            failures.append((tex_path, str(e)))
        # Omission gate — the ONLY check that looks for content going MISSING.
        # Advisory by design: dropping an off-lane project is legitimate tailoring
        # (system-prompt rule 3), so this never blocks the compile. It exists because
        # nothing else in the pipeline would notice, and twice it has mattered:
        # the renderer blanked every tech stack for ~2.5 months, and the selector
        # quietly stopped shipping the one live deployment.
        # Resume artifacts only. A cover letter has no Projects section and no
        # experience bullets by design, so comparing one against the base CV reports
        # every project and every bullet as "dropped" — 6 notes of pure noise on the
        # one real run, which is how this was caught.
        if "cover" not in label.lower() and "cover" not in fname.lower():
            try:
                omission_report(tex, base_cv, label, jd_text)
            except Exception as e:  # never let an advisory check break a good run
                print(f"   Omission gate:   skipped ({type(e).__name__}: {e})")
    if failures:
        for tex_path, msg in failures:
            print("\n" + msg)
            print(f"   -> fix in place and compile: {tex_path}")
        print("\nNo PDFs were compiled. Every artifact's .tex is saved in output/; "
              "fix the flagged one(s) and compile, or re-run to re-roll the LLM.")
        sys.exit(1)


def _slugify(text: str) -> str:
    """Convert text to a filename-safe slug."""
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"[\s]+", "_", text).strip("_")[:40]


def parse_args():
    p = argparse.ArgumentParser(description="Tailor resume for a job posting")
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--url", help="Job posting URL to scrape")
    group.add_argument("--jd-text", help="Job description text (paste directly)")
    group.add_argument("--jd-file", help="Path to a text file with the job description")
    p.add_argument("--company", help="Company name for output filename")
    p.add_argument("--dry-run", action="store_true", help="Preview changes without generating PDF")
    p.add_argument("--cover-letter", action="store_true", help="Also generate a cover letter PDF")
    p.add_argument("--refresh-github", action="store_true", help="Re-fetch GitHub data instead of using cache")
    p.add_argument("--instructions", help="Extra tailoring instructions (e.g. from career-ops Block E)")
    p.add_argument("--company-url", help="Company homepage URL for cover letter research hook")
    p.add_argument("--force", action="store_true", help="Override the pre-flight JD blocker gate (deliberate longshot)")
    p.add_argument("--location", default=None,
                   help="Resume-header residence, e.g. \"Berlin, Germany\". Only a city the "
                        "candidate actually lives in — never the JD city. Defaults to "
                        "CANDIDATE_LOCATION.")
    p.add_argument("--track", default=config.DEFAULT_TRACK, choices=sorted(config.AVAILABILITY),
                   help="Which availability story to tell (defined under 'tracks' in the "
                        "candidate profile). Default: %(default)s.")
    return p.parse_args()


if __name__ == "__main__":
    main()
