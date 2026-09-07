# Resume Tailor

Tailors a LaTeX resume and cover letter to a job description with an LLM, then refuses to compile anything it cannot prove. Three gates run on the rendered LaTeX before pdflatex: a regression gate for banned claims and attribution slips, a claim-backing gate that traces every skill, number and tool name back to your base CV, and an omission gate that reports what the tailoring dropped.

The pipeline ships with no personal data. Everything about you lives in one gitignored directory, `candidate/`, and the code reads it at import time. The rules are generic; the facts are yours.

![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Why the gates exist

A tailoring LLM does three things you did not ask for. It moves personal-project work under your employer's heading. It invents numbers, including numbers about its own tooling. And it adds whatever the job description asks for, whether you have it or not. Prompting does not stop any of this reliably. Measured over 329 real generations, the first two showed up in a large share of outputs and the regression gate alone caught none of the third class.

So the gates are mechanical and run on the final LaTeX. A resume with an unbacked number does not compile. A cover letter that puts a personal repo in the same sentence as your employer does not compile. The `.tex` is saved for you to fix by hand, and nothing reaches a PDF until every artifact is clean.

## What it does

```
Job description (URL or .txt file)
        |
  1. Fetch JD, archive it under data/jd-archive/
  2. Optional pre-flight gate (language wall, years floor, seniority)   -> stops before any LLM call
  3. Parse base resume         LaTeX -> structured dict
  4. Fetch GitHub repos        offered as project candidates
  5. Tailor with LLM           JSON diff: summary, bullet order, project selection, skills
  6. Humanize                  strip AI writing patterns, auto-fix known overclaims
  7. Validate                  no fabrication, LaTeX escaping
  8. Render                    Jinja2 -> LaTeX
  9. GATE                      regression + claim-backing + omission, on every artifact
 10. Compile                   pdflatex, twice, only if all gates pass
 11. Cover letter (optional)   cites one verifiable fact from the company site; same gates
        |
output/<Prefix>_<Company>_<YYYYMMDD>.pdf
output/<Prefix>_CoverLetter_<Company>_<YYYYMMDD>.pdf

Then:  python ats_check.py output/<file>.pdf path/to/jd.txt
```

## Setup

### 1. Prerequisites

- Python 3.11+
- A LaTeX distribution with `pdflatex` (MiKTeX on Windows, MacTeX, or `texlive-latex-extra` on Linux)
- An LLM: any OpenAI-compatible endpoint, or the Claude Code CLI, or the `agy` CLI

### 2. Install

```bash
git clone https://github.com/axon011/resume-tailor-public.git
cd resume-tailor-public
pip install -r requirements.txt
cp .env.example .env
cp -r candidate.example candidate
```

### 3. Write your profile

`candidate/` holds three files. The example profile is a complete, synthetic "Jane Example" you can run as-is to see the pipeline work.

| File | What goes in it |
|---|---|
| `resume.tex` | Your base CV, using the `\jobtitle`, `\projecttitle`, `\skillcat` and `\edutitle` macros from the example. This is the ground truth every gate checks against. A claim that is not in here cannot appear in a tailored resume. |
| `gates.json` | The facts the gates need: your employers and which ones had real production deploys, the vocabulary of your personal projects, phrases that must never appear, the only evaluation scores that exist, availability stories per track, your header. Regexes are Python syntax, case-insensitive. |
| `rules.md` | Free-text rules injected verbatim into the tailoring and cover-letter prompts. Headings tagged `[tailoring]`, `[cover]` or `[both]` route each section. Every sentence here is something the model will treat as true about you. |

Then set `RESUME_TEX_PATH`, the LLM variables and `PDFLATEX_PATH` in `.env`. The pipeline fails at startup, naming the missing variable, rather than producing a generic letter.

### 4. Run

```bash
# Dry run first: see what the LLM would change, no PDF
python main.py --jd-file jd.txt --company acme --dry-run

# Full run with a cover letter
python main.py --jd-file jd.txt --company acme --company-url https://acme.example --cover-letter

# From a URL
python main.py --url "https://jobs.example.com/ai-engineer" --company acme --cover-letter

# Pick a track (defined under "tracks" in gates.json)
python main.py --jd-file jd.txt --company acme --track parttime

# Score the result
python ats_check.py output/Jane_Example_acme_20260417.pdf jd.txt
```

## The three gates

**Regression gate** (`pipeline/regression.py`). Runs per sentence on the rendered LaTeX. Bans the phrases listed in `gates.json`. Allows the word "production" only in a sentence that names an employer marked `production_ok`. Allows "real-time" only in the context your profile names. Flags a personal-project term sharing a sentence with your employer unless an ownership marker ("independently", "on my own projects", "eigene") is in the same sentence. Flags any evaluation score not in `real_metric_values`, and any retrieval metric stated without its retrieval context.

**Claim-backing gate** (`pipeline/claimgate.py`). Needs the base CV, which the regression gate never sees. On a resume it blocks: personal-project vocabulary inside an employer block, capability nouns claimed at an employer the base does not support, skill categories that grew or overlap, any number not in the base, any tech-stack token or well-known tool name not in the base. On a cover letter, numbers are skipped and tool names only warn, because a letter legitimately quotes the employer's own stack and figures.

**Omission gate** (`pipeline/omissions.py`). Advisory. Reports projects that vanished and whether their tech went with them, employer clauses listed under `protected_clauses` that did not survive, words listed under `banned_additions` that were added to an employer block, and the live project dropped when the job asks for exactly what it demonstrates. Dropping content is often correct tailoring, so this never blocks.

## Tracks

A profile may run more than one job search at once with different availability stories. The example defines `fulltime` and `parttime`; each has an `availability` text for the tailoring prompt and a `cover_directive` for the letter. The two are never mixed in one document. A track marked `geo_header: true` makes the resume header answer the geography question: a home city from `home_cities` when the job is near one, otherwise an explicit relocation line naming the job's city. Availability strings may use `{thesis_submission}` (from `.env`) and `{in_two_months}`, which is computed from today's date so it never goes stale.

## Project layout

```
resume-tailor-public/
  main.py                    CLI entry point
  config.py                  .env + profile -> settings; hard-fails on missing values
  ats_check.py               ATS keyword gate + LLM fit score for a finished PDF
  candidate.example/         synthetic profile: gates.json, rules.md, resume.tex
  candidate/                 YOUR profile (gitignored)
  pipeline/
    profile.py               loads candidate/ into typed fields and compiled regexes
    parser.py                LaTeX -> dict
    tailoring.py             LLM call, JSON diff, header location
    prompts.py               tailoring system prompt; injects rules.md [tailoring]
    humanizer.py             AI-pattern auto-fix and flags
    validator.py             fabrication check, LaTeX escaping
    regression.py            gate 1
    claimgate.py             gate 2
    omissions.py             gate 3
    renderer.py              Jinja2 -> LaTeX -> pdflatex
    llm_client.py            Claude Code CLI backend
    agy_client.py            agy / Gemini backend
  coverletter/
    generator.py             prompt, LLM, contractions, attribution auto-split
    prompts.py               anti-AI rules; injects rules.md [cover]
  ingest/                    JD scraper, GitHub fetcher, company homepage fetch
  templates/                 resume.tex.j2, cover_letter.tex.j2 (header from the profile)
  tests/                     106 tests, all against candidate.example
```

## Tests

```bash
python -m pytest tests -q
```

Every gate rule is tested in both directions: a string that must be caught and a string that must not be flagged. A gate that is only tested in the catch direction gets tightened until it fails the build on correct text. The suite runs against the example profile, so it passes on a fresh clone with no `candidate/` directory.

## Adapting the LaTeX template

`templates/resume.tex.j2` renders the tailored dict. The header, links, languages line and publications come from `gates.json`; everything else comes from your base CV through the parser. If your CV uses different macros, either rename them to the four the parser expects or update the regexes in `pipeline/parser.py`.

## Honest limitations

- The gates check overclaiming mechanically and omission advisorily. They do not check that a true claim is a good claim.
- The gates have no notion of subject. A cover letter sentence about the employer's stack can still trip the tool check, which is why it only warns there.
- `ingest/scraper.py` is best-effort. For anything unusual, save the JD to a file and use `--jd-file`.
- The template is single-column and opinionated. It is ATS-safe; it is not pretty.
- A passing ATS score does not predict an interview. Hard filters (language, years, applicant count) decide most outcomes before anyone reads the PDF. The optional pre-flight gate exists to stop you spending a generation on those.

## License

MIT
