"""Tests for the claim-backing gate, on the example profile.

Each blocking test is a defect class that ACTUALLY SHIPPED past the regression gate in
real use. The false-positive tests matter just as much: the first version of this gate
flagged 86% of 329 real generations, which is a gate nobody would keep. Anything that
pushes it back toward blocking true claims is a regression.

Run: python -m pytest tests -q
"""
import pytest

from pipeline.claimgate import (
    ClaimError,
    check_claims,
    check_category_bloat,
    check_employer_contamination,
    check_unbacked_numbers,
    check_unbacked_skills,
    check_unbacked_tech_stack,
    check_unbacked_tools,
    employer_block,
    skill_categories,
)

EMPLOYER = "Acme IoT"

BASE = r"""
\jobtitle{Software Engineer}{Acme IoT GmbH}{Berlin}{Jan 2024 -- Dec 2025}
\begin{itemize}
  \item Developed Python and Go backend services over live MQTT sensor streams, exposed as
        versioned FastAPI REST APIs; containerized with Docker/Kubernetes and CI/CD.
  \item Led the containerization workstream for the AcmeHub platform ---
        SensorFlow, the alerting service, the dashboard; profiling, QA, and deployment integration.
  \item Owned the model-benchmarking workstream; evaluated retrieval speed and generation
        quality across model variants.
\end{itemize}

\jobtitle{Software Engineer Trainee}{Globex Bank}{Frankfurt}{Mar 2023 -- Nov 2023}
\begin{itemize}
  \item Mainframe applications (COBOL, JCL, DB2) with production deployment discipline.
\end{itemize}

\section{Projects}
\projecttitle{Recipe-RAG: Multi-Agent Cooking Assistant}{Next.js}
\item Hybrid retrieval (BM25 + dense with reciprocal rank fusion), RAGAs and MLflow
      evaluation, sub-200ms p95, call-level observability.

\section{Skills}
\skillcat{AI \& Agents}{LangGraph, LangChain, CrewAI, RAG, Prompt Engineering, MCP}
\skillcat{LLMOps}{Langfuse (Tracing, Evals), RAGAs (Faithfulness, Relevance), MLflow}
\skillcat{ML \& NLP}{Python, PyTorch, Hugging Face Transformers, scikit-learn}
\skillcat{Backend \& Infra}{Go, FastAPI, Docker, Kubernetes, Qdrant, ChromaDB, PostgreSQL}
\skillcat{Languages}{English (C2 Fluent), German (B2 Working), Spanish (Native)}
"""


# ── parsing ───────────────────────────────────────────────────────────────────

def test_skill_categories_parsed():
    cats = dict(skill_categories(BASE))
    assert len(cats) == 5
    assert "LangGraph" in cats["AI & Agents"]


def test_parenthesised_commas_do_not_split():
    tex = r"\skillcat{Data}{Vector Databases (Qdrant, ChromaDB), PostgreSQL}"
    tokens = skill_categories(tex)[0][1]
    assert tokens == ["Vector Databases (Qdrant, ChromaDB)", "PostgreSQL"]


def test_employer_block_stops_at_next_job():
    blk = employer_block(BASE, EMPLOYER)
    assert "SensorFlow" in blk
    assert "COBOL" not in blk          # must not bleed into the next employer


# ── check 1: unbacked skills ──────────────────────────────────────────────────

def test_jd_mirrored_skill_is_flagged():
    tex = BASE + "\n" + r"\skillcat{Statistics}{Statistical Analysis, Hypothesis Testing}"
    flagged = {ev for _, ev in check_unbacked_skills(tex, BASE)}
    assert "Statistical Analysis" in flagged
    assert "Hypothesis Testing" in flagged


def test_base_skills_are_all_backed():
    assert check_unbacked_skills(BASE, BASE) == []


def test_multiword_token_backed_by_its_head():
    tex = r"\skillcat{AI}{Multi-Agent Orchestration}"
    assert check_unbacked_skills(tex, BASE) == []


def test_language_proficiencies_not_flagged():
    tex = r"\skillcat{Languages}{English (C2 Fluent), German (B2 Working)}"
    assert check_unbacked_skills(tex, BASE) == []


# ── check 2: employer contamination (profile claimgate_vocabulary) ─────────────

@pytest.mark.parametrize("leak", [
    "multi-agent orchestration",
    "RAGAs evaluation",
    "MLflow experiment tracking",
])
def test_personal_work_inside_employer_block_is_flagged(leak):
    tex = BASE.replace("Owned the model-benchmarking workstream;",
                       f"Owned the model-benchmarking workstream with {leak};", 1)
    assert check_employer_contamination(tex, BASE, EMPLOYER)


@pytest.mark.parametrize("genuine", [
    "hybrid retrieval with BM25",
    "the RAG pipeline",
])
def test_employer_rag_work_is_not_flagged(genuine):
    """Capabilities the employer role may genuinely have covered are NOT leaks. The
    boundary is the named personal repos and their metrics, not the capability."""
    tex = BASE.replace("Owned the model-benchmarking workstream;",
                       f"Owned the model-benchmarking workstream with {genuine};", 1)
    assert not check_employer_contamination(tex, BASE, EMPLOYER)


def test_same_claims_are_fine_in_the_projects_section():
    assert check_employer_contamination(BASE, BASE, EMPLOYER) == []


def test_default_employer_comes_from_the_profile():
    """No employer argument = the profile's primary employer."""
    tex = BASE.replace("Owned the model-benchmarking workstream;",
                       "Owned the model-benchmarking workstream with RAGAs evaluation;", 1)
    assert check_employer_contamination(tex, BASE)


def test_capability_drift_observability_vs_profiling():
    tex = BASE.replace("profiling, QA, and deployment", "observability, QA, and deployment", 1)
    hits = check_employer_contamination(tex, BASE, EMPLOYER)
    assert any("observability" in why for why, _ in hits)


@pytest.mark.parametrize("word", [
    "Orchestrated containerization",
    "containerization workstream",
    "deployment integration",
    "evaluation benchmarks",
    "retrieval speed",
])
def test_capabilities_the_base_supports_are_not_flagged(word):
    tex = BASE.replace("Owned the model-benchmarking workstream;", f"{word};", 1)
    hits = check_employer_contamination(tex, BASE, EMPLOYER)
    assert not any(f"claimed at {EMPLOYER}" in why for why, _ in hits)


# ── check 3: category bloat ───────────────────────────────────────────────────

def test_extra_category_is_flagged():
    tex = BASE + "\n" + r"\skillcat{Data Science}{Python, PyTorch, scikit-learn}"
    assert any("categories vs" in why for why, _ in check_category_bloat(tex, BASE))


def test_overlapping_categories_flagged():
    tex = BASE + "\n" + r"\skillcat{AI Engineering}{LangGraph, LangChain, CrewAI, RAG, MCP}"
    assert any("share" in why for why, _ in check_category_bloat(tex, BASE))


def test_base_has_no_bloat():
    assert check_category_bloat(BASE, BASE) == []


# ── entry point + severity split ──────────────────────────────────────────────

def test_base_passes_itself():
    check_claims(BASE, BASE, "base")          # must not raise


def test_contamination_blocks():
    tex = BASE.replace("Owned the model-benchmarking",
                       "Owned RAGAs-scored model-benchmarking", 1)
    with pytest.raises(ClaimError, match="RAGAs"):
        check_claims(tex, BASE, "resume")


# ── check 5: unbacked numbers ─────────────────────────────────────────────────

def test_invented_number_is_flagged_and_base_number_is_not():
    base = r"\item Reached 0.91 hit@5 on a 50-question set."
    tex = r"\item Blocks fabricated claims across 47 applications; reached 0.91 hit@5."
    hits = check_unbacked_numbers(tex, base)
    assert any("'47'" in why for why, _ in hits)
    assert not any("'0.91'" in why or "'5'" in why for why, _ in hits)


def test_years_and_reformatted_base_numbers_pass():
    base = r"\item cut reporting lag from roughly 48 hours to a daily refresh; 2,000+ word reports."
    tex = r"\item Available full-time from November 2026; cut lag from 48h; 2000+ word reports."
    assert check_unbacked_numbers(tex, base) == []


def test_invented_number_blocks_a_resume():
    tex = BASE.replace("Led the containerization workstream",
                       "Led the containerization workstream across 47 services", 1)
    with pytest.raises(ClaimError, match="'47'"):
        check_claims(tex, BASE, "resume")


def test_numbers_are_not_checked_on_cover_letters():
    check_claims("Your 17,000 flats and 300 staff; I shipped Docker services.",
                 "Docker services.", "cover letter")          # must not raise


# ── check 6: unbacked tech ────────────────────────────────────────────────────

def test_tech_stack_token_not_in_base_is_flagged():
    base = r"\projecttitle{RAG Evaluation System}{FastAPI, Qdrant} PostgreSQL elsewhere."
    tex = (r"\projecttitle{RAG Evaluation System}{FastAPI, PostgreSQL-compatible pipelines, Qdrant"
           r" \,|\, \href{https://github.com/x/y}{y}}")
    hits = check_unbacked_tech_stack(tex, base)
    assert any("PostgreSQL-compatible pipelines" in why for why, _ in hits)
    assert not any("'FastAPI'" in why or "'Qdrant'" in why or "'y'" in why for why, _ in hits)


def test_tool_name_in_prose_flagged_unless_in_base():
    base = "Docker and Kubernetes with GitHub Actions."
    tex = r"\item Orchestrated Airflow DAGs on Kubernetes, at the helm of a Helm-free team."
    hits = check_unbacked_tools(tex, base)
    names = [why.split("'")[1] for why, _ in hits]
    assert names == ["Airflow", "Helm"]


def test_cover_letter_tools_are_advisory_not_blocking(capsys):
    base = "Docker and Kubernetes."
    check_claims("Your platform runs on Airflow and RabbitMQ; I shipped Docker services.",
                 base, "cover letter")                     # must not raise
    out = capsys.readouterr().out
    assert "Airflow" in out and "RabbitMQ" in out
    with pytest.raises(ClaimError, match="Airflow"):
        check_claims(r"\item Built Airflow DAGs.", base, "resume")


def test_unbacked_skill_is_advisory_not_blocking(capsys):
    tex = BASE + "\n" + r"\skillcat{Languages}{Fortran}"
    tex = tex.replace(r"\skillcat{Languages}{English (C2 Fluent), German (B2 Working), Spanish (Native)}", "", 1)
    check_claims(tex, BASE, "resume")          # must NOT raise
    assert "Fortran" in capsys.readouterr().out


def test_cover_letter_without_markup_is_clean():
    check_claims("Dear Hiring Team, I built RAG pipelines.\n", BASE, "cover")


# ── cover letters: the document class the gate was structurally blind to ──────

COVER = (
    "Dear Hiring Team,\n\n"
    "I spent the last two years doing exactly that at Acme IoT --- multi-agent systems, "
    "refined through actual user loops.\n\n"
    "Best regards,\nJane\n"
)


def test_cover_letter_contamination_is_caught():
    with pytest.raises(ClaimError, match="multi-agent"):
        check_claims(COVER, BASE, "cover")


def test_cover_letter_with_personal_marker_is_clean():
    ok = COVER.replace(
        "at Acme IoT --- multi-agent systems",
        "at Acme IoT. Independently I build multi-agent systems",
    )
    check_claims(ok, BASE, "cover")


def test_cover_letter_github_marker_counts_as_personal():
    ok = COVER.replace(
        "at Acme IoT --- multi-agent systems",
        "at Acme IoT. See github.com/jane-example for my multi-agent systems",
    )
    check_claims(ok, BASE, "cover")


def test_cover_letter_employer_alone_is_clean():
    plain = "I spent two years at Acme IoT building Go and Python backend services over MQTT."
    check_claims(plain, BASE, "cover")


def test_prose_check_does_not_fire_without_the_employer():
    check_claims("I built multi-agent systems with sub-200ms p95 latency.", BASE, "cover")


def test_layout_commands_are_not_numbers():
    """\\needspace{6\\baselineskip} is layout, not a claim; it blocked a resume on 2026-09-09."""
    tex = "\\needspace{6\\baselineskip}\n\\vspace{2pt}\n" + BASE
    check_claims(tex, BASE, "resume")          # must not raise
    with pytest.raises(ClaimError, match="'47'"):   # and the strip must not hide a real invented number
        check_claims("\\needspace{6\\baselineskip}\n" + BASE.replace("Led the containerization workstream",
                     "Led the containerization workstream across 47 services", 1), BASE, "resume")
