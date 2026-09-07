"""Both-directions tests for the pre-ship regression gate, on the example profile.

Every rule is tested in BOTH directions: a string that must be caught and a string
that must not be flagged. A gate that only tests the catch direction gets tightened
until it fails the build on correct text, which is exactly how the attribution rule
once false-failed a correct summary.

Run: python -m pytest tests -q
"""
import pytest

from pipeline.regression import check_regressions, RegressionError


def flagged(text: str) -> bool:
    try:
        check_regressions(text)
        return False
    except RegressionError:
        return True


# --- 1. availability / immigration-status claims (profile banned_phrases) -------

CAUGHT_AVAILABILITY = [
    "Available immediately for the role.",
    "My own projects go deeper into PyTorch training. Available full-time immediately.",
    "I am available for full-time work immediately.",
    "Immediately available to join the team.",
    "Available now for a working student position.",
    "I can start immediately.",
    "Ich bin ab sofort verfuegbar.",
]

NOT_FLAGGED_AVAILABILITY = [
    "Finishing an M.Sc. at Example University; available for full-time work on thesis completion.",
    "M.Sc. Computer Science, Oct 2024 -- Oct 2026 (thesis phase, available full-time)",
    "Oct 2024 -- Oct 2026 (thesis phase; full-time on completion)",
]


@pytest.mark.parametrize("text", CAUGHT_AVAILABILITY)
def test_availability_caught(text):
    assert flagged(text), f"immediate-availability claim missed: {text!r}"


@pytest.mark.parametrize("text", NOT_FLAGGED_AVAILABILITY)
def test_availability_not_flagged(text):
    assert not flagged(text), f"correct availability phrasing false-flagged: {text!r}"


def test_status_claim_is_banned():
    assert flagged("Work permit holder, eligible to work.")


# --- 2. real time, all three spellings (profile realtime_ok) ---------------------

@pytest.mark.parametrize("spelling", ["real-time", "realtime", "real time"])
def test_realtime_caught(spelling):
    assert flagged(f"Dashboard visualizes cost breakdowns in {spelling} for AI systems.")


@pytest.mark.parametrize("spelling", ["real-time", "realtime", "real time"])
def test_realtime_allowed_in_the_profile_context(spelling):
    """The example profile marks MQTT sensor streams as genuinely real-time."""
    assert not flagged(
        f"Python and Go backend services connecting LLM workflows to {spelling} "
        "MQTT sensor streams, exposed as versioned FastAPI REST APIs."
    )


def test_realtime_substring_not_flagged():
    assert not flagged("Tracked wall-clock time and real user latency per request.")


# --- 3. "production" scoped to production_ok employers ---------------------------

CAUGHT_PRODUCTION = [
    "Automated release on every code push to production.",
    "Wired production alerts into the observability dashboard.",
    "Tracks per-model cost breakdowns for production AI systems.",
    "Built a production RAG pipeline over the corpus.",
    "Shipped production-grade inference code.",
]

NOT_FLAGGED_PRODUCTION = [
    "Developed and maintained mainframe applications (COBOL, JCL, DB2) in agile sprints; "
    "built structured debugging practices and production deployment discipline across "
    "enterprise banking systems.",
    "Delivered production-grade microservices at Acme IoT.",
    "Deployed the AcmeHub services to production at Acme IoT.",
]


@pytest.mark.parametrize("text", CAUGHT_PRODUCTION)
def test_production_caught(text):
    assert flagged(text), f"production overclaim missed: {text!r}"


@pytest.mark.parametrize("text", NOT_FLAGGED_PRODUCTION)
def test_production_not_flagged(text):
    assert not flagged(text), f"employer production bullet false-flagged: {text!r}"


def test_production_not_laundered_by_an_employer_mention_elsewhere():
    """An employer mention in a NEIGHBOURING sentence must not clear the claim."""
    text = ("Two years of software engineering, including enterprise banking work "
            "at Globex Bank. I ship LangGraph agents to production every week.")
    assert flagged(text)


# --- 4. attribution, clause-aware (profile personal_projects.vocabulary) ---------

CAUGHT_ATTRIBUTION = [
    "At Acme IoT I built a multi-agent orchestration pipeline with LangGraph.",
    "Acme IoT work included the recipe-rag evaluation harness.",
    "Owned the graph-search-kit workstream at Acme IoT GmbH.",
    "Fine-tuned a small model with lora-tuner for Acme IoT's internal platform.",
]

NOT_FLAGGED_ATTRIBUTION = [
    "Software Engineer with two years at Acme IoT GmbH building backend infrastructure "
    "for an industrial-IoT platform. Independently, I build multi-agent orchestration "
    "systems with LangGraph and fine-tune small models with lora-tuner.",
    "Two years at Acme IoT on backend and benchmarking; on personal projects I build "
    "LangGraph multi-agent orchestration.",
    "At Acme IoT I own containerization. On my own projects I built recipe-rag.",
    "Acme IoT gave me the infra grounding; I personally built graph-search-kit as a side project.",
    # German ownership markers
    "Neben der Arbeit bei Acme IoT baue ich eigene Multi-Agent-Systeme.",
]


@pytest.mark.parametrize("text", CAUGHT_ATTRIBUTION)
def test_attribution_caught(text):
    assert flagged(text), f"genuine mis-attribution missed: {text!r}"


@pytest.mark.parametrize("text", NOT_FLAGGED_ATTRIBUTION)
def test_attribution_not_flagged(text):
    assert not flagged(text), f"correct attribution false-flagged: {text!r}"


def test_projects_section_exempt_from_attribution():
    tex = ("\\section{Projects}\n"
           "\\item Architected multi-agent orchestration with LangGraph, reusing the "
           "benchmarking approach from Acme IoT.\n")
    assert not flagged(tex)


def test_attribution_rule_reactivates_after_projects_section():
    tex = ("\\section{Projects}\n"
           "\\item Architected multi-agent orchestration with LangGraph.\n"
           "\\section{Professional Experience}\n"
           "\\item At Acme IoT I built the multi-agent research pipeline.\n")
    assert flagged(tex)


# --- 5. metrics: invention and relabelling (profile real_metric_values) -----------

def test_invented_metric_value_is_caught():
    assert flagged("RAGAs scores: faithfulness 0.82, answer relevance 0.89.")


def test_real_metric_value_passes_with_context():
    assert not flagged("Reached 0.91 hit@5 on the recipe-rag retrieval eval set.")


def test_retrieval_metric_without_context_is_relabelling():
    assert flagged("Achieved 0.91 hit@5 on the benchmark.")


# --- 6. tool bans from the profile ----------------------------------------------

@pytest.mark.parametrize("text", [
    "Knowledge graph built on Neo4j.",
    "Retrieval stack uses LlamaIndex.",
    "Built a production RAG service.",
])
def test_profile_bans_fire(text):
    assert flagged(text), f"profile-banned pattern not caught: {text!r}"


def test_template_comments_are_skipped():
    assert not flagged("% Built a production RAG service on Neo4j, available immediately.")
