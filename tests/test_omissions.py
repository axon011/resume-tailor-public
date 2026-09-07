"""Omission gate: the only check that looks for content going MISSING or being ADDED
to an employer block. Tested against KNOWN answers on the example base CV: a check
that never fires looks exactly like a clean run.

Run: python -m pytest tests -q
"""
import io
import os

import pytest

from pipeline import profile
from pipeline.omissions import check_omissions

BASE = os.environ["RESUME_TEX_PATH"]
base = io.open(BASE, encoding="utf-8").read()
_P = profile.load()

PROTECTED = _P.protected_clauses["acme iot"]
BANNED = _P.banned_additions["acme iot"]


@pytest.mark.parametrize("clause", PROTECTED)
def test_example_base_contains_every_protected_clause(clause):
    assert clause in base, "candidate.example/resume.tex and gates.json disagree"


def _attribution_warnings(tex):
    return [x for x in check_omissions(tex, BASE, label="test") if x.startswith("ATTRIBUTION:")]


def test_unmodified_base_does_not_fire():
    assert _attribution_warnings(base) == []


def test_dropped_protected_clause_fires():
    clause = PROTECTED[0]
    tex = base.replace(clause, "did some retrieval work", 1)
    blob = " || ".join(_attribution_warnings(tex))
    assert "protected clause missing" in blob and clause in blob


def test_added_banned_word_fires():
    word = BANNED[0]
    # insert into the employer block: right after the first employer bullet marker
    marker = "\\item Built "
    assert marker in base, "example base changed; update the test"
    tex = base.replace(marker, f"\\item Built {word} ", 1)
    blob = " || ".join(_attribution_warnings(tex))
    assert f'"{word}" was ADDED' in blob


def test_live_project_drop_is_reported_when_the_jd_asks_for_it():
    tex = (base.replace("recipe-rag", "recipe-search").replace("Recipe-RAG", "Recipe-Search")
           .replace("recipes.jane-example.dev", "search.jane-example.dev"))
    jd = "We need a TypeScript / React web app engineer."
    w = check_omissions(tex, BASE, label="test", jd_text=jd)
    assert any("LIVE DEPLOYMENT DROPPED" in x for x in w)


def test_live_project_kept_is_silent():
    w = check_omissions(base, BASE, label="test", jd_text="TypeScript / React web app")
    assert not any("LIVE DEPLOYMENT DROPPED" in x for x in w)
