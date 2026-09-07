"""The LLM-JSON repair in tailoring.py, pinned to the defect that actually failed.

A perfect LLM response once failed 5/5 attempts on
    "items": "…",\n    },
because every repair attempt fixed a BACKSLASH and the defect was a COMMA. The response
is kept as a fixture in spirit (shape, not the 8 KB), plus the known-good cases that the
strip must leave alone.

Run: python -m pytest test_tailoring_json_repair.py -q
"""
import json

import pytest

from pipeline.tailoring import _strip_trailing_commas


@pytest.mark.parametrize("broken", [
    '{"a": "x",\n    }',
    '{"skill_order": [\n {"category": "ML", "items": "Python, PyTorch",\n },\n ]}',
    '{"projects": ["one", "two", ], }',
])
def test_trailing_commas_are_repaired(broken):
    with pytest.raises(json.JSONDecodeError):
        json.loads(broken)
    json.loads(_strip_trailing_commas(broken))          # must parse


@pytest.mark.parametrize("valid", [
    '{"summary": "Built services, shipped CI/CD, and more."}',
    '{"bullets": ["a, b", "c"], "n": 1}',
    '{"tech": "Python, Go, FastAPI"}',
])
def test_valid_json_is_untouched(valid):
    assert _strip_trailing_commas(valid) == valid
    json.loads(valid)


def test_shape_of_the_real_failure():
    """The real response: a trailing comma closing an object inside skill_order."""
    raw = ('{\n  "tagline": "Backend Engineer",\n  "skill_order": [\n    {\n'
           '      "category": "Backend \\\\& Infra",\n      "items": "Go, FastAPI",\n    },\n'
           '    {\n      "category": "ML \\\\& NLP",\n      "items": "Python",\n    }\n  ]\n}')
    with pytest.raises(json.JSONDecodeError, match="trailing comma"):
        json.JSONDecoder().raw_decode(raw)
    obj, _ = json.JSONDecoder().raw_decode(_strip_trailing_commas(raw))
    assert [c["category"] for c in obj["skill_order"]] == ["Backend \\& Infra", "ML \\& NLP"]
