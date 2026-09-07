"""The profile loader is the one place a silent failure would disable every gate, so
it must fail loudly and split rules.md the way the prompts expect.
"""
import os

import pytest

from pipeline import profile


def test_example_profile_loads():
    p = profile.load()
    assert p.primary_employer == "Acme IoT"
    assert p.production_ok_re.search("Deployed at Acme IoT")
    assert p.production_ok_re.search("MQTT broker")
    assert not p.production_ok_re.search("my hobby repo")
    assert "fulltime" in p.tracks and p.default_track == "fulltime"
    assert p.live_project_re.search("recipe-rag")


def test_missing_profile_fails_loudly(tmp_path):
    with pytest.raises(profile.ProfileError, match="gates.json"):
        profile.Profile(tmp_path)


def test_bad_json_fails_loudly(tmp_path):
    (tmp_path / "gates.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(profile.ProfileError, match="invalid JSON"):
        profile.Profile(tmp_path)


def test_no_employers_fails_loudly(tmp_path):
    (tmp_path / "gates.json").write_text('{"employers": []}', encoding="utf-8")
    with pytest.raises(profile.ProfileError, match="employers"):
        profile.Profile(tmp_path)


def test_empty_production_list_never_matches_everything(tmp_path):
    """An empty alternation matches every string and would disable the production rule."""
    (tmp_path / "gates.json").write_text(
        '{"employers": [{"name": "Nowhere Inc"}]}', encoding="utf-8")
    p = profile.Profile(tmp_path)
    assert not p.production_ok_re.search("anything at all")
    assert not p.personal_repo_re.search("anything at all")


def test_rules_are_routed_by_section_tag():
    p = profile.load()
    tailoring, cover = p.rules_for("tailoring"), p.rules_for("cover")
    assert "Frozen employer bullets" in tailoring and "Frozen employer bullets" not in cover
    assert "Cover-letter voice" in cover and "Cover-letter voice" not in tailoring
    assert "Attribution" in tailoring and "Attribution" in cover        # [both]
    assert "<!--" not in tailoring                                       # comment stripped


def test_header_env_overrides_profile(monkeypatch):
    p = profile.load()
    assert p.header_field("name") == "Jane Example"
    monkeypatch.setenv("CANDIDATE_NAME", "Someone Else")
    assert p.header_field("name") == "Someone Else"


def test_unknown_track_is_an_error():
    with pytest.raises(ValueError, match="unknown track"):
        profile.load().track_availability("night-shift")
