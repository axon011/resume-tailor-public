"""Test fixtures run against the SYNTHETIC profile in candidate.example/, never against
candidate/. Set before any pipeline module is imported: profile.PROFILE_DIR is read at
import time, and config.py hard-fails without the LLM/candidate variables.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLE = os.path.join(ROOT, "candidate.example")

os.environ["CANDIDATE_DIR"] = EXAMPLE
os.environ["RESUME_TEX_PATH"] = os.path.join(EXAMPLE, "resume.tex")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("LLM_PROVIDER", "openai")
# Header fields must come from the example profile, not from a developer's .env.
for _k in ("CANDIDATE_NAME", "CANDIDATE_LOCATION", "CANDIDATE_DEGREE", "CANDIDATE_CONTEXT",
           "RESUME_TEX_PATH_PARTTIME"):
    os.environ.pop(_k, None)

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
