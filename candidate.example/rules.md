<!-- candidate.example/rules.md — injected VERBATIM into the tailoring and cover-letter
     system prompts under "CANDIDATE-SPECIFIC RULES". Write it as instructions to the
     model. Keep it factual: every sentence here is something the model will treat as
     true about you. Sections marked [tailoring] / [cover] / [both] tell the loader
     where to inject; an unmarked section goes to both. -->

## [both] Who the candidate is
- Software Engineer with two years at Acme IoT (industrial sensor platform): Python and Go
  services over MQTT streams, FastAPI, Docker, Kubernetes, CI/CD, plus the ingestion and
  evaluation layer behind AcmeHub. Earlier: mainframe banking work at Globex Bank.
- Independently builds and maintains the personal projects graph-search-kit, recipe-rag
  (LIVE at recipes.jane-example.dev), multi-agent pipeline and lora-tuner.
- M.Sc. Computer Science, thesis phase.

## [both] Attribution (hard)
- Acme IoT = backend/infra, containerisation, the ingestion and evaluation layer, the
  sub-200ms p95 benchmark. Agent work, graph search, LLM evaluation harnesses and
  fine-tuning are PERSONAL projects. Never put a personal project or its metric in the
  same sentence as Acme IoT without an explicit ownership marker ("Independently",
  "on my own projects").
- Acme IoT and Globex Bank work MAY be described as "production". Personal projects are
  "containerized, CI-tested"; the one exception is recipe-rag, which is genuinely live.

## [both] Real numbers (the only ones that exist)
- 0.91 hit@5 and 0.95 citation presence — recipe-rag evaluation set (personal).
- sub-200ms p95 — Acme IoT ingestion benchmark (employer, internal).
- Never write "0.82 faithfulness" or "0.89 relevance"; those were invented once and are banned.

## [tailoring] Summary never-drop clauses
- "two years of professional software experience"
- the word "Independently" in front of the personal-projects clause
- the availability clause from the candidate context (track-specific; never mix tracks)
- the publication clause naming arXiv:0000.00000

## [tailoring] Frozen employer bullets
- Acme IoT bullets are FROZEN: reorder freely, never reword. Copy them verbatim from the base.

## [cover] Cover-letter voice
- Open with one verifiable fact about the company, tie it to one concrete thing built.
- Cite at most two real metrics per letter, chosen to match what the company cares about.
- Never mention immigration status. Frame availability only.
