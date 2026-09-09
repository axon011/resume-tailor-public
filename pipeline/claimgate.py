"""Claim-backing gate — the checks `regression.py` structurally cannot make.

`regression.py` matches a banned-phrase list. That catches known-bad *strings*, and
it is blind to the defect class that actually keeps shipping: claims that are
individually innocuous but **not backed by the base résumé**, or backed by the base
but **attached to the wrong employer**.

Two packages generated on one night passed the regression gate on both documents,
first try, with five live defects between them. All five need the base .tex as ground
truth, which regression.py never receives — hence a separate gate.

Checks implemented:

1. UNBACKED SKILLS — every token in a \\skillcat must appear somewhere in the base.
   One resume shipped "Statistical Analysis, Hypothesis Testing" solely because the JD
   asked for it.
2. EMPLOYER CONTAMINATION — personal-project work relocated into an employer's block.
   The profile's `claimgate_vocabulary` names the personal-project terms and metrics.
3. EMPLOYER CAPABILITY DRIFT — capability nouns claimed at the employer that the base's
   own block for that employer does not support. One resume claimed "observability"
   where the base says "profiling", landing an unbacked claim on the exact word the JD
   centred on.
4. SKILL-CATEGORY BLOAT — new JD-shaped categories added without the originals removed.
   One resume shipped 8 categories with 5- and 9-item overlaps.
5. UNBACKED NUMBERS — every number in the summary/experience/projects must already be in
   the base. A resume once shipped "blocks fabricated claims across 47 applications" — a
   fabricated number describing the fabrication gate. Nothing saw it: regression.py
   matches phrases, and this gate checked \\skillcat tokens only.
6. UNBACKED TECH — every token in a \\projecttitle tech stack, and every well-known tool
   name in the prose, must be in the base. Blocking on resumes; advisory on cover
   letters, where a tool name is often the EMPLOYER's stack being quoted back.

Deliberately NOT here: anything regression.py already covers. This gate is additive.
Every candidate-specific fact comes from the profile (candidate/gates.json).
"""
from __future__ import annotations

import re
from pathlib import Path

from . import profile as _profile

_P = _profile.load()


class ClaimError(Exception):
    """Raised when a generated artifact makes a claim the base does not support."""


# ── parsing ───────────────────────────────────────────────────────────────────

_SKILLCAT = re.compile(r"\\skillcat\{(.+?)\}\{(.+?)\}", re.S)
_JOBTITLE = re.compile(r"\\jobtitle\{(.+?)\}\{(.+?)\}")


# Layout-only commands carry digits that are not claims: \needspace{6\baselineskip}
# left a bare '6' after _strip_tex and blocked a resume (2026-09-09). Drop them whole.
_LAYOUT_ONLY = re.compile(r"\\(?:needspace|vspace\*?|hspace\*?|smallskip|medskip|bigskip|newpage|clearpage|pagebreak|nopagebreak)\s*(?:\{[^}]*\})?")


def strip_layout(s: str) -> str:
    return _LAYOUT_ONLY.sub(" ", s)


def _strip_tex(s: str) -> str:
    """Drop the LaTeX escaping that would otherwise break token comparison."""
    s = re.sub(r"\\[a-zA-Z]+\s*", " ", s)     # control sequences
    s = s.replace("\\&", "&").replace("\\%", "%").replace("\\_", "_")
    s = re.sub(r"[{}$~]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm(s: str) -> str:
    """Comparison form: lowercase, punctuation folded to single spaces."""
    return re.sub(r"[^a-z0-9+#/.]+", " ", _strip_tex(s).lower()).strip()


def _split_top_level(body: str) -> list[str]:
    """Split on commas OUTSIDE parentheses.

    A naive `.split(",")` shreds "Vector Databases (Qdrant, ChromaDB)" into two bogus
    tokens, one of which can never be backed — which is a gate that manufactures its
    own false positives.
    """
    out, buf, depth = [], [], 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [t.strip() for t in out if t.strip()]


def skill_categories(tex: str) -> list[tuple[str, list[str]]]:
    """[(category name, [skill tokens])] — parenthetical glosses kept with their token."""
    out = []
    for name, body in _SKILLCAT.findall(tex):
        out.append((_strip_tex(name), _split_top_level(_strip_tex(body))))
    return out


def employer_block(tex: str, employer: str) -> str:
    """The bullets belonging to one employer: from its \\jobtitle to the next, or to
    the next \\section — whichever comes first."""
    m = re.search(r"\\jobtitle\{[^}]*\}\{[^}]*" + re.escape(employer), tex)
    if not m:
        return ""
    rest = tex[m.end():]
    stop = len(rest)
    for pat in (r"\\jobtitle\{", r"\\section\{"):
        n = re.search(pat, rest)
        if n:
            stop = min(stop, n.start())
    return rest[:stop]


# ── check 1: unbacked skills ──────────────────────────────────────────────────

# Tokens legitimately absent from the base: language/proficiency wording, and
# generic connectors that a category label may carry.
_SKILL_ALLOW = re.compile(
    r"^(english|german|malayalam|native|fluent|actively improving|c1|b1|b2|a2)\b"
    r"|^(and|or|etc)$",
    re.I,
)


def _token_backed(token: str, base_norm: str) -> bool:
    """A token counts as backed if it, or its pre-parenthesis head, is in the base."""
    t = _norm(token)
    if not t or _SKILL_ALLOW.search(token.strip()):
        return True
    if t in base_norm:
        return True
    head = _norm(re.sub(r"\(.*?\)", "", token))          # "RAGAs (Faithfulness)" -> "ragas"
    if head and head in base_norm:
        return True
    # Multi-word tokens back off to the HEAD, not the longest word. The head carries
    # the identity: "Multi-Agent Orchestration" is backed because the base has
    # multi-agent work, while "Statistical Analysis" is not backed by "analysis"
    # appearing somewhere unrelated. Backing off to the longest word gets this exactly
    # backwards — it rejected all 62 "Multi-Agent Orchestration" tokens on the word
    # "orchestration" while the real head was sitting in the base.
    # 2026-08-28: a SLASH-JOINED token is a list, not one skill. "OpenAI/Anthropic APIs"
    # was reported unbacked even though the base lists OpenAI and Anthropic separately
    # under "LLMs & APIs". Treat it as backed only if EVERY alternative is backed, so a
    # real fabrication smuggled in as "Python/Rust" still fails on Rust.
    if "/" in head:
        parts = [q.strip() for q in head.split("/") if q.strip()]
        if len(parts) > 1 and all(_token_backed(q, base_norm) for q in parts):
            return True
    words = head.split()
    if len(words) > 1:
        for n in (2, 1):                       # "hugging face" then "hugging"
            if len(words) >= n:
                lead = " ".join(words[:n])
                if len(lead) >= 5 and lead in base_norm:
                    return True
    return False


def check_unbacked_skills(tex: str, base: str) -> list[tuple[str, str]]:
    base_norm = _norm(base)
    hits = []
    for cat, tokens in skill_categories(tex):
        for tok in tokens:
            if not _token_backed(tok, base_norm):
                hits.append((
                    f"skill '{tok}' (category '{cat}') is NOT in the base resume — "
                    "JD-mirrored fabrication unless you can point at where it is backed",
                    tok,
                ))
    return hits


# ── check 2 + 3: the employer block ────────────────────────────────────────────

# Personal-project vocabulary, from the profile. Present inside an employer block =
# relocated work. Deliberately the NAMED personal repos and their metrics, not the
# capability: implementing AI may well have been the candidate's actual mandate at the
# employer, and blocking the capability words turns real commercial experience into
# "backend + hobby projects" on every resume.
_PERSONAL_IN_EMPLOYER = _P.claimgate_vocabulary

# Capability nouns that drift onto an employer. Deliberately NARROW: only the
# LLMOps-flavoured words that a JD tempts the model into upgrading, and only where the
# base's own block for that employer does not support them.
#
# Words removed after measuring against 329 real generations, because each was a false
# positive: orchestrat* (the base block says Kubernetes, which IS orchestration),
# containeriz*, deployment, benchmark*, evaluation, retrieval, streaming, anomaly
# detection — all genuinely supported by the base employer block already. A gate that
# fires on true claims trains you to ignore it.
_CAPABILITY = re.compile(
    r"\b(observability|monitoring|alerting|tracing|instrumentation|"
    r"SLOs?|on-call|incident response)\b",
    re.I,
)


# Phrases that assign work to the candidate personally. Their presence in the same
# sentence is what makes an employer/personal-work co-occurrence correct. Mirrors
# regression.py's _PERSONAL_MARKER so the two gates agree on what "personal" looks like.
_GITHUB_MARKER = (r"|" + _P.github_marker) if _P.github_marker else ""
_PERSONAL_MARKER = re.compile(
    r"(?:my\s+own|personal|side|own)\s+(?:project|repo|time)\w*"
    r"|personally\s+(?:built|build|wrote|developed)"
    r"|\bindependently\b|\bon\s+my\s+own\b|\boutside\s+(?:of\s+)?work\b"
    + _GITHUB_MARKER +
    # GERMAN markers (added 2026-09-01). A cover letter was written in
    # German because the JD was, and this gate failed it on
    # "Neben der Arbeit baue ich EIGENE RAG- und Agenten-Systeme ..." — a sentence that
    # states ownership explicitly and correctly. The regex was English-only, so every
    # German-language cover letter would hit the same false positive, and German is the
    # majority language of this market. "eigen-" is the direct equivalent of "own".
    r"|\beigene[nrsm]?\b|\beigenständig\b|\bprivat(?:es|en|e)?\b"
    r"|\bin\s+meiner\s+Freizeit\b|\bneben\s+(?:der\s+)?Arbeit\b"
    r"|\bselbst\s+(?:gebaut|entwickelt|geschrieben|implementiert)\b",
    re.I,
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def check_prose_contamination(text: str, employer: str = None) -> list[tuple[str, str]]:
    """Sentence-level contamination check for documents with no employer markup.

    Cover letters carry no \\jobtitle, so `employer_block` finds nothing and the
    structural check no-ops — the gate reports clean on a document it never read.
    In one audit, 39 of 302 shipped cover letters put personal-project work in the same
    sentence as the employer, and neither gate saw it: the regression gate's vocabulary
    was narrower than this one's.

    A sentence naming the employer alongside personal work is a leak UNLESS it also
    carries an explicit personal marker ("independently", "on my own projects").
    """
    employer = employer or _P.primary_employer
    emp = re.compile(re.escape(employer), re.I)
    hits = []
    for sentence in _SENTENCE_SPLIT.split(text):
        if not emp.search(sentence) or _PERSONAL_MARKER.search(sentence):
            continue
        for pat, why in _PERSONAL_IN_EMPLOYER:
            m = pat.search(sentence)
            if m:
                hits.append((
                    f"'{m.group(0)}' shares a sentence with {employer} — {why}. "
                    f"Add an explicit personal marker or split the sentence",
                    sentence.strip()[:150],
                ))
                break
    return hits


def check_employer_contamination(tex: str, base: str, employer: str = None) -> list[tuple[str, str]]:
    employer = employer or _P.primary_employer
    block = employer_block(tex, employer)
    if not block:
        # No \jobtitle markup: prose document (cover letter). Fall back to sentences
        # rather than returning clean on an uninspected file.
        return check_prose_contamination(tex, employer)
    base_block = employer_block(base, employer)
    base_block_norm = _norm(base_block)
    hits = []

    for pat, why in _PERSONAL_IN_EMPLOYER:
        m = pat.search(block)
        if m and not pat.search(base_block):
            hits.append((
                f"'{m.group(0)}' appears inside the {employer} block — {why}. "
                f"Attribution leak: this is a personal project, not {employer} work "
                f"(the boundary is the named personal repos and their metrics, not the "
                f"capability the employer role may genuinely have covered)",
                m.group(0),
            ))

    for m in _CAPABILITY.finditer(block):
        word = m.group(0)
        stem = _norm(word)[:7]                       # observability/observab, profiling/profili
        if stem and stem not in base_block_norm:
            hits.append((
                f"'{word}' claimed at {employer}, but the base's {employer} block does not "
                f"support it (base says: {_base_capabilities(base_block) or 'nothing comparable'})",
                word,
            ))
    return hits


def _base_capabilities(base_block: str) -> str:
    found = sorted({m.group(0).lower() for m in _CAPABILITY.finditer(base_block)})
    return ", ".join(found)


# ── check 4: skill-category bloat ─────────────────────────────────────────────

def check_category_bloat(tex: str, base: str, overlap_threshold: int = 4) -> list[tuple[str, str]]:
    gen = skill_categories(tex)
    base_cats = skill_categories(base)
    hits = []

    if len(gen) > len(base_cats):
        hits.append((
            f"{len(gen)} skill categories vs {len(base_cats)} in the base — a JD-shaped "
            "category was ADDED without the original being removed",
            f"{len(gen)} categories",
        ))

    seen = {}
    for cat, tokens in gen:
        key = _norm(cat)
        if key in seen:
            hits.append((f"duplicate skill category '{cat}'", cat))
        seen[key] = {_norm(t) for t in tokens}

    names = list(seen)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = seen[a] & seen[b]
            if len(shared) >= overlap_threshold:
                sample = ", ".join(sorted(shared)[:5])
                hits.append((
                    f"skill categories '{a}' and '{b}' share {len(shared)} items ({sample}) "
                    "— they were duplicated, not rewritten",
                    f"{a} / {b}",
                ))
    return hits


# ── check 5: unbacked numbers ─────────────────────────────────────────────────

# The dynamic body: summary, experience, projects. The header carries a phone number,
# the skills block has its own check, and Publications/Education are static template
# text — so those are excluded rather than re-verified on every run.
_DYNAMIC_START = re.compile(r"\\section\{Professional Summary\}")
_DYNAMIC_END = re.compile(r"\\section\{(?:Technical )?Skills\}")


def _dynamic_body(tex: str) -> str:
    s = _DYNAMIC_START.search(tex)
    if not s:
        return tex                                    # no markup (cover letter, test snippet)
    e = _DYNAMIC_END.search(tex, s.end())
    return tex[s.end(): e.start() if e else len(tex)]


# A number is a digit run with optional thousands commas and one decimal part, not glued
# to a letter or a dot on the left (so "Qwen2", "GPT-4o", "v2" and IP-ish dotted runs do
# not split into bogus numerals) nor to another digit/comma on the right.
_NUMBER = re.compile(r"(?<![A-Za-z0-9.])(\d[\d,]*(?:\.\d+)?)(?![\d,])")
_YEAR = re.compile(r"^(?:19[89]\d|20[0-4]\d)$")


def _numbers(text: str) -> set[str]:
    """Normalised numeric tokens: '2,000+' -> '2000', 'sub-300ms' -> '300', '48\\%' -> '48'.

    Years are dropped — dates and availability months ("November 2026") are not claims
    this check is about, and a date that is wrong is caught by the human, not a gate.
    URL arguments are dropped because arXiv/LinkedIn ids are not claims either.
    """
    text = re.sub(r"\\\\\[[^\]]*\]", " ", text)          # \\[4pt] line spacing
    text = re.sub(r"\\(?:v|h)space\{[^}]*\}", " ", text)
    text = re.sub(r"\\href\{[^}]*\}", " ", text)          # keep the link TEXT, drop the URL
    out = set()
    for m in _NUMBER.finditer(text):
        n = m.group(1).rstrip(",")
        if _YEAR.match(n):
            continue
        out.add(n.replace(",", ""))
    return out


def _sentence_with(text: str, needle: str) -> str:
    for s in _SENTENCE_SPLIT.split(_strip_tex(text)):
        if re.search(r"(?<![A-Za-z0-9.])" + re.escape(needle) + r"(?![\d])", s):
            return s.strip()[:160]
    return needle


def check_unbacked_numbers(tex: str, base: str) -> list[tuple[str, str]]:
    """Every number in the generated body must appear somewhere in the base."""
    body = _dynamic_body(tex)
    fresh = _numbers(body) - _numbers(base)
    hits = []
    for n in sorted(fresh, key=lambda x: (len(x), x)):
        hits.append((
            f"number '{n}' is NOT in the base resume — a metric or count the model invented "
            f"(or a base metric re-expressed in a new form; restore the base wording)",
            _sentence_with(body, n),
        ))
    return hits


# ── check 6: unbacked tech tokens ─────────────────────────────────────────────

def _project_tech_stacks(tex: str) -> list[tuple[str, str]]:
    """[(project title, tech-stack argument)] read with brace balancing, since titles
    carry nested \\href{}{} — the same trap parser._parse_projects hit (Cohere #577)."""
    out = []
    for m in re.finditer(r"\\projecttitle\s*(?=\{)", tex):
        i = m.end()
        args = []
        for _ in range(2):
            while i < len(tex) and tex[i].isspace():
                i += 1
            if i >= len(tex) or tex[i] != "{":
                break
            depth, j = 0, i
            while j < len(tex):
                if tex[j] == "{" and tex[j - 1] != "\\":
                    depth += 1
                elif tex[j] == "}" and tex[j - 1] != "\\":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            args.append(tex[i + 1:j])
            i = j + 1
        if len(args) == 2:
            out.append((_strip_tex(args[0]), args[1]))
    return out


_STOP = {"and", "or", "the", "of", "for", "with", "a", "an", "to", "in", "on", "via", "over"}


def _stem(w: str) -> str:
    """Crude suffix fold so "automation" backs onto "automating" and "conversational"
    onto "conversation". Not a real stemmer; it only has to agree with itself."""
    # "ion" not "ation": stripping "ation" sent "automation" to "autom" while
    # "automating" went to "automat" — the two forms have to land on the same stem.
    for suf in ("ional", "ions", "ion", "ising", "izing", "ing", "ies",
                "ed", "al", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)]
    return w


def _tech_backed(token: str, base_norm: str, base_stems: set[str]) -> bool:
    """A tech-stack token is backed if the base has it whole, by its pre-parenthesis head,
    by its first two words, or — for descriptive tags the model recombines from base
    vocabulary ("Knowledge Graphs", "Structured Data Extraction") — if EVERY content
    word is a word the base uses.

    Measured 2026-09-06 over 23 real generations: a whole-token-only rule flagged 9
    descriptive tags for one real fabrication. The word-bag rule keeps the catch:
    "PostgreSQL-compatible pipelines" fails on "compatible", which the base never says.
    Deliberately NOT `_token_backed`'s single-word back-off, which passes that token
    on "postgresql" alone."""
    t = _norm(token)
    if not t or t in base_norm:
        return True
    head = _norm(re.sub(r"\(.*?\)", "", token))
    if head and head in base_norm:
        return True
    words = [w for w in head.split() if w not in _STOP]
    if not words:
        return True
    if len(words) > 1:
        lead = " ".join(words[:2])
        if len(lead) >= 5 and lead in base_norm:
            return True
    return all(_stem(w) in base_stems for w in words)


def check_unbacked_tech_stack(tex: str, base: str) -> list[tuple[str, str]]:
    base_norm = _norm(base)
    base_stems = {_stem(w) for w in base_norm.split()}
    hits = []
    for title, stack in _project_tech_stacks(tex):
        # "Python, Qdrant \,|\, \href{...}{repo}" — the repo link after the bar is not a tool.
        tools = re.split(r"\\,\|\\,|\|", stack, maxsplit=1)[0]
        for tok in _split_top_level(_strip_tex(tools)):
            if not _tech_backed(tok, base_norm, base_stems):
                hits.append((
                    f"tech-stack token '{tok}' on project '{title[:40]}' is NOT in the base — "
                    f"a JD tool appended to a project that does not use it",
                    tok,
                ))
    return hits


# Well-known tool names, matched CASE-SENSITIVELY as conventionally written so that
# "at the helm" or "spark interest" never fire. Names already in the base are skipped at
# check time, so this list can be generous. Ambiguous English words (Go, R, Ray, Spring,
# Glue, Hive, Beam, Vault, Lambda, Flink, Presto, Swift) are deliberately absent.
_TOOL_NAMES = [
    "Airflow", "Dagster", "Prefect", "dbt", "Spark", "PySpark", "Databricks", "Hadoop",
    "Kafka", "RabbitMQ", "ZeroMQ", "NATS", "Pulsar", "Celery", "Snowflake", "BigQuery",
    "Redshift", "Trino", "Iceberg", "Delta Lake", "Lake Formation", "Airbyte", "Fivetran",
    "Great Expectations", "DuckDB", "Polars",
    "Terraform", "Ansible", "Pulumi", "CloudFormation", "Helm", "Istio", "ArgoCD", "Argo CD",
    "Jenkins", "CircleCI", "Prometheus", "Grafana", "Datadog", "Splunk", "Kibana", "Jaeger",
    "OpenTelemetry", "Loki",
    "GCP", "Google Cloud", "Vertex AI", "SageMaker", "Lambda", "EKS", "GKE", "ECS", "Fargate",
    "Azure ML", "Azure OpenAI", "Azure Functions",
    "TensorFlow", "Keras", "JAX", "XGBoost", "LightGBM", "CatBoost", "ONNX", "TensorRT",
    "OpenVINO", "CUDA", "Triton", "vLLM", "TorchServe", "BentoML", "KServe", "Seldon",
    "Kubeflow", "Metaflow", "ZenML", "DVC", "Weights & Biases", "wandb",
    "LlamaIndex", "Llama Index", "Haystack", "DSPy", "AutoGen", "Semantic Kernel",
    "LangSmith", "DeepEval", "promptfoo", "Guardrails AI", "NeMo",
    "Neo4j", "Pinecone", "Weaviate", "Milvus", "pgvector", "FAISS",
    "Redis", "MongoDB", "Cassandra", "DynamoDB", "MySQL", "SQLite", "Supabase", "Firebase",
    "GraphQL", "Django", "Flask", "Spring Boot", ".NET", "C#", "Rust", "Scala", "Kotlin",
    "Java", "MATLAB", "Julia", "Vue", "Angular", "Svelte", "Node.js", "NestJS", "Express.js",
    "Streamlit", "Gradio", "Playwright", "Cypress", "Selenium",
    "Keycloak", "Auth0", "Okta", "OAuth2", "OIDC", "SAML",
    "Thingsboard", "ThingsBoard", "OPC UA", "Modbus",
]
_TOOL_RE = re.compile(
    r"(?<![A-Za-z0-9])(" + "|".join(re.escape(t) for t in sorted(_TOOL_NAMES, key=len, reverse=True))
    + r")(?![A-Za-z0-9])"
)


def check_unbacked_tools(tex: str, base: str, whole_document: bool = False) -> list[tuple[str, str]]:
    """Tool names in the prose that the base never mentions."""
    base_norm = _norm(base)
    text = tex if whole_document else _dynamic_body(tex)
    seen, hits = set(), []
    for m in _TOOL_RE.finditer(text):
        name = m.group(1)
        key = _norm(name)
        if key in seen or key in base_norm:
            continue
        seen.add(key)
        hits.append((
            f"tool '{name}' is named in the text but appears nowhere in the base resume — "
            f"if it is the employer's stack being described, say so in the sentence; if it is "
            f"claimed as his, it is unbacked",
            _sentence_with(text, name),
        ))
    return hits


# ── entry point ───────────────────────────────────────────────────────────────

def check_claims(
    tex: str,
    base_tex: str | Path,
    label: str = "document",
    employer: str = None,
) -> None:
    """Raise ClaimError if the artifact claims anything the base does not support.

    `base_tex` is the authoritative résumé (config.RESUME_TEX_PATH) — a path or its
    contents. Cover letters have no \\skillcat or \\jobtitle markup, so those checks
    no-op and only the shared ones apply.
    """
    employer = employer or _P.primary_employer
    base = base_tex
    # 2026-08-28 ROOT-CAUSE FIX. The old guard was:
    #     if isinstance(...) and "\\\\" not in str(base_tex)[:200] and Path(...).exists():
    # It used "does this string contain a backslash?" to tell a PATH from LaTeX CONTENT.
    # On Windows the path IS backslashes (C:\\Users\\...), so the file was NEVER read: `base`
    # stayed as the literal path string, no \\skillcat matched inside it, and every skill was
    # reported "NOT in the base resume" with "0 categories in the base". That is the
    # long-standing "claim gate misfire" - the gate has never actually validated anything
    # on this machine. Detect CONTENT by real LaTeX markers instead, and treat
    # anything else as a path.
    _cand = str(base_tex)
    _is_content = ("\\begin{document}" in _cand) or ("\\documentclass" in _cand) or len(_cand) > 4096
    if not _is_content:
        try:
            _p = Path(_cand)
            if _p.exists():
                base = _p.read_text(encoding="utf-8")
        except OSError:
            pass

    tex = strip_layout(tex)
    base = strip_layout(base)

    # Severity is split deliberately.
    #
    # BLOCKING — mechanically decidable. A personal-project metric sitting under an
    # employer heading, or eight skill categories where the base has five, is wrong on
    # its face; no judgement call can rescue it.
    #
    # ADVISORY — needs a human. An unbacked skill may be genuinely backed by a repo
    # that the base résumé simply never listed (pandas and NumPy are real in the
    # projects, absent from the base). Blocking those would stop every run over a
    # question only the author can answer, and a gate that blocks constantly gets
    # bypassed with --force, which is worse than no gate.
    blocking: list[tuple[str, str]] = []
    blocking += check_employer_contamination(tex, base, employer)
    blocking += check_category_bloat(tex, base)
    advisory = check_unbacked_skills(tex, base)
    # Checks 5 and 6. A cover letter legitimately carries numbers that are the COMPANY's
    # ("17,000 flats", "founded 1999") and names the employer's own stack, and this gate
    # has no notion of subject — so on a letter the number check is skipped and the tool
    # check only warns. On a resume every number and tool is a claim about him.
    is_cover = "cover" in label.lower()
    if is_cover:
        advisory += check_unbacked_tools(tex, base, whole_document=True)
    else:
        blocking += check_unbacked_numbers(tex, base)
        blocking += check_unbacked_tech_stack(tex, base)
        blocking += check_unbacked_tools(tex, base)

    def _dedupe(items):
        seen, out = set(), []
        for why, ev in items:
            if why in seen:
                continue
            seen.add(why)
            out.append((why, ev))
        return out

    blocking, advisory = _dedupe(blocking), _dedupe(advisory)

    if advisory:
        print(f"   Claim gate: {len(advisory)} unbacked claim(s) in {label} "
              "— confirm each is backed by a real repo, or delete it:")
        for why, _ in advisory:
            print(f"      - {why.split(' — ')[0]}")

    if not blocking:
        return

    lines = [f"Claim-backing gate FAILED for {label} — {len(blocking)} unbacked claim(s):"]
    for i, (why, evidence) in enumerate(blocking, 1):
        lines.append(f"  {i}. {why}")
        lines.append(f"     evidence: {evidence}")
    if advisory:
        lines.append(f"  (plus {len(advisory)} advisory warning(s) above)")
    lines.append(
        "\nEvery one of these passes the regression gate. Fix in the .tex, or prove the "
        "claim is backed and add it to the base resume first."
    )
    raise ClaimError("\n".join(lines))
