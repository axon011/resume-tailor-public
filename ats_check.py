"""ATS Resume Scorer v6 — Keyword analysis + LLM fit scoring + risk prediction.

Based on 82 applications, 23 rejections, 0 cold-apply interviews. Calibrated against actual outcomes.
Three scores: ATS Pass (keyword gate) + LLM Fit (semantic match) + Interview Probability.

v6 changes (Apr 25, 2026):
  - Split keyword score from structural score (structural inflated results)
  - Added 15+ synonym pairs (backend/back-end, testing/QA, etc.)
  - Upgraded LLM to sonnet for fit scoring
  - Removed text truncation (full resume+JD to LLM)
  - Recalibrated interview probability (divided by 3x, 0/82 reality)
  - Added German detection patterns for missed formats
  - Added keyword density per section
  - Added pre-check early exit for blockers

Usage:
    python ats_check.py output/resume.pdf /path/to/jd.txt
    python ats_check.py output/resume.pdf /path/to/jd.txt --no-llm   # skip LLM (fast mode)
"""
import sys
import re
import os
import json
import fitz
from collections import Counter

sys.stdout.reconfigure(encoding="utf-8")

# ─── LLM CONFIG ───
LLM_FIT_PROMPT = """You are a hiring manager screening resumes. Today's date is April 2026. Evaluate how well this candidate's ACTUAL EXPERIENCE matches this job description. Do NOT just check keyword overlap — assess whether the candidate can realistically do this job.

IMPORTANT: When calculating experience duration, note that TODAY IS APRIL 2026. "Jun 2024 -- Present" means ~22 months (almost 2 years), NOT 8-10 months. "Oct 2021 -- May 2022" is 7 months. Total professional experience is approximately 2.7 years.

## Candidate Resume:
{resume}

## Job Description:
{jd}

Score each dimension 1-5 and give a one-line reason. Be brutally honest.

Respond in this exact JSON format:
{{
  "domain_fit": {{"score": N, "reason": "Does their domain experience (IoT, fintech, healthcare, etc.) match?"}},
  "skills_demonstrated": {{"score": N, "reason": "Are required skills shown in experience bullets, or just listed in skills section?"}},
  "seniority_match": {{"score": N, "reason": "Does their experience depth match what the role needs?"}},
  "project_relevance": {{"score": N, "reason": "Do their projects demonstrate relevant capabilities?"}},
  "production_readiness": {{"score": N, "reason": "Have they shipped to production, or is it academic/toy projects?"}},
  "overall_fit": {{"score": N, "reason": "Would you interview this person? One-line honest assessment."}},
  "red_flags": ["list any dealbreakers a human reviewer would catch"],
  "strengths": ["list 2-3 things that would make a reviewer want to talk to them"]
}}"""


def _parse_fit_json(raw: str) -> dict | None:
    """Extract the scorecard JSON object from a raw LLM reply (strips code fences)."""
    s = (raw or "").strip()
    if not s:
        return None
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\n?", "", s)
        s = re.sub(r"\n?```$", "", s)
    json_match = re.search(r'\{[\s\S]*\}', s)
    try:
        return json.loads(json_match.group() if json_match else s)
    except json.JSONDecodeError:
        return None


def _fit_via_agy(prompt: str) -> dict | None:
    """Fit-score via Antigravity `agy` (Gemini, free Google quota). Saves Claude tokens.

    Low-stakes, high-volume call — ideal to offload. Returns None on any failure so
    the caller can fall back to Claude.
    """
    try:
        from agy_delegate import ask_agy
    except Exception:  # noqa: BLE001 — helper missing / import error
        return None
    raw = ask_agy(prompt, timeout_sec=180, workdir=os.path.dirname(os.path.abspath(__file__)))
    if not raw:
        return None
    return _parse_fit_json(raw)


def _fit_via_claude(prompt: str) -> dict | None:
    """Fit-score via the `claude` CLI (OAuth session). 3 retries on transient failure."""
    import subprocess
    import time

    attempts = 3
    last_err = "unknown"
    for attempt in range(1, attempts + 1):
        raw = ""
        try:
            result = subprocess.run(
                ["claude", "-p", "-", "--model", "claude-sonnet-4-6", "--output-format", "text",
                 "--strict-mcp-config", "--mcp-config", '{"mcpServers":{}}'],
                input=prompt, capture_output=True, text=True, timeout=180, encoding="utf-8"
            )
            raw = (result.stdout or "").strip()
            if not raw:
                last_err = (f"exit {result.returncode}, empty response"
                            + (f" (stderr: {result.stderr[:160]})" if result.stderr else ""))
            else:
                parsed = _parse_fit_json(raw)
                if parsed is not None:
                    return parsed
                last_err = f"JSON parse failed | head: {raw[:120]}"
        except subprocess.TimeoutExpired:
            last_err = "timed out (180s)"
        except FileNotFoundError:
            print(f"  [!] Claude CLI not found")
            return None  # not transient — no point retrying
        except Exception as e:
            last_err = str(e)

        if attempt < attempts:
            print(f"  [!] Claude fit attempt {attempt}/{attempts} failed ({last_err}); retrying...")
            time.sleep(2 * attempt)  # linear backoff: 2s, 4s

    print(f"  [!] Claude fit scoring failed after {attempts} attempts: {last_err}")
    return None


def llm_fit_score(resume_text: str, jd_text: str) -> dict | None:
    """Score resume↔JD fit. Routes to a cheap backend first to conserve Claude tokens.

    Provider via ATS_FIT_PROVIDER env:
      - "gemini" (default) → try `agy`/Gemini (free quota), fall back to Claude.
      - "gemini-only"      → Gemini only, no Claude fallback.
      - "claude"           → Claude only (original behavior).

    Retries / fallbacks keep reliability: a single slow or rate-limited call would
    otherwise silently drop the job (None -> Fit 0/5).
    """
    prompt = LLM_FIT_PROMPT.format(
        resume=resume_text[:8000],
        jd=jd_text[:6000]
    )

    provider = os.getenv("ATS_FIT_PROVIDER", "gemini").lower()

    if provider in ("gemini", "agy", "gemini-only", "agy-only"):
        result = _fit_via_agy(prompt)
        if result is not None:
            return result
        if provider in ("gemini-only", "agy-only"):
            print("  [!] LLM fit scoring failed (gemini-only, no fallback)")
            return None
        print("  [!] Gemini fit scoring unavailable; falling back to Claude")

    return _fit_via_claude(prompt)

# ─── STOP WORDS ───
STOP_WORDS = set("""
the a an and or but in on at to for of with by from as is are was were be
been being have has had do does did will would could should may might must
shall can this that these those it its you your we our they their he she
his her i me my not no all each every both few more most other some such
than too very just also about up out if then so what which who whom when
where why how any new well work working across including within between
through over under after after before during into onto role team company
position job candidate apply application required preferred nice plus bonus
strong good excellent ability skills experience knowledge understanding
familiarity proficiency years year months month full time part looking
seeking join responsible help support develop build create design implement
maintain ensure manage lead provide use using used based related relevant
similar equivalent etc like such various high low key main core primary
secondary first last next best top great real world end day way make take
get set need want know think see come go give able meet find keep let say
tell ask try start show move play run turn put open close read write learn
grow change follow stop call hold live stand bring begin seem leave feel
point thing right look still own always never often usually around along
away back even only already enough ever much many long while here there now
today really quite almost m f d w genders gender und der die das ein eine
oder mit fur bei von hybrid full-time permanent entry level career about
""".split())

# ─── KNOWN TECH TERMS ───
KNOWN_TECH = {
    "Python", "Java", "JavaScript", "TypeScript", "Rust", "Golang", "C++", "C#",
    "SQL", "NoSQL", "R",
    "PostgreSQL", "MySQL", "MongoDB", "Redis", "DynamoDB", "Elasticsearch",
    "Docker", "Kubernetes", "AWS", "GCP", "Azure", "Linux",
    "FastAPI", "Flask", "Django", "Spring Boot", "Node.js", "Next.js", "React", "Vue.js",
    "LangChain", "LangGraph", "LlamaIndex", "CrewAI", "AutoGen",
    "PyTorch", "TensorFlow", "scikit-learn", "Keras", "Hugging Face",
    "Qdrant", "Weaviate", "Pinecone", "ChromaDB", "pgvector",
    "OpenAI", "Anthropic", "Claude", "GPT", "Gemini", "Mistral", "DeepSeek",
    "RAG", "LLM", "NLP", "NER", "MLOps", "LLMOps",
    "MLflow", "RAGAS", "Langfuse",
    "Git", "GitHub", "GitLab", "CI/CD",
    "OAuth2", "REST", "GraphQL", "gRPC", "MQTT", "SSE", "MCP",
    "Bedrock", "SageMaker", "OpenSearch", "S3", "Lambda", "EC2",
    "Vertex AI", "BigQuery",
    "Neo4j", "n8n", "Terraform", "Airflow", "Kafka", "RabbitMQ",
    "Pydantic", "SQLAlchemy", "Celery", "Playwright",
}

# ─── SEMANTIC SYNONYMS ───
SYNONYMS = {
    "machine learning": ["ml", "machine-learning"],
    "deep learning": ["dl", "deep-learning"],
    "natural language processing": ["nlp"],
    "retrieval-augmented generation": ["rag"],
    "large language model": ["llm", "large language models", "llms"],
    "artificial intelligence": ["ai"],
    "continuous integration": ["ci/cd", "ci cd", "continuous deployment"],
    "docker": ["containerization", "containers"],
    "kubernetes": ["k8s"],
    "problem-solving": ["problem solving", "troubleshooting"],
    "cross-functional": ["cross functional", "interdisciplinary"],
    "full-stack": ["full stack", "fullstack"],
    "real-time": ["real time", "realtime"],
    "async": ["asyncio", "asynchronous"],
    "golang": ["go language", "go programming", "go (golang)"],
    "backend": ["back-end", "server-side", "server side"],
    "frontend": ["front-end", "client-side", "client side"],
    "testing": ["qa", "quality assurance", "test automation"],
    "microservices": ["micro-services", "distributed systems", "service-oriented"],
    "rest api": ["restful", "rest apis", "restful api"],
    "observability": ["monitoring", "tracing", "telemetry"],
    "fine-tuning": ["fine tuning", "finetuning", "model training"],
    "agile": ["scrum", "kanban", "sprint"],
    "serverless": ["lambda", "cloud functions", "faas"],
    "embeddings": ["embedding", "vector embeddings", "dense embeddings"],
    "vector database": ["vector db", "vector store", "vector databases"],
    "prompt engineering": ["prompt design", "prompt optimization"],
    "agentic": ["agentic ai", "ai agents", "agent-based"],
    "mlops": ["ml ops", "machine learning operations"],
    "llmops": ["llm ops", "llm operations"],
}

# ─── GERMAN LANGUAGE SIGNALS ───
GERMAN_REQUIRED = [
    r"deutsch\s*(?:flie[sß]end|c[12]|muttersprachl|verhandlungssicher|perfekt)",
    r"german\s*(?:fluent|c[12]|native|required|mandatory|must|proficien)",
    r"flie[sß]ende?\s*deutsch",
    r"verhandlungssicher(?:e[sn]?)?\s*deutsch",
    r"(?:must|need|require)\s+(?:to\s+)?(?:speak|know|have)\s+german",
    r"german\s+(?:is\s+)?(?:a\s+)?(?:must|required|mandatory|essential)",
    r"kommunikation\s+(?:auf|in)\s+deutsch",
    r"german\s+language\s+jd",
    r"sprache:\s*deutsch",
    r"verstehen\s+deutsch",
    r"gute\s+deutsch",
    r"deutsch.{0,15}und\s+englisch\s*(?:mindestens|mind\.?)?\s*(?:b[2-9]|c[12])",
    r"deutsch\s+und\s+englisch\s+(?:in\s+wort|flie[sß]end|kenntnisse)",
    r"(?:sehr\s+)?gute\s+(?:deutsch|german)\w*\s*(?:kenntnisse|skills)",
    r"deutsch\w*\s+(?:in\s+wort\s+und\s+schrift|written\s+and\s+spoken)",
    r"sie\s+sprechen\s+.*deutsch",
    r"du\s+sprichst\s+.*deutsch",
]

GERMAN_NICE = [
    r"german\s*(?:nice|plus|bonus|advantage|preferred|beneficial|helpful|b[12])",
    r"deutsch\s*(?:b[12]|von\s*vorteil|w[uü]nschenswert|hilfreich)",
    r"(?:nice|plus|bonus|advantage|preferred).*german",
    r"german.*(?:nice|plus|bonus|advantage|preferred|beneficial)",
    r"german\s*(?:is\s*)?(?:a\s*)?(?:plus|bonus)",
]

# ─── SENIORITY SIGNALS ───
SENIOR_SIGNALS = [
    r"(?:5|6|7|8|9|10)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|expertise|professional)",
    r"senior\s+(?:ai|ml|software|backend|fullstack|data|platform)",
    r"lead\s+(?:ai|ml|software|engineer)",
    r"staff\s+engineer",
    r"principal\s+engineer",
    r"(?:extensive|deep|significant)\s+(?:professional\s+)?experience",
]

MID_LEVEL_SIGNALS = [
    r"(?:2|3|4)\s*[-–]\s*(?:3|4|5|6)\s*(?:years?|yrs?)",
    r"(?:2|3|4)\+\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|expertise|professional)",
    r"mid.?(?:level|senior)",
    r"(?:experienced|proven)\s+(?:engineer|developer|professional)",
    r"seniority:\s*(?:experienced|mid)",
]

JUNIOR_SIGNALS = [
    r"(?:entry.?level|graduate|junior|trainee|intern|new\s*grad|early\s*career)",
    r"(?:0|1)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|expertise)",
    r"no\s+(?:prior\s+)?experience\s+(?:required|needed|necessary)",
    r"potential\s+over\s+(?:credentials|experience)",
    r"fresh\s+graduate",
]

# ─── COMPETITION SIGNALS ───
HIGH_COMPETITION = [
    r"(?:50|100|150|200|300|500)\s*(?:million|m)\+?\s*(?:users?|downloads?)",
    r"(?:faang|big\s*tech)\b",
    r"well.?known\s+(?:brand|company|product)",
    r"(?:series\s*[c-z]|ipo|public\s*company|fortune\s*\d+|nasdaq|nyse)",
    r"(?:thousands?|hundreds?)\s+(?:of\s+)?applicat",
    r"(?:evernote|meetup|notion|slack|spotify|shopify|stripe|figma|canva)",
    r"(?:5000|10000)\+?\s*(?:employees?|people|team)",
    r"massive\s+user\s+base",
]

LOW_COMPETITION = [
    r"(?:seed|pre.?seed|series\s*a|early.?stage|founding)",
    r"(?:small|lean|tight.?knit)\s+team",
    r"(?:startup|stealth)",
    r"(?:5|10|15|20|25|30)\s*(?:employees?|people|team\s*members?)",
]


def extract_jd_skills(jd_text: str) -> dict:
    """Extract skills/keywords from JD with smart filtering."""
    skills = {}
    jd_lower = jd_text.lower()

    for tech in KNOWN_TECH:
        if len(tech) < 2:
            continue
        # 2026-08-28 FIX: word boundaries were only applied to terms of <=3 chars.
        # Longer terms matched as bare substrings, so "Rust" matched inside "earn
        # trust" and was reported MISSING against every resume, and "Java" matches
        # inside "JavaScript". That silently deflated Keyword Match on every run.
        # Apply \b to any term that starts AND ends alphanumeric; leave symbol
        # terms (C++, C#, .NET, Node.js) on plain escaping, where \b misbehaves.
        if re.match(r"^[A-Za-z0-9]", tech) and re.search(r"[A-Za-z0-9]$", tech):
            pattern = rf"\b{re.escape(tech)}\b"
        else:
            pattern = re.escape(tech)
        count = len(re.findall(pattern, jd_text, re.IGNORECASE))
        if count > 0:
            skills[tech] = {"count": count, "type": "tech"}

    skill_phrases = [
        r"machine learning", r"deep learning", r"data science",
        r"computer vision", r"natural language processing",
        r"agentic workflow\w*", r"multi.?agent\w*", r"multi.?step",
        r"vector databas\w*", r"retrieval.augmented",
        r"prompt engineering", r"fine.?tuning", r"model training",
        r"agent orchestration", r"tool invocation", r"tool calling", r"tool use",
        r"evaluation framework\w*", r"testing framework\w*",
        r"quality assurance", r"automated (?:testing|evaluation|QA)",
        r"low.?latency", r"real.?time", r"event.?driven",
        r"speech.?to.?speech", r"text.?to.?speech",
        r"model context protocol",
        r"LLM.?as.?(?:a.?)?judge",
        r"A/?B test\w*",
        r"data pipeline\w*", r"ML pipeline\w*", r"ETL",
        r"microservice\w*", r"software engineer\w*",
        r"streaming\b", r"async\w*",
        r"unstructured data", r"semantic search",
        r"hybrid retrieval", r"embedding\w*", r"re.?ranking",
        r"guardrail\w*", r"safety check\w*",
        r"API design", r"RESTful\b",
        r"version control", r"code review\w*",
        r"cloud platform\w*", r"container\w*",
        r"data security", r"data privacy", r"GDPR",
        r"AI observability", r"model monitoring",
        r"problem.?solving", r"cross.?functional",
        r"full.?stack\b", r"front.?end\b", r"back.?end\b",
    ]
    for pattern in skill_phrases:
        matches = re.findall(pattern, jd_lower)
        if matches:
            clean = matches[0].strip().lower()
            if clean and len(clean) > 2 and clean not in skills:
                skills[clean] = {"count": len(matches), "type": "phrase"}

    return skills


def check_resume_match(resume_lower: str, skill: str) -> bool:
    skill_lower = skill.lower()
    if skill_lower in resume_lower:
        return True
    for canonical, syns in SYNONYMS.items():
        if skill_lower == canonical or skill_lower in syns:
            all_forms = [canonical] + syns
            for form in all_forms:
                if form in resume_lower:
                    return True
    return False


def count_resume_match(resume_lower: str, skill: str) -> int:
    skill_lower = skill.lower()
    count = resume_lower.count(skill_lower)
    for canonical, syns in SYNONYMS.items():
        if skill_lower == canonical or skill_lower in syns:
            for form in [canonical] + syns:
                if form != skill_lower:
                    count += resume_lower.count(form)
    return count


def analyze_real_world_risk(jd_text: str) -> dict:
    """Analyze JD for real-world rejection risk factors.
    Calibrated against 53 applications, 21 rejections."""
    jd_lower = jd_text.lower()
    risks = {}
    risk_score = 0  # negative = bad, 0 = neutral

    # 1. GERMAN LANGUAGE (biggest predictable filter, avg 1d rejection)
    # Check for negation: "no German required", "German not required", etc.
    german_negated = bool(re.search(
        r"(?:no|not|without|kein)\s+german\s+(?:required|needed|necessary|language)",
        jd_lower
    )) or bool(re.search(r"german\s+(?:is\s+)?not\s+(?:required|needed|necessary)", jd_lower))
    # 2026-08-28: a DISJUNCTION is not a requirement. DEMECAN asks for "entweder sehr gute
    # Deutsch- ODER Englischkenntnisse" and the bare pattern "gute deutsch" fired a -40
    # BLOCKER, dropping Risk to -37 and forcing a SKIP verdict on a role with no language
    # wall at all. English alone satisfies "X or Y", so treat an explicit either/or between
    # German and English as negation. Mirrors the same fix in jd_blocker_check.py. An "und"
    # / "and" pairing is untouched and still counts as required.
    german_negated = german_negated or bool(re.search(
        r"deutsch[\s\-\u2013]*(?:oder|or)\s+(?:sehr\s+gute?\s+)?englisch"
        r"|englisch[\s\-\u2013]*(?:oder|or)\s+(?:sehr\s+gute?\s+)?deutsch"
        r"|german\s+or\s+english|english\s+or\s+german",
        jd_lower))

    german_required = not german_negated and any(re.search(p, jd_lower) for p in GERMAN_REQUIRED)
    german_nice = not german_negated and any(re.search(p, jd_lower) for p in GERMAN_NICE)
    if german_required:
        risks["german"] = {"level": "BLOCKER", "detail": "German fluency required — auto-reject for B1", "penalty": -40}
        risk_score -= 40
    elif german_nice:
        risks["german"] = {"level": "RISK", "detail": "German nice-to-have — slight disadvantage vs native speakers", "penalty": -5}
        risk_score -= 5
    else:
        risks["german"] = {"level": "OK", "detail": "No German requirement — English-only", "penalty": 0}

    # 2. SENIORITY MISMATCH (Working Student vs FTE)
    is_senior = any(re.search(p, jd_lower) for p in SENIOR_SIGNALS)
    is_mid = any(re.search(p, jd_lower) for p in MID_LEVEL_SIGNALS)
    is_junior = any(re.search(p, jd_lower) for p in JUNIOR_SIGNALS)
    years_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|expertise|professional)', jd_lower)
    years_range = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*(?:years?|yrs?)', jd_lower)
    years_required = int(years_match.group(1)) if years_match else 0
    if years_range:
        years_required = max(years_required, int(years_range.group(1)))

    if is_senior or years_required >= 5:
        risks["seniority"] = {"level": "BLOCKER", "detail": f"Senior/{years_required}yr+ required — Working Student won't pass", "penalty": -35}
        risk_score -= 35
    elif years_required >= 3 or (is_mid and years_required >= 2):
        risks["seniority"] = {"level": "RISK", "detail": f"Mid-level/{years_required}yr+ — competing against FTE candidates", "penalty": -15}
        risk_score -= 15
    elif is_mid:
        risks["seniority"] = {"level": "RISK", "detail": "Mid-level role — Working Student disadvantage vs FTEs", "penalty": -10}
        risk_score -= 10
    elif is_junior:
        risks["seniority"] = {"level": "GOOD", "detail": "Junior/grad role — seniority is a match", "penalty": 5}
        risk_score += 5
    else:
        risks["seniority"] = {"level": "OK", "detail": "No explicit years requirement", "penalty": 0}
        risk_score += 0

    # 3. COMPETITION LEVEL (company size/brand → applicant volume)
    high_comp = any(re.search(p, jd_lower) for p in HIGH_COMPETITION)
    low_comp = any(re.search(p, jd_lower) for p in LOW_COMPETITION)
    if high_comp:
        risks["competition"] = {"level": "HIGH", "detail": "Large/well-known company — expect 100+ applicants", "penalty": -20}
        risk_score -= 20
    elif low_comp:
        risks["competition"] = {"level": "LOW", "detail": "Early-stage/small team — fewer applicants, more visibility", "penalty": 10}
        risk_score += 10
    else:
        risks["competition"] = {"level": "MEDIUM", "detail": "Mid-size company — moderate competition", "penalty": -2}
        risk_score -= 2

    # 4. VISA/WORK AUTHORIZATION RISK
    visa_blockers = [
        r"(?:must|need)\s+(?:have|hold|possess)\s+(?:a\s+)?(?:valid\s+)?(?:work\s+)?(?:permit|visa|authorization)",
        r"(?:eu|eea|schengen)\s+(?:citizen|national|passport)\s+(?:required|only|preferred)",
        r"no\s+(?:visa\s+)?sponsorship",
        r"(?:right|permission|authorization)\s+to\s+work.*(?:required|must|need)",
    ]
    visa_friendly = [
        r"visa\s+(?:sponsorship|support|assistance)\s+(?:available|offered|provided)",
        r"we\s+(?:sponsor|support|assist\s+with)\s+visa",
        r"relocation\s+(?:support|package|assistance)",
        r"blue\s*card",
    ]
    visa_block = any(re.search(p, jd_lower) for p in visa_blockers)
    visa_good = any(re.search(p, jd_lower) for p in visa_friendly)
    if visa_block and not visa_good:
        risks["visa"] = {"level": "RISK", "detail": "Work permit required — student permit may not satisfy", "penalty": -15}
        risk_score -= 15
    elif visa_good:
        risks["visa"] = {"level": "GOOD", "detail": "Visa sponsorship/relocation offered", "penalty": 5}
        risk_score += 5
    else:
        risks["visa"] = {"level": "OK", "detail": "No visa language — likely flexible", "penalty": 0}

    # 5. LOCATION RISK
    location_patterns = {
        "on-site-only": [r"(?:on.?site|in.?office)\s+(?:only|required|mandatory)", r"no\s+remote"],
        "remote-ok": [r"(?:fully\s+)?remote", r"work\s+from\s+(?:home|anywhere)"],
        "hybrid": [r"hybrid"],
    }
    is_remote = any(re.search(p, jd_lower) for p in location_patterns["remote-ok"])
    is_onsite = any(re.search(p, jd_lower) for p in location_patterns["on-site-only"])
    is_hybrid = any(re.search(p, jd_lower) for p in location_patterns["hybrid"])

    # Check if location is outside Germany
    non_de_cities = ["london", "paris", "amsterdam", "new york", "san francisco", "singapore", "tokyo"]
    de_cities = ["berlin", "munich", "hamburg", "frankfurt", "cologne", "stuttgart", "cottbus",
                 "karlsruhe", "bonn", "leipzig", "heilbronn", "bremen", "dresden", "dusseldorf",
                 "münchen", "köln", "düsseldorf", "graz", "vienna", "wien"]
    in_germany = any(c in jd_lower for c in de_cities) or "germany" in jd_lower or "deutschland" in jd_lower
    outside_de = any(c in jd_lower for c in non_de_cities) and not in_germany

    if outside_de and not is_remote:
        risks["location"] = {"level": "BLOCKER", "detail": "Location outside Germany — student visa not valid", "penalty": -40}
        risk_score -= 40
    elif is_onsite and not in_germany:
        risks["location"] = {"level": "RISK", "detail": "On-site required, unclear if Germany", "penalty": -10}
        risk_score -= 10
    elif is_remote or (is_hybrid and in_germany):
        risks["location"] = {"level": "GOOD", "detail": "Remote or hybrid in Germany — compatible with the candidate location", "penalty": 5}
        risk_score += 5
    else:
        risks["location"] = {"level": "OK", "detail": "Location seems compatible", "penalty": 0}

    # 6. POSTING AGE / URGENCY SIGNALS
    urgency_signals = [
        r"(?:immediate|asap|urgent|right\s+away|start\s+immediately)",
        r"(?:actively|currently)\s+(?:hiring|looking|seeking)",
    ]
    has_urgency = any(re.search(p, jd_lower) for p in urgency_signals)
    if has_urgency:
        risks["timing"] = {"level": "GOOD", "detail": "Active/urgent hiring — faster response likely", "penalty": 3}
        risk_score += 3
    else:
        risks["timing"] = {"level": "OK", "detail": "Standard posting", "penalty": 0}

    risks["_total"] = risk_score
    return risks


def compute_interview_probability(keyword_pct: float, risks: dict, llm_fit: float = None) -> float:
    """Combine keyword match + LLM fit + risk factors into interview probability.

    Calibrated: 0 interviews from 82 cold applications (Apr 2026).
    Previous formula was 3x too optimistic. Recalibrated downward.
    LLM fit is the primary signal — keyword match is pass/fail gate.
    """
    if keyword_pct < 50:
        base = 1
    elif keyword_pct < 70:
        base = 2
    elif keyword_pct < 85:
        base = 4
    else:
        base = 5

    if llm_fit is not None:
        if llm_fit >= 4.5:
            base += 12
        elif llm_fit >= 4.0:
            base += 8
        elif llm_fit >= 3.5:
            base += 4
        elif llm_fit >= 3.0:
            base += 1
        elif llm_fit >= 2.0:
            base -= 3
        else:
            base -= 5

    risk_total = risks.get("_total", 0)

    adjusted = base + risk_total
    return max(1, min(50, adjusted))


def check_ats(pdf_path: str, jd_path: str):
    doc = fitz.open(pdf_path)
    resume = ""
    for page in doc:
        resume += page.get_text()
    doc.close()

    with open(jd_path, "r", encoding="utf-8") as f:
        jd = f.read()

    resume_lower = resume.lower()
    jd_lower = jd.lower()

    # ═══════════════════════════════════════════════════════════════
    # PRE-CHECK: Early blocker detection (before wasting time)
    # ═══════════════════════════════════════════════════════════════
    german_negated = bool(re.search(
        r"(?:no|not|without|kein)\s+german\s+(?:required|needed|necessary|language)",
        jd_lower
    )) or bool(re.search(r"german\s+(?:is\s+)?not\s+(?:required|needed|necessary)", jd_lower))

    german_required = not german_negated and any(re.search(p, jd_lower) for p in GERMAN_REQUIRED)
    is_senior = any(re.search(p, jd_lower) for p in SENIOR_SIGNALS)
    years_match = re.search(r'(\d+)\+?\s*(?:years?|yrs?)\s*(?:of\s+)?(?:experience|expertise|professional)', jd_lower)
    years_required = int(years_match.group(1)) if years_match else 0

    blockers = []
    if german_required:
        blockers.append("GERMAN FLUENCY REQUIRED (you have B1)")
    if is_senior or years_required >= 5:
        blockers.append(f"SENIOR / {years_required}yr+ REQUIRED (you have ~2.7yr)")

    if blockers:
        print("=" * 65)
        print("PRE-CHECK: BLOCKERS DETECTED")
        print("=" * 65)
        for b in blockers:
            print(f"  [!!] {b}")
        print(f"\n  RECOMMENDATION: SKIP this role. Pipeline will continue but")
        print(f"  interview probability is near zero with these blockers.")
        print("=" * 65)

    jd_skills = extract_jd_skills(jd)

    critical = {}
    important = {}
    phrases = {}

    for skill, info in jd_skills.items():
        if info["type"] == "tech" and info["count"] >= 2:
            critical[skill] = info
        elif info["type"] == "tech":
            important[skill] = info
        elif info["type"] == "phrase":
            phrases[skill] = info

    # ═══════════════════════════════════════════════════════════════
    # PART 1A: KEYWORD MATCH (the real signal)
    # ═══════════════════════════════════════════════════════════════
    kw_score = 0
    kw_max = 0
    details = []

    print("=" * 65)
    print("PART 1A: KEYWORD MATCH (skills from JD)")
    print("=" * 65)
    print(f"\n--- CRITICAL SKILLS (mentioned 2+ times in JD, 3pts each) ---")
    for skill, info in sorted(critical.items(), key=lambda x: -x[1]["count"]):
        kw_max += 3
        matched = check_resume_match(resume_lower, skill)
        if matched:
            kw_score += 3
            rcount = count_resume_match(resume_lower, skill)
            print(f"  [Y] {skill} (JD:{info['count']}x Resume:{rcount}x)")
        else:
            details.append(f"MISSING CRITICAL: {skill}")
            print(f"  [X] {skill} (JD:{info['count']}x) -- MISSING")

    print(f"\n--- IMPORTANT TECH (mentioned in JD, 2pts each) ---")
    for skill in sorted(important.keys()):
        kw_max += 2
        matched = check_resume_match(resume_lower, skill)
        if matched:
            kw_score += 2
            print(f"  [Y] {skill}")
        else:
            details.append(f"Missing tech: {skill}")
            print(f"  [X] {skill} -- MISSING")

    print(f"\n--- SKILL PHRASES (1pt each) ---")
    for phrase in sorted(phrases.keys()):
        kw_max += 1
        matched = check_resume_match(resume_lower, phrase)
        if matched:
            kw_score += 1
            print(f"  [Y] \"{phrase}\"")
        else:
            details.append(f"Missing phrase: {phrase}")
            print(f"  [X] \"{phrase}\" -- MISSING")

    kw_pct = 100 * kw_score / max(kw_max, 1)

    print(f"\n  KEYWORD SCORE: {kw_score}/{kw_max} = {kw_pct:.0f}/100")
    if kw_pct >= 85:
        print("  STATUS: STRONG MATCH")
    elif kw_pct >= 70:
        print("  STATUS: PARTIAL MATCH -- add missing keywords")
    elif kw_pct >= 50:
        print("  STATUS: WEAK MATCH -- significant gaps")
    else:
        print("  STATUS: POOR MATCH -- wrong role?")

    if details:
        print(f"\n  GAPS TO FIX:")
        for d in details[:10]:
            print(f"    - {d}")

    # ═══════════════════════════════════════════════════════════════
    # PART 1B: KEYWORD DENSITY (where keywords appear)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n--- KEYWORD DENSITY (multi-section presence) ---")
    sections = {}
    resume_upper = resume.upper()
    section_markers = ["PROFESSIONAL SUMMARY", "PROFESSIONAL EXPERIENCE", "PROJECTS", "SKILLS", "EDUCATION"]
    positions = []
    for marker in section_markers:
        idx = resume_upper.find(marker)
        if idx >= 0:
            positions.append((idx, marker))
    positions.sort()

    for i, (pos, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(resume)
        sections[name] = resume[pos:end].lower()

    all_jd_skills = list(critical.keys()) + list(important.keys())
    multi_section = 0
    single_section = 0
    for skill in all_jd_skills:
        matched_sections = [s for s, text in sections.items() if check_resume_match(text, skill)]
        if len(matched_sections) >= 2:
            multi_section += 1
        elif len(matched_sections) == 1:
            single_section += 1
            print(f"  [!] {skill} — only in {matched_sections[0]} (add to another section)")

    if all_jd_skills:
        total_matched = multi_section + single_section
        print(f"\n  Multi-section keywords: {multi_section}/{total_matched}")
        print(f"  Single-section keywords: {single_section}/{total_matched} (weaker ATS signal)")

    # ═══════════════════════════════════════════════════════════════
    # PART 1C: STRUCTURAL SCORE (always passes — shown separately)
    # ═══════════════════════════════════════════════════════════════
    struct_score = 0
    struct_max = 0

    print(f"\n--- STRUCTURE (always-pass baseline, shown separately) ---")
    headings = ["PROFESSIONAL SUMMARY", "PROFESSIONAL EXPERIENCE", "PROJECTS", "SKILLS", "EDUCATION"]
    for heading in headings:
        found = heading in resume_upper
        struct_max += 1
        if found:
            struct_score += 1
        print(f"  [{'Y' if found else 'X'}] {heading}")

    contact_checks = {
        "Email": bool(re.search(r'[\w.]+@[\w.]+\.\w+', resume)),
        "GitHub/Portfolio": "github.com" in resume or "portfolio" in resume_lower,
        "LinkedIn": "linkedin.com" in resume,
    }
    for label, found in contact_checks.items():
        struct_max += 1
        if found:
            struct_score += 1
        print(f"  [{'Y' if found else 'X'}] {label}")

    metrics = re.findall(r'\d+%|\d+\+?\s*(?:users|customers|documents|queries)', resume)
    latency = re.findall(r'(?:sub-?)?\d+\s*ms', resume)

    struct_max += 3
    if metrics:
        struct_score += 1
        print(f"  [Y] Percentages/counts: {metrics[:5]}")
    else:
        print(f"  [X] No percentage/count metrics found")
    if latency:
        struct_score += 1
        print(f"  [Y] Latency metrics: {latency}")
    else:
        print(f"  [X] No latency metrics found")
    if len(metrics) >= 3:
        struct_score += 1
        print(f"  [Y] 3+ quantified achievements")
    else:
        print(f"  [!] Only {len(metrics)} quantified achievements (aim for 3+)")

    struct_max += 2
    lines = resume.strip().split("\n")
    name_first = bool(re.match(r'^[A-Z][A-Z\s]+$', lines[0].strip())) if lines else False
    garbled = len(re.findall(r'[^\x20-\x7E\n\r\t]', resume[:300]))
    readable = garbled < 5

    if name_first:
        struct_score += 1
        print(f"  [Y] Name in first line (correct reading order)")
    else:
        print(f"  [X] Name not in first line (reading order issue)")
    if readable:
        struct_score += 1
        print(f"  [Y] Text extraction clean ({garbled} non-ASCII chars)")
    else:
        print(f"  [X] Text extraction issues ({garbled} non-ASCII chars)")

    struct_pct = 100 * struct_score / max(struct_max, 1)
    print(f"\n  STRUCTURAL SCORE: {struct_score}/{struct_max} = {struct_pct:.0f}/100 (baseline, not predictive)")

    # Combined ATS for backward compatibility
    total_score = kw_score + struct_score
    total_max = kw_max + struct_max
    ats_pct = 100 * total_score / max(total_max, 1)
    print(f"  COMBINED ATS: {total_score}/{total_max} = {ats_pct:.0f}/100")

    # ═══════════════════════════════════════════════════════════════
    # PART 2: REAL-WORLD RISK ANALYSIS (the human filter)
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 65}")
    print("PART 2: REAL-WORLD RISK ANALYSIS")
    print("Calibrated against 53 apps, 21 rejections (as of Apr 2026)")
    print("=" * 65)

    risks = analyze_real_world_risk(jd)

    level_icons = {
        "BLOCKER": "!!",
        "HIGH": "!!",
        "RISK": "! ",
        "OK": "  ",
        "GOOD": "++",
        "LOW": "++"
    }

    risk_order = ["german", "seniority", "competition", "visa", "location", "timing"]
    for factor in risk_order:
        if factor in risks:
            r = risks[factor]
            icon = level_icons.get(r["level"], "  ")
            penalty_str = f"{r['penalty']:+d}" if r["penalty"] != 0 else " 0"
            print(f"  [{icon}] {factor.upper():12s} | {r['level']:8s} | {penalty_str:>4s}pts | {r['detail']}")

    # ═══════════════════════════════════════════════════════════════
    # PART 3: LLM FIT SCORING (semantic match — the real filter)
    # ═══════════════════════════════════════════════════════════════
    llm_score = 0
    skip_llm = "--no-llm" in sys.argv
    fit_result = None

    if not skip_llm:
        print(f"\n{'=' * 65}")
        print("PART 3: LLM FIT SCORING (semantic match)")
        print("How well does your ACTUAL experience match this role?")
        print("=" * 65)

        fit_result = llm_fit_score(resume, jd)
        if fit_result:
            dimensions = ["domain_fit", "skills_demonstrated", "seniority_match",
                          "project_relevance", "production_readiness", "overall_fit"]
            dim_labels = {
                "domain_fit": "Domain Fit",
                "skills_demonstrated": "Skills Proven",
                "seniority_match": "Seniority Match",
                "project_relevance": "Project Relevance",
                "production_readiness": "Production Ready",
                "overall_fit": "Overall Fit",
            }
            total = 0
            count = 0
            for dim in dimensions:
                if dim in fit_result:
                    s = int(fit_result[dim].get("score", 0))
                    r = fit_result[dim].get("reason", "")
                    total += s
                    count += 1
                    bar = "#" * s + "." * (5 - s)
                    icon = "++" if s >= 4 else "  " if s == 3 else "!!" if s == 2 else "XX"
                    print(f"  [{icon}] {dim_labels.get(dim, dim):18s} [{bar}] {s}/5  {r}")

            llm_score = round(total / max(count, 1), 1)
            print(f"\n  LLM FIT SCORE: {llm_score}/5.0")
            if llm_score >= 4.0:
                print("  ASSESSMENT: STRONG FIT -- experience genuinely matches")
            elif llm_score >= 3.0:
                print("  ASSESSMENT: MODERATE FIT -- partial match, some gaps")
            elif llm_score >= 2.0:
                print("  ASSESSMENT: WEAK FIT -- significant gaps, likely rejection")
            else:
                print("  ASSESSMENT: POOR FIT -- domain/experience mismatch")

            if fit_result.get("red_flags"):
                print(f"\n  RED FLAGS:")
                for flag in fit_result["red_flags"]:
                    print(f"    - {flag}")
            if fit_result.get("strengths"):
                print(f"\n  STRENGTHS:")
                for s in fit_result["strengths"]:
                    print(f"    + {s}")
        else:
            print("  [!] LLM scoring unavailable — using keyword-only mode")

    # ═══════════════════════════════════════════════════════════════
    # PART 4: INTERVIEW PROBABILITY (combined)
    # ═══════════════════════════════════════════════════════════════
    prob = compute_interview_probability(kw_pct, risks, llm_score if fit_result else None)

    print(f"\n{'=' * 65}")
    print("INTERVIEW PROBABILITY")
    print("=" * 65)
    print(f"  Keyword Match:      {kw_pct:.0f}/100 ({'PASS' if kw_pct >= 70 else 'WEAK' if kw_pct >= 50 else 'FAIL'})")
    print(f"  Structural:         {struct_pct:.0f}/100 (baseline)")
    if fit_result:
        print(f"  LLM Fit Score:      {llm_score}/5.0")
    print(f"  Risk Adjustment:    {risks['_total']:+d} pts")
    print(f"  Interview Chance:   {prob:.0f}%")

    if prob >= 20:
        print(f"  VERDICT: STRONG -- prioritize, send LinkedIn outreach TODAY")
        verdict = "STRONG"
    elif prob >= 10:
        print(f"  VERDICT: GOOD -- worth applying")
        verdict = "GOOD"
    elif prob >= 5:
        print(f"  VERDICT: LONGSHOT -- apply only if fast, don't over-invest")
        verdict = "LONGSHOT"
    elif prob >= 3:
        print(f"  VERDICT: UNLIKELY -- skip unless dream role + referral")
        verdict = "UNLIKELY"
    else:
        print(f"  VERDICT: SKIP -- blocker detected, don't waste time")
        verdict = "SKIP"

    risk_blockers = [f for f in risk_order if f in risks and risks[f]["level"] == "BLOCKER"]
    high_risks = [f for f in risk_order if f in risks and risks[f]["level"] in ("HIGH", "RISK")]

    if risk_blockers:
        print(f"\n  BLOCKERS: {', '.join(b.upper() for b in risk_blockers)}")
        print(f"  >> Skip this role or address blockers before applying")
    elif high_risks:
        print(f"\n  WATCH: {', '.join(h.upper() for h in high_risks)}")
        print(f"  >> Apply but temper expectations")

    llm_tag = f" | Fit {llm_score}/5" if fit_result else ""
    print(f"\n{'=' * 65}")
    print(f"  FINAL: KW {kw_pct:.0f}/100 | ATS {ats_pct:.0f}/100{llm_tag} | Risk {risks['_total']:+d} | Interview {prob:.0f}% | {verdict}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "output/resume.pdf"
    jd = sys.argv[2] if len(sys.argv) > 2 else "jd.txt"
    check_ats(pdf, jd)
