"""Fetch light company context (homepage text) to ground cover letters.

The goal is to give the LLM one or two concrete, verifiable facts about the
company — product, domain, recent positioning — so the cover letter can cite
something specific instead of reading the JD back at the reader.

We deliberately keep this lightweight: a single HTTP GET, HTML to text,
truncate to ~2000 chars. No Selenium, no retries — if it fails, the letter
just falls back to the JD-only path.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

import httpx

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; resume-tailor/1.0; +https://github.com/axon011/resume-tailor-public)"
    )
}


def _strip_html(html: str) -> str:
    """Collapse HTML into whitespace-normalized plain text."""
    # Drop script / style blocks entirely
    html = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    html = re.sub(r"<style\b[^>]*>.*?</style>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    # Replace tags with spaces, collapse whitespace
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _candidate_urls(url_or_domain: str) -> list[str]:
    """Given a company URL or bare domain, return URLs worth trying."""
    parsed = urlparse(url_or_domain)
    if not parsed.scheme:
        parsed = urlparse(f"https://{url_or_domain}")
    base = f"{parsed.scheme}://{parsed.netloc}"
    return [base, f"{base}/about", f"{base}/about-us", f"{base}/company"]


def fetch_company_context(url_or_domain: str, max_chars: int = 2000, timeout: float = 6.0) -> str:
    """Fetch a short plain-text summary of the company's homepage/about page.

    Returns '' on any failure so the caller can decide to skip the hook.
    """
    if not url_or_domain:
        return ""

    for candidate in _candidate_urls(url_or_domain):
        try:
            resp = httpx.get(candidate, headers=_HEADERS, timeout=timeout, follow_redirects=True)
            if resp.status_code != 200 or "text/html" not in resp.headers.get("content-type", ""):
                continue
            text = _strip_html(resp.text)
            if len(text) < 200:
                continue
            return text[:max_chars]
        except Exception:
            continue
    return ""
