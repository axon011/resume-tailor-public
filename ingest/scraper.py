"""Scrape job descriptions from URLs."""

import re

import requests
from bs4 import BeautifulSoup


def scrape_job(url: str) -> dict:
    """Fetch and extract job title, company, and description from a URL.

    Returns dict with 'title', 'company', 'description', 'url' keys.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")

    title = _extract_title(soup)
    company = _extract_company(soup, url)
    description = _extract_description(soup)

    return {"title": title, "company": company, "description": description, "url": url}


def _extract_company(soup: BeautifulSoup, url: str) -> str:
    """Try to extract company name from page or URL."""
    # Try common HTML selectors
    selectors = [
        ".company-name",
        "[class*='company']",
        ".topcard__org-name-link",
        "a[data-tracking-control-name*='company']",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)

    # Try LinkedIn URL pattern: ...-at-COMPANY-NAME-1234567
    match = re.search(r"-at-(.+?)-\d{5,}$", url)
    if match:
        return match.group(1).replace("-", " ").title()

    # Try page title: "Job Title at Company | LinkedIn"
    if soup.title:
        title_text = soup.title.get_text(strip=True)
        at_match = re.search(r"\bat\s+(.+?)(?:\s*[\|–-]\s*|\s*$)", title_text, re.IGNORECASE)
        if at_match:
            return at_match.group(1).strip()

    return ""


def _extract_title(soup: BeautifulSoup) -> str:
    """Try to extract job title from common patterns."""
    # Common selectors for job titles
    selectors = [
        "h1.job-title",
        "h1.posting-headline",
        "h1[class*='title']",
        "h1[class*='job']",
        ".job-title",
        ".posting-headline",
        "h1",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el and el.get_text(strip=True):
            return el.get_text(strip=True)
    return soup.title.get_text(strip=True) if soup.title else "Unknown Position"


def _extract_description(soup: BeautifulSoup) -> str:
    """Try to extract job description from common patterns."""
    # Remove script/style
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Common selectors for job descriptions
    selectors = [
        ".job-description",
        ".posting-description",
        "[class*='description']",
        "[class*='job-detail']",
        "[class*='jobDetail']",
        "article",
        "main",
    ]
    for sel in selectors:
        el = soup.select_one(sel)
        if el:
            text = el.get_text(separator="\n", strip=True)
            if len(text) > 100:
                return text

    # Fallback: get body text
    body = soup.find("body")
    if body:
        return body.get_text(separator="\n", strip=True)[:5000]

    return ""


def parse_jd_text(text: str, company: str = "") -> dict:
    """Wrap raw JD text into the same format as scrape_job output."""
    lines = text.strip().split("\n")
    title = lines[0] if lines else "Unknown Position"
    return {"title": title, "company": company, "description": text, "url": "manual-input"}
