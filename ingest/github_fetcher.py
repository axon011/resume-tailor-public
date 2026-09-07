"""Fetch public GitHub profile and repos for context, with local caching."""

import json
import os

import requests

import config

CACHE_FILE = os.path.join(os.path.dirname(__file__), ".github_cache.json")


def fetch_github_context(username: str = None, refresh: bool = False) -> str:
    """Fetch public repos and return a summary string for LLM context.

    Uses local cache by default. Pass refresh=True to re-fetch from GitHub API.
    """
    username = username or config.GITHUB_USERNAME

    if not refresh and os.path.exists(CACHE_FILE):
        return _load_cache("context")

    repos_data = _fetch_from_github(username)
    context = _format_context(username, repos_data)
    _save_cache(context, repos_data)

    return context


def get_github_repos(username: str = None, refresh: bool = False) -> list[dict]:
    """Get structured GitHub repo data for project injection.

    Returns list of dicts with name, description, language, topics, url.
    """
    username = username or config.GITHUB_USERNAME

    if not refresh and os.path.exists(CACHE_FILE):
        raw = _load_cache("repos")
        if raw:
            return _structure_repos(raw)

    repos_data = _fetch_from_github(username)
    context = _format_context(username, repos_data)
    _save_cache(context, repos_data)

    return _structure_repos(repos_data)


def _structure_repos(repos: list[dict]) -> list[dict]:
    """Convert raw GitHub API repos to clean structured format."""
    result = []
    for repo in repos:
        if repo.get("fork"):
            continue
        result.append({
            "name": repo["name"],
            "description": repo.get("description") or "No description",
            "language": repo.get("language") or "N/A",
            "topics": repo.get("topics", []),
            "stars": repo.get("stargazers_count", 0),
            "url": repo.get("html_url", ""),
        })
    return result


def _fetch_from_github(username: str) -> list[dict]:
    """Fetch repos from GitHub API."""
    url = f"https://api.github.com/users/{username}/repos"
    params = {"sort": "updated", "per_page": 30, "type": "owner"}

    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"   Warning: Could not fetch GitHub repos: {e}")
        return []


def _format_context(username: str, repos: list[dict]) -> str:
    """Format repo data into LLM context string."""
    non_fork = [r for r in repos if not r.get("fork")]
    lines = [f"GitHub: {username} - {len(non_fork)} public repos\n"]
    for repo in non_fork:
        name = repo["name"]
        desc = repo.get("description") or "No description"
        lang = repo.get("language") or "N/A"
        stars = repo.get("stargazers_count", 0)
        topics = ", ".join(repo.get("topics", []))
        line = f"- {name} ({lang}, {stars} stars): {desc}"
        if topics:
            line += f" [{topics}]"
        lines.append(line)
    return "\n".join(lines)


def _save_cache(context: str, repos_data: list[dict]):
    """Save GitHub data to local cache."""
    cache = {"context": context, "repos": repos_data}
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def _load_cache(key: str = "context"):
    """Load cached data by key."""
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return cache.get(key)
