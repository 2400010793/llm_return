"""Conservative public-source ingestion helpers.

These helpers are intentionally not anti-bot scrapers. They only fetch URLs
that the user is authorized to access, check robots.txt when possible, apply a
low request rate, and preserve the raw response for audit. Dynamic pages,
login walls, CAPTCHAs, and rate limits are reported rather than bypassed.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FetchPolicy:
    """Request policy for a permitted public endpoint."""

    user_agent: str = "llm-return-research/0.1 (academic prototype)"
    pause_seconds: float = 3.0
    timeout_seconds: float = 30.0
    check_robots: bool = True


def _robots_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, "/robots.txt", "", "", ""))


def robots_allowed(url: str, policy: FetchPolicy = FetchPolicy()) -> bool:
    """Return whether robots.txt allows this user agent to fetch a URL."""
    if not policy.check_robots:
        return True
    try:
        from urllib.robotparser import RobotFileParser

        parser = RobotFileParser()
        parser.set_url(_robots_url(url))
        parser.read()
        return parser.can_fetch(policy.user_agent, url)
    except OSError:
        # An unavailable robots file is not treated as permission to crawl.
        return False


def fetch_public_url(url: str, policy: FetchPolicy = FetchPolicy()) -> bytes:
    """Fetch one permitted public URL without bypassing access controls."""
    if not robots_allowed(url, policy):
        raise PermissionError(f"robots.txt does not permit fetching: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": policy.user_agent})
    try:
        with urllib.request.urlopen(request, timeout=policy.timeout_seconds) as response:
            payload = response.read()
    finally:
        time.sleep(policy.pause_seconds)
    return payload


def save_raw_response(url: str, output_path: str | Path, policy: FetchPolicy = FetchPolicy()) -> Path:
    """Fetch and save an authorized response plus its provenance metadata."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = fetch_public_url(url, policy)
    destination.write_bytes(payload)
    metadata = {
        "url": url,
        "user_agent": policy.user_agent,
        "fetched_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes": len(payload),
    }
    destination.with_suffix(destination.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return destination
