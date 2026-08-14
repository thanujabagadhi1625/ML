from __future__ import annotations

import json
import os
from getpass import getpass
from pathlib import Path
from typing import List

import requests


def _fetch_page(session: requests.Session, offset: int, limit: int = 20) -> dict:
    csrf = session.cookies.get("csrftoken", domain="leetcode.com") or session.cookies.get("csrftoken") or ""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Referer": "https://leetcode.com/submissions/",
        "X-Requested-With": "XMLHttpRequest",
    }
    if csrf:
        headers["x-csrftoken"] = csrf
        headers["X-CSRFToken"] = csrf

    url = f"https://leetcode.com/api/submissions/?offset={offset}&limit={limit}"
    try:
        r = session.get(url, headers=headers, timeout=30)
        if r.ok:
            return r.json()
    except Exception:
        pass

    # GraphQL fallback
    gql_url = "https://leetcode.com/graphql"
    query = """
    query submissionList($offset: Int!, $limit: Int!, $slug: String) {
      submissionList(offset: $offset, limit: $limit, questionSlug: $slug) {
        hasNext
        submissions {
          id
          title
          titleSlug
          statusDisplay
          lang
          timestamp
          url
          isPending
          memory
          runtime
        }
      }
    }
    """
    headers["Content-Type"] = "application/json"
    r2 = session.post(gql_url, json={"query": query, "variables": {"offset": offset, "limit": limit}}, headers=headers, timeout=30)
    r2.raise_for_status()
    return r2.json()


def _normalize_submission_item(item: dict) -> dict:
    title_slug = item.get("title_slug") or item.get("titleSlug") or item.get("title", "")
    title = item.get("title") or title_slug
    return {
        "user_id": 1,
        "question_id": title_slug,
        "title": title,
        "difficulty": item.get("difficulty", "Medium"),
        "status": item.get("status_display", item.get("statusDisplay", item.get("status", "Accepted"))),
        "timestamp": item.get("timestamp"),
        "runtime_ms": item.get("runtime", 0),
        "language": item.get("lang", "Python3"),
        "topic_tags": item.get("topic_tags", []),
        "acceptance_rate": item.get("acceptance_rate", 0.5),
    }


def fetch_and_save_submissions(output_path: Path | None = None, max_items: int = 500) -> Path:
    """
    Fetch recent submissions using the legacy submissions API and save a
    normalized JSON export compatible with the rest of the pipeline.

    The function looks for the `LEETCODE_SESSION` cookie in the environment.
    If not found, it prompts the user to paste it once (hidden input).
    """
    output_path = output_path or Path("leetcode_history_downloaded.json")

    cookie = os.environ.get("LEETCODE_SESSION")
    if not cookie:
        print("Provide your LeetCode session cookie (one-time). This value is sensitive and never sent anywhere by this script.")
        cookie = getpass("LEETCODE_SESSION cookie: ")

    submissions = fetch_submissions(cookie, max_items=max_items)

    out_payload = {"submissions": submissions}
    output_path.write_text(json.dumps(out_payload, indent=2, default=str), encoding="utf-8")
    return output_path


def fetch_submissions(session_cookie: str, max_items: int = 500) -> List[dict]:
    """Fetch submissions and return a normalized list (no file IO).

    This function keeps the session cookie only in memory for the duration
    of the HTTP requests and never persists it to disk.
    """
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0", "Referer": "https://leetcode.com"})
    session.cookies.set("LEETCODE_SESSION", session_cookie, domain="leetcode.com")

    # Perform initial request to get csrftoken
    try:
        r_init = session.get("https://leetcode.com", timeout=10)
    except Exception:
        pass

    submissions: List[dict] = []
    offset = 0
    page_size = 20
    while len(submissions) < max_items:
        payload = _fetch_page(session, offset, page_size)
        page_items = (
            payload.get("submissions_dump")
            or payload.get("submission_list")
            or payload.get("submissions")
            or (payload.get("data", {}).get("submissionList", {}).get("submissions") if payload.get("data") else None)
            or []
        )
        if not page_items:
            break
        for itm in page_items:
            submissions.append(_normalize_submission_item(itm))
            if len(submissions) >= max_items:
                break

        has_next = payload.get("has_next")
        if has_next is None and payload.get("data") and "submissionList" in payload["data"]:
            has_next = payload["data"]["submissionList"].get("hasNext")

        if has_next is False:
            break

        offset += len(page_items)

    return submissions


if __name__ == "__main__":
    out = fetch_and_save_submissions()
    print(f"Saved export to: {out}")
