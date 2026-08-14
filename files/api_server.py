from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from leetcode_fetcher import fetch_submissions

app = FastAPI(title="LeetCode Local Fetcher API")

# Allow browser pages (running on LeetCode) to POST to this local server.
# This is intended for local development only. Do NOT expose it to public networks.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SYNC_STORE_PATH = Path(__file__).parent / "sync_store.json"


class FetchRequest(BaseModel):
    session: str
    max_items: int = 500


class SyncSubmission(BaseModel):
    submission_id: str = Field(..., min_length=1)
    problem_slug: str = Field(..., min_length=1)
    problem_title: str = Field(..., min_length=1)
    difficulty: str = Field("Medium", min_length=1)
    status: str = Field("Accepted", min_length=1)
    language: str = Field("cpp", min_length=1)
    user_id: int = Field(1, ge=1)
    timestamp: str = Field(..., min_length=1)
    runtime: float | None = None
    memory: float | None = None
    topics: List[str] = Field(default_factory=list)

    @field_validator("submission_id", mode="before")
    def normalize_submission_id(cls, value):
        if value is None:
            raise ValueError("submission_id is required")
        return str(value)

    @field_validator("problem_slug", mode="before")
    def normalize_slug(cls, value, info):
        if value:
            return str(value)
        return "unknown-problem"

    @field_validator("problem_title", mode="before")
    def normalize_title(cls, value, info):
        if value:
            return str(value)
        return "Unknown Problem"

    @field_validator("timestamp", mode="before")
    def normalize_timestamp(cls, value):
        if value is None:
            raise ValueError("timestamp is required")
        return str(value)

    @field_validator("runtime", mode="before")
    def normalize_runtime(cls, value):
        if value is None or value == "" or value == "N/A":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        match = __import__("re").search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None

    @field_validator("memory", mode="before")
    def normalize_memory(cls, value):
        if value is None or value == "" or value == "N/A":
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value)
        match = __import__("re").search(r"-?\d+(?:\.\d+)?", text)
        return float(match.group(0)) if match else None

    @field_validator("topics", mode="before")
    def ensure_topics_list(cls, value):
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)


class SyncPayload(BaseModel):
    username: str = Field(..., min_length=1)
    submissions: List[SyncSubmission] = Field(..., min_items=1)
    sync_mode: str = Field("sync_new")  # "sync_new" or "rebuild"
    rebuild_history: bool = Field(False)
    fetch_limit: Optional[int] = Field(None)


class SyncResponse(BaseModel):
    username: str
    total_submissions: int
    new_submissions: int
    new_found: int
    synced_submissions: int
    last_synced: str
    sync_mode: str


class StatusResponse(BaseModel):
    local_api: bool
    synced_users: int
    store_exists: bool
    last_synced: str | None = None
    total_submissions: int = 0
    last_username: str | None = None


SyncSubmission.model_rebuild()
SyncPayload.model_rebuild()
SyncResponse.model_rebuild()
StatusResponse.model_rebuild()


def _load_sync_store() -> Dict[str, Any]:
    if not SYNC_STORE_PATH.exists():
        return {}
    try:
        return json.loads(SYNC_STORE_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_sync_store(store: Dict[str, Any]) -> None:
    SYNC_STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def _to_dict(item: Any) -> Dict[str, Any]:
    if hasattr(item, "model_dump"):
        return item.model_dump()
    if hasattr(item, "dict"):
        return item.dict()
    return dict(item)


def _merge_sync_payload(payload: SyncPayload) -> Dict[str, Any]:
    store = _load_sync_store()
    is_rebuild = payload.sync_mode == "rebuild" or payload.rebuild_history

    print(f"[FastAPI /api/sync] Received payload: username='{payload.username}', sync_mode='{payload.sync_mode}', rebuild_history={payload.rebuild_history}, fetch_limit={payload.fetch_limit}, items={len(payload.submissions)}")

    username = payload.username
    if username in ["leetcode_user", "default_user", "placeholder"]:
        non_demo_users = [u for u in store.keys() if not store[u].get("is_demo_data", False)]
        if len(non_demo_users) == 1:
            username = non_demo_users[0]

    incoming_list = [_to_dict(item) for item in payload.submissions]
    if payload.fetch_limit and payload.fetch_limit > 0:
        incoming_list = incoming_list[:payload.fetch_limit]

    if is_rebuild:
        current = {
            "username": username,
            "submissions": incoming_list,
            "last_synced": datetime.now(timezone.utc).isoformat(),
            "total_submissions": len(incoming_list),
            "new_submissions": len(incoming_list),
            "new_found": len(incoming_list),
            "synced_submissions": len(incoming_list),
            "sync_mode": "rebuild",
        }
        store[username] = current
        _save_sync_store(store)
        print(f"[FastAPI /api/sync] REBUILD COMPLETE: Written {len(incoming_list)} submissions to store for user '{username}'.")
        return current

    # Incremental sync mode: merge new non-duplicate submissions
    current = store.get(username, {
        "username": username,
        "submissions": [],
        "last_synced": None,
        "total_submissions": 0,
    })

    existing_keys = set()
    for item in current.get("submissions", []):
        sub_id = str(item.get("submission_id", ""))
        p_slug = str(item.get("problem_slug", ""))
        ts = str(item.get("timestamp", ""))
        if sub_id:
            existing_keys.add(sub_id)
        if p_slug and ts:
            existing_keys.add(f"{p_slug}_{ts}")

    new_submissions = []
    for item in payload.submissions:
        d = _to_dict(item)
        sub_id = str(d.get("submission_id", ""))
        p_slug = str(d.get("problem_slug", ""))
        ts = str(d.get("timestamp", ""))
        
        is_dup = (sub_id and sub_id in existing_keys) or (p_slug and ts and f"{p_slug}_{ts}" in existing_keys)
        if not is_dup:
            new_submissions.append(d)
            if sub_id:
                existing_keys.add(sub_id)
            if p_slug and ts:
                existing_keys.add(f"{p_slug}_{ts}")

    current["submissions"].extend(new_submissions)
    current["last_synced"] = datetime.now(timezone.utc).isoformat()
    current["total_submissions"] = len(current["submissions"])
    current["new_submissions"] = len(new_submissions)
    current["new_found"] = len(payload.submissions)
    current["sync_mode"] = "sync_new"

    store[username] = current
    _save_sync_store(store)
    return current


@app.post("/api/sync", response_model=SyncResponse)
def sync_endpoint(payload: SyncPayload) -> Any:
    try:
        record = _merge_sync_payload(payload)
        return SyncResponse(
            username=record["username"],
            total_submissions=record["total_submissions"],
            new_submissions=record.get("new_submissions", 0),
            new_found=record.get("new_found", 0),
            synced_submissions=record["total_submissions"],
            last_synced=record["last_synced"],
            sync_mode=record.get("sync_mode", "sync_new"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status", response_model=StatusResponse)
def status_endpoint() -> Any:
    store = _load_sync_store()
    if not store:
        return StatusResponse(local_api=True, synced_users=0, store_exists=False)
    last_username, last_record = max(
        store.items(),
        key=lambda item: item[1].get("last_synced") or "",
    )
    return StatusResponse(
        local_api=True,
        synced_users=len(store),
        store_exists=True,
        last_synced=last_record.get("last_synced"),
        total_submissions=last_record.get("total_submissions", 0),
        last_username=last_username,
    )


@app.post("/fetch")
def fetch_submissions_endpoint(req: FetchRequest) -> Any:
    """Fetch submissions using the provided session cookie (POST body).

    WARNING: This endpoint is intended for localhost for your own use.
    Do not expose it to public networks. The server does not persist cookies.
    """
    if not req.session:
        raise HTTPException(status_code=400, detail="Missing session cookie")
    try:
        subs = fetch_submissions(req.session, max_items=req.max_items)
        return {"submissions": subs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload")
async def upload_export(file: UploadFile = File(...)) -> Any:
    """Accept an uploaded export file (JSON) from the browser and return the parsed submissions.

    This allows a browser paste + upload flow instead of a cookie.
    """
    content = await file.read()
    try:
        payload = json.loads(content)
        subs = payload.get("submissions") or payload.get("data") or payload
        return {"submissions": subs}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
