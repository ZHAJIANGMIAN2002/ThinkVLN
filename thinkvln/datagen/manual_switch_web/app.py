from __future__ import annotations

import hashlib
import json
import random
import secrets
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from thinkvln.datagen.generation.watcher_utils import append_jsonl, load_jsonl


APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))
ROLE_ADMIN = "admin"
ROLE_ANNOTATOR = "annotator"


@dataclass
class ManualSwitchConfig:
    bundle_root: Path
    manifest_file: Path
    output_file: Path
    summary_full_path: Path | None = None
    database_file: Path | None = None
    image_stride: int = 3
    max_samples: int | None = None
    shuffle: bool = False
    seed: int | None = None
    title: str = "Watcher Manual Switch Annotation"
    page_size: int = 20
    claim_lease_seconds: int = 1800
    bootstrap_admin_user: str = "admin"
    bootstrap_admin_password: str | None = None


def hash_password(password: str) -> str:
    salt = secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
    return f"{salt}:{digest}"


def verify_password(password: str, stored: str) -> bool:
    parts = str(stored).split(":", maxsplit=1)
    if len(parts) != 2:
        return False
    salt, expected = parts
    got = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 120_000).hex()
    return secrets.compare_digest(got, expected)


@dataclass
class ManualSwitchStore:
    config: ManualSwitchConfig
    samples: List[Dict[str, Any]]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.sample_by_id = {sample["sample_id"]: sample for sample in self.samples}
        self.db_path = (self.config.database_file or self.config.output_file.with_suffix(".sqlite3")).resolve()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._seed_tasks()
        self._bootstrap_admin_if_needed()
        self._import_existing_annotations()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    role TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS invite_codes (
                    code TEXT PRIMARY KEY,
                    remaining_uses INTEGER NOT NULL,
                    expires_at INTEGER,
                    created_by_user_id INTEGER,
                    created_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS tasks (
                    sample_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    assigned_user_id INTEGER,
                    lease_until INTEGER,
                    updated_at INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS annotations (
                    sample_id TEXT PRIMARY KEY,
                    user_id INTEGER NOT NULL,
                    should_switch INTEGER,
                    updated_at INTEGER NOT NULL
                );
                """
            )
            conn.commit()

    def _seed_tasks(self) -> None:
        now = int(time.time())
        sample_ids = {sample["sample_id"] for sample in self.samples}
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "DELETE FROM tasks WHERE sample_id NOT IN ({})".format(",".join("?" for _ in sample_ids))
                if sample_ids
                else "DELETE FROM tasks",
                tuple(sample_ids),
            )
            conn.execute(
                "DELETE FROM annotations WHERE sample_id NOT IN ({})".format(",".join("?" for _ in sample_ids))
                if sample_ids
                else "DELETE FROM annotations",
                tuple(sample_ids),
            )
            conn.executemany(
                """
                INSERT OR IGNORE INTO tasks(sample_id, status, assigned_user_id, lease_until, updated_at)
                VALUES (?, 'pending', NULL, NULL, ?)
                """,
                [(sample["sample_id"], now) for sample in self.samples],
            )
            conn.commit()

    def _bootstrap_admin_if_needed(self) -> None:
        with self._connect() as conn:
            existing = conn.execute("SELECT COUNT(*) AS c FROM users WHERE role = ?", (ROLE_ADMIN,)).fetchone()
            if int(existing["c"]) > 0:
                return
            if not self.config.bootstrap_admin_password:
                raise RuntimeError(
                    "No admin user found. Set MANUAL_SWITCH_ADMIN_PASSWORD or pass --bootstrap_admin_password."
                )
            now = int(time.time())
            conn.execute(
                """
                INSERT INTO users(username, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, 1, ?)
                """,
                (
                    self.config.bootstrap_admin_user,
                    hash_password(self.config.bootstrap_admin_password),
                    ROLE_ADMIN,
                    now,
                ),
            )
            conn.commit()
            print(f"[manual-switch] bootstrapped admin user: {self.config.bootstrap_admin_user}")

    def _import_existing_annotations(self) -> None:
        if not self.config.output_file.exists():
            return
        rows = load_jsonl(self.config.output_file)
        now = int(time.time())
        with self._connect() as conn:
            for row in rows:
                sample_id = str(row.get("sample_id", "")).strip()
                if sample_id not in self.sample_by_id:
                    continue
                raw = row.get("should_switch")
                should_switch = None
                if isinstance(raw, bool):
                    should_switch = 1 if raw else 0
                conn.execute(
                    """
                    INSERT OR IGNORE INTO annotations(sample_id, user_id, should_switch, updated_at)
                    VALUES (?, 0, ?, ?)
                    """,
                    (sample_id, should_switch, now),
                )
                conn.execute(
                    """
                    UPDATE tasks SET status='done', lease_until=NULL, updated_at=?
                    WHERE sample_id=?
                    """,
                    (now, sample_id),
                )
            conn.commit()

    def _get_user_by_token(self, token: str) -> Dict[str, Any]:
        now = int(time.time())
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT u.id, u.username, u.role, u.is_active
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token = ? AND s.expires_at > ?
                """,
                (token, now),
            ).fetchone()
            if row is None:
                raise HTTPException(status_code=401, detail="invalid or expired token")
            if int(row["is_active"]) != 1:
                raise HTTPException(status_code=403, detail="user is inactive")
            return {"id": int(row["id"]), "username": str(row["username"]), "role": str(row["role"])}

    def auth_user(self, request: Request) -> Dict[str, Any]:
        header = str(request.headers.get("authorization", "")).strip()
        if not header.lower().startswith("bearer "):
            raise HTTPException(status_code=401, detail="missing bearer token")
        token = header.split(" ", maxsplit=1)[1].strip()
        if not token:
            raise HTTPException(status_code=401, detail="missing bearer token")
        return self._get_user_by_token(token)

    def create_session(self, user_id: int) -> str:
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        expires = now + 14 * 24 * 3600
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO sessions(token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (token, int(user_id), expires, now),
            )
            conn.commit()
        return token

    def register_with_invite(self, username: str, password: str, invite_code: str) -> None:
        username = username.strip()
        invite_code = invite_code.strip()
        if not username or not password or not invite_code:
            raise HTTPException(status_code=400, detail="username, password, invite_code are required")
        now = int(time.time())
        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            invite = conn.execute("SELECT * FROM invite_codes WHERE code = ?", (invite_code,)).fetchone()
            if invite is None:
                raise HTTPException(status_code=400, detail="invalid invite code")
            expires_at = invite["expires_at"]
            if expires_at is not None and int(expires_at) <= now:
                raise HTTPException(status_code=400, detail="invite code expired")
            if int(invite["remaining_uses"]) <= 0:
                raise HTTPException(status_code=400, detail="invite code exhausted")
            exists = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
            if exists is not None:
                raise HTTPException(status_code=409, detail="username already exists")
            conn.execute(
                """
                INSERT INTO users(username, password_hash, role, is_active, created_at)
                VALUES (?, ?, ?, 1, ?)
                """,
                (username, hash_password(password), ROLE_ANNOTATOR, now),
            )
            conn.execute(
                "UPDATE invite_codes SET remaining_uses = remaining_uses - 1 WHERE code = ?",
                (invite_code,),
            )
            conn.commit()

    def login(self, username: str, password: str) -> Dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT id, username, password_hash, role, is_active FROM users WHERE username = ?",
                (username.strip(),),
            ).fetchone()
        if row is None or not verify_password(password, str(row["password_hash"])):
            raise HTTPException(status_code=401, detail="invalid username or password")
        if int(row["is_active"]) != 1:
            raise HTTPException(status_code=403, detail="user is inactive")
        token = self.create_session(int(row["id"]))
        return {
            "token": token,
            "user": {
                "id": int(row["id"]),
                "username": str(row["username"]),
                "role": str(row["role"]),
            },
        }

    def _base_status_row(self, conn: sqlite3.Connection) -> Dict[str, int]:
        total = conn.execute("SELECT COUNT(*) AS c FROM tasks").fetchone()
        done = conn.execute("SELECT COUNT(*) AS c FROM tasks WHERE status = 'done'").fetchone()
        return {
            "total": int(total["c"]),
            "labeled": int(done["c"]),
            "remaining": int(total["c"]) - int(done["c"]),
        }

    def status(self, user_id: int | None = None) -> Dict[str, int]:
        now = int(time.time())
        with self._connect() as conn:
            # Lease timeout recovery.
            conn.execute(
                """
                UPDATE tasks
                SET status='pending', assigned_user_id=NULL, lease_until=NULL, updated_at=?
                WHERE status='claimed' AND lease_until IS NOT NULL AND lease_until <= ?
                """,
                (now, now),
            )
            out = self._base_status_row(conn)
            if user_id is not None:
                mine_done = conn.execute(
                    "SELECT COUNT(*) AS c FROM annotations WHERE user_id = ?",
                    (int(user_id),),
                ).fetchone()
                mine_claimed = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM tasks
                    WHERE status='claimed' AND assigned_user_id=? AND lease_until > ?
                    """,
                    (int(user_id), now),
                ).fetchone()
                out["mine_done"] = int(mine_done["c"])
                out["mine_claimed"] = int(mine_claimed["c"])
            conn.commit()
        return out

    def _sample_with_annotation(self, conn: sqlite3.Connection, sample_id: str) -> Dict[str, Any]:
        sample = self.sample_by_id.get(sample_id)
        if sample is None:
            raise HTTPException(status_code=404, detail=f"sample not found: {sample_id}")
        payload = dict(sample)
        row = conn.execute("SELECT should_switch, user_id FROM annotations WHERE sample_id = ?", (sample_id,)).fetchone()
        annotation = None
        if row is not None:
            raw = row["should_switch"]
            if raw is None:
                parsed = None
            else:
                parsed = bool(int(raw))
            annotation = {"sample_id": sample_id, "should_switch": parsed, "user_id": int(row["user_id"])}
        payload["annotation"] = annotation
        return payload

    def claim_next(self, user_id: int) -> Dict[str, Any]:
        now = int(time.time())
        lease_until = now + int(self.config.claim_lease_seconds)
        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                """
                UPDATE tasks
                SET status='pending', assigned_user_id=NULL, lease_until=NULL, updated_at=?
                WHERE status='claimed' AND lease_until IS NOT NULL AND lease_until <= ?
                """,
                (now, now),
            )
            current = conn.execute(
                """
                SELECT t.sample_id
                FROM tasks t
                LEFT JOIN annotations a ON a.sample_id=t.sample_id
                WHERE t.status='claimed' AND t.assigned_user_id=? AND (t.lease_until IS NULL OR t.lease_until > ?)
                  AND a.sample_id IS NULL
                LIMIT 1
                """,
                (int(user_id), now),
            ).fetchone()
            if current is None:
                next_row = conn.execute(
                    "SELECT sample_id FROM tasks WHERE status='pending' ORDER BY sample_id LIMIT 1"
                ).fetchone()
                if next_row is None:
                    conn.commit()
                    return {"sample": None, "status": self._base_status_row(conn)}
                sample_id = str(next_row["sample_id"])
                conn.execute(
                    """
                    UPDATE tasks
                    SET status='claimed', assigned_user_id=?, lease_until=?, updated_at=?
                    WHERE sample_id=?
                    """,
                    (int(user_id), lease_until, now, sample_id),
                )
            else:
                sample_id = str(current["sample_id"])
            sample_payload = self._sample_with_annotation(conn, sample_id)
            conn.commit()
            return {"sample": sample_payload, "status": self._base_status_row(conn)}

    def submit_annotation(self, user: Dict[str, Any], sample_id: str, should_switch: bool | None) -> Dict[str, Any]:
        now = int(time.time())
        with self.lock, self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            task = conn.execute(
                "SELECT status, assigned_user_id FROM tasks WHERE sample_id = ?",
                (sample_id,),
            ).fetchone()
            if task is None:
                raise HTTPException(status_code=404, detail=f"sample not found: {sample_id}")
            assigned = task["assigned_user_id"]
            if task["status"] != "claimed" or assigned is None or int(assigned) != int(user["id"]):
                raise HTTPException(status_code=409, detail="task is not currently assigned to this user")
            value = None if should_switch is None else (1 if bool(should_switch) else 0)
            conn.execute(
                """
                INSERT INTO annotations(sample_id, user_id, should_switch, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(sample_id) DO UPDATE SET
                    user_id=excluded.user_id,
                    should_switch=excluded.should_switch,
                    updated_at=excluded.updated_at
                """,
                (sample_id, int(user["id"]), value, now),
            )
            conn.execute(
                """
                UPDATE tasks
                SET status='done', assigned_user_id=?, lease_until=NULL, updated_at=?
                WHERE sample_id=?
                """,
                (int(user["id"]), now, sample_id),
            )
            sample = self.sample_by_id[sample_id]
            output_payload = {
                "sample_id": sample["sample_id"],
                "episode_key": sample["episode_key"],
                "episode_id": sample["episode_id"],
                "scene_id": sample["scene_id"],
                "pivot_frame": sample["pivot_frame"],
                "rollout_id": sample["rollout_id"],
                "subtask_id": sample["subtask_id"],
                "instruction": sample["instruction"],
                "active_subtask": sample["active_subtask"],
                "next_subtask": sample["next_subtask"],
                "should_switch": should_switch,
                "annotator": user["username"],
                "annotated_at": now,
            }
            append_jsonl(self.config.output_file, output_payload)
            conn.commit()
            return {"annotation": output_payload, "status": self._base_status_row(conn)}

    def create_invite(self, created_by: int, remaining_uses: int, expires_in_days: int | None) -> Dict[str, Any]:
        if remaining_uses <= 0:
            raise HTTPException(status_code=400, detail="remaining_uses must be > 0")
        now = int(time.time())
        expires_at = None
        if expires_in_days is not None:
            expires_at = now + max(1, int(expires_in_days)) * 86400
        code = secrets.token_urlsafe(12).replace("-", "A").replace("_", "B")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO invite_codes(code, remaining_uses, expires_at, created_by_user_id, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (code, int(remaining_uses), expires_at, int(created_by), now),
            )
            conn.commit()
        return {
            "code": code,
            "remaining_uses": int(remaining_uses),
            "expires_at": expires_at,
        }

    def list_users_with_progress(self) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.id,
                    u.username,
                    u.role,
                    u.is_active,
                    COALESCE(a.done_count, 0) AS done_count
                FROM users u
                LEFT JOIN (
                    SELECT user_id, COUNT(*) AS done_count
                    FROM annotations
                    GROUP BY user_id
                ) a ON a.user_id = u.id
                ORDER BY u.role DESC, u.username ASC
                """
            ).fetchall()
        return [
            {
                "id": int(row["id"]),
                "username": str(row["username"]),
                "role": str(row["role"]),
                "is_active": bool(int(row["is_active"])),
                "done_count": int(row["done_count"]),
            }
            for row in rows
        ]


def extract_scene_id(scene_id_or_path: str) -> str:
    if not scene_id_or_path:
        return ""
    parts = scene_id_or_path.split("/")
    if len(parts) >= 2:
        return parts[-2]
    return parts[-1]


def build_episode_key(scene_id: str, episode_id: Any) -> str:
    return f"{scene_id}_{episode_id}"


def normalize_plan_steps(plan: Any) -> List[str]:
    if isinstance(plan, list):
        return [str(step).strip() for step in plan if str(step).strip()]
    if isinstance(plan, str):
        return [line.strip() for line in plan.splitlines() if line.strip()]
    return []


def load_summary_full(path: Path | None) -> Dict[str, Dict[str, Any]]:
    if path is None or not str(path).strip():
        return {}
    if not path.exists():
        raise FileNotFoundError(f"summary_full_path not found: {path}")
    summary: Dict[str, Dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                continue
            episode_key = str(record.get("episode_key", "")).strip()
            if episode_key:
                summary[episode_key] = record
            scene_id = record.get("scene_id")
            episode_id = record.get("episode_id", record.get("id"))
            if scene_id is not None and episode_id is not None:
                summary[build_episode_key(extract_scene_id(str(scene_id)), episode_id)] = record
    return summary


def resolve_summary_record(record: Dict[str, Any], summary_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any] | None:
    episode_key = str(record.get("episode_key", "")).strip()
    if episode_key and episode_key in summary_lookup:
        return summary_lookup[episode_key]
    scene_id = record.get("scene_id")
    episode_id = record.get("episode_id")
    if scene_id is None or episode_id is None:
        return None
    return summary_lookup.get(build_episode_key(extract_scene_id(str(scene_id)), episode_id))


def enrich_record_from_summary(record: Dict[str, Any], summary_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if not summary_lookup:
        return dict(record)
    summary_record = resolve_summary_record(record, summary_lookup)
    if summary_record is None:
        return dict(record)
    enriched = dict(record)
    plan_steps = normalize_plan_steps(summary_record.get("plan", []))
    if plan_steps:
        enriched["plan"] = plan_steps
    if not str(enriched.get("instruction", "")).strip():
        enriched["instruction"] = str(summary_record.get("instruction", ""))
    return enriched


def select_stride_paths(paths: List[str], stride: int) -> List[str]:
    if not paths:
        return []
    stride = max(1, int(stride))
    selected = [paths[idx] for idx in range(0, len(paths), stride)]
    if selected[-1] != paths[-1]:
        selected.append(paths[-1])
    return selected


def split_plan_state(record: Dict[str, Any], plan_steps: List[str]) -> Tuple[List[str], List[str], List[str]]:
    if not plan_steps:
        active_step = str(record.get("subtask_text", "")).strip()
        return [], [active_step] if active_step else [], []
    subtask_id = max(1, int(record.get("subtask_id", 1)))
    index = min(subtask_id - 1, len(plan_steps) - 1)
    return plan_steps[:index], [plan_steps[index]], plan_steps[index + 1 :]


def resolve_bundle_image_path(bundle_root: Path, base_image_path: str, relpath: str) -> Path:
    return bundle_root / base_image_path / relpath


def build_asset_url(relpath: str) -> str:
    rel = relpath.replace("\\", "/").lstrip("/")
    return f"/assets/{rel}"


def prepare_sample(bundle_root: Path, record: Dict[str, Any], image_stride: int) -> Dict[str, Any]:
    plan_steps = normalize_plan_steps(record.get("plan", []))
    done_steps, active_steps, pending_steps = split_plan_state(record, plan_steps)
    active_subtask = active_steps[0] if active_steps else str(record.get("subtask_text", "")).strip()
    next_subtask = pending_steps[0] if pending_steps else "stop"

    base_image_path = str(record["base_image_path"])
    pivot_relpath = f"{base_image_path}/{record['pivot_image_relpath']}".replace("\\", "/")
    pivot_path = resolve_bundle_image_path(bundle_root, base_image_path, str(record["pivot_image_relpath"]))
    rollout_relpaths = [
        f"{base_image_path}/{relpath}".replace("\\", "/")
        for relpath in select_stride_paths([str(path) for path in record.get("rollout_image_relpaths", [])], image_stride)
    ]
    rollout_paths = [
        resolve_bundle_image_path(bundle_root, base_image_path, relpath)
        for relpath in select_stride_paths([str(path) for path in record.get("rollout_image_relpaths", [])], image_stride)
    ]
    if not rollout_paths:
        raise ValueError("rollout_image_relpaths is empty")
    if not pivot_path.exists():
        raise FileNotFoundError(f"pivot image not found: {pivot_path}")
    for path in rollout_paths:
        if not path.exists():
            raise FileNotFoundError(f"rollout frame not found: {path}")

    return {
        "sample_id": str(record.get("sample_id", "")),
        "episode_key": str(record.get("episode_key", "")),
        "episode_id": int(record.get("episode_id", 0)),
        "scene_id": str(record.get("scene_id", "")),
        "pivot_frame": int(record.get("pivot_frame", 0)),
        "rollout_id": int(record.get("rollout_id", 0)),
        "subtask_id": int(record.get("subtask_id", 0)),
        "instruction": str(record.get("instruction", "")),
        "done_steps": done_steps,
        "active_subtask": active_subtask,
        "pending_steps": pending_steps,
        "next_subtask": next_subtask,
        "pivot_image_url": build_asset_url(pivot_relpath),
        "rollout_frame_urls": [build_asset_url(relpath) for relpath in rollout_relpaths],
    }


def build_store(config: ManualSwitchConfig) -> ManualSwitchStore:
    manifest_rows = load_jsonl(config.manifest_file)
    summary_lookup = load_summary_full(config.summary_full_path)
    rows = [enrich_record_from_summary(row, summary_lookup) for row in manifest_rows]
    default_sample = config.max_samples is None
    if config.shuffle or default_sample:
        rng = random.Random(config.seed)
        rng.shuffle(rows)
    if default_sample:
        rows = rows[: max(1, int(len(rows) * 0.1))] if rows else []
    else:
        rows = rows[: max(0, int(config.max_samples))]
    samples = [prepare_sample(config.bundle_root, row, config.image_stride) for row in rows]
    return ManualSwitchStore(config=config, samples=samples)


def create_app(config: ManualSwitchConfig) -> FastAPI:
    app = FastAPI(title=config.title)
    store = build_store(config)
    app.state.store = store
    app.state.config = config
    app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return TEMPLATES.TemplateResponse(
            "index.html",
            {"request": request, "title": config.title, "page_size": int(config.page_size)},
        )

    @app.post("/api/auth/register")
    async def register(payload: Dict[str, Any]) -> Dict[str, Any]:
        store.register_with_invite(
            username=str(payload.get("username", "")),
            password=str(payload.get("password", "")),
            invite_code=str(payload.get("invite_code", "")),
        )
        return {"ok": True}

    @app.post("/api/auth/login")
    async def login(payload: Dict[str, Any]) -> Dict[str, Any]:
        return store.login(username=str(payload.get("username", "")), password=str(payload.get("password", "")))

    @app.get("/api/me")
    async def me(request: Request) -> Dict[str, Any]:
        user = store.auth_user(request)
        return {"user": user, "status": store.status(user_id=user["id"])}

    @app.post("/api/tasks/claim")
    async def claim_task(request: Request) -> Dict[str, Any]:
        user = store.auth_user(request)
        return store.claim_next(user_id=int(user["id"]))

    @app.post("/api/tasks/{sample_id}/annotate")
    async def annotate_task(sample_id: str, payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        user = store.auth_user(request)
        raw = payload.get("should_switch")
        if raw is None:
            should_switch = None
        elif isinstance(raw, bool):
            should_switch = raw
        else:
            raise HTTPException(status_code=400, detail="should_switch must be boolean or null")
        return store.submit_annotation(user=user, sample_id=str(sample_id), should_switch=should_switch)

    @app.post("/api/admin/invites")
    async def create_invite(payload: Dict[str, Any], request: Request) -> Dict[str, Any]:
        user = store.auth_user(request)
        if user["role"] != ROLE_ADMIN:
            raise HTTPException(status_code=403, detail="admin only")
        remaining_uses = int(payload.get("remaining_uses", 1))
        expires_in_days = payload.get("expires_in_days")
        expires_days = None if expires_in_days is None else int(expires_in_days)
        invite = store.create_invite(
            created_by=int(user["id"]),
            remaining_uses=remaining_uses,
            expires_in_days=expires_days,
        )
        return {"invite": invite}

    @app.get("/api/admin/users")
    async def list_users(request: Request) -> Dict[str, Any]:
        user = store.auth_user(request)
        if user["role"] != ROLE_ADMIN:
            raise HTTPException(status_code=403, detail="admin only")
        return {"users": store.list_users_with_progress(), "status": store.status()}

    @app.get("/assets/{asset_path:path}")
    async def get_asset(asset_path: str) -> FileResponse:
        candidate = (config.bundle_root / asset_path).resolve()
        bundle_root = config.bundle_root.resolve()
        if bundle_root != candidate and bundle_root not in candidate.parents:
            raise HTTPException(status_code=404, detail="asset not found")
        if not candidate.exists() or not candidate.is_file():
            raise HTTPException(status_code=404, detail="asset not found")
        return FileResponse(candidate)

    return app
