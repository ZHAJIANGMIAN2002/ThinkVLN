from __future__ import annotations

import math
import json
import random
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from thinkvln.datagen.generation.watcher_utils import append_jsonl, load_jsonl, load_jsonl_by_key


APP_DIR = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(APP_DIR / "templates"))


@dataclass
class ManualSwitchConfig:
    bundle_root: Path
    manifest_file: Path
    output_file: Path
    summary_full_path: Path | None = None
    image_stride: int = 3
    max_samples: int | None = None
    shuffle: bool = False
    seed: int | None = None
    title: str = "Watcher Manual Switch Annotation"
    page_size: int = 20


@dataclass
class ManualSwitchStore:
    config: ManualSwitchConfig
    samples: List[Dict[str, Any]]
    annotations: Dict[str, Dict[str, Any]]
    lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self.sample_by_id = {sample["sample_id"]: sample for sample in self.samples}

    def status(self) -> Dict[str, int]:
        labeled = sum(1 for item in self.annotations.values() if isinstance(item.get("should_switch"), bool))
        total = len(self.samples)
        return {"total": total, "labeled": labeled, "remaining": total - labeled}

    def _annotation_for(self, sample_id: str) -> Dict[str, Any] | None:
        annotation = self.annotations.get(sample_id)
        if not annotation:
            return None
        return {
            "sample_id": sample_id,
            "should_switch": annotation.get("should_switch"),
        }

    def _with_annotation(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        payload = dict(sample)
        payload["annotation"] = self._annotation_for(sample["sample_id"])
        return payload

    def list_samples(self, page: int, page_size: int, view: str) -> Dict[str, Any]:
        page = max(1, int(page))
        page_size = max(1, int(page_size))
        if view == "pending":
            filtered = [
                sample for sample in self.samples
                if not isinstance((self.annotations.get(sample["sample_id"]) or {}).get("should_switch"), bool)
            ]
        else:
            filtered = list(self.samples)
        total_items = len(filtered)
        total_pages = max(1, math.ceil(total_items / page_size)) if total_items else 1
        if total_items and page > total_pages:
            page = total_pages
        start = (page - 1) * page_size
        end = start + page_size
        page_samples = [self._with_annotation(sample) for sample in filtered[start:end]]
        return {
            "page": page,
            "page_size": page_size,
            "total_items": total_items,
            "total_pages": total_pages,
            "view": view,
            "samples": page_samples,
            "status": self.status(),
        }

    def save_annotation(self, sample_id: str, should_switch: bool | None) -> Dict[str, Any]:
        sample = self.sample_by_id.get(sample_id)
        if sample is None:
            raise KeyError(sample_id)
        payload = {
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
        }
        with self.lock:
            append_jsonl(self.config.output_file, payload)
            self.annotations[sample_id] = payload
        return payload


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
    if config.shuffle:
        rng = random.Random(config.seed)
        rng.shuffle(rows)
    if config.max_samples is not None:
        rows = rows[: max(0, int(config.max_samples))]
    samples = [prepare_sample(config.bundle_root, row, config.image_stride) for row in rows]
    annotations = load_jsonl_by_key(config.output_file, "sample_id") if config.output_file.exists() else {}
    return ManualSwitchStore(config=config, samples=samples, annotations=annotations)


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
            {
                "request": request,
                "title": config.title,
                "page_size": int(config.page_size),
                "status": store.status(),
            },
        )

    @app.get("/api/status")
    async def status() -> Dict[str, int]:
        return store.status()

    @app.get("/api/samples")
    async def list_samples(
        page: int = 1,
        page_size: int | None = None,
        view: Literal["pending", "all"] = "pending",
    ) -> Dict[str, Any]:
        size = int(page_size or config.page_size)
        return store.list_samples(page=page, page_size=size, view=view)

    @app.post("/api/annotations")
    async def save_annotation(payload: Dict[str, Any]) -> Dict[str, Any]:
        sample_id = str(payload.get("sample_id", "")).strip()
        if not sample_id:
            raise HTTPException(status_code=400, detail="sample_id is required")
        raw = payload.get("should_switch")
        if raw is None:
            should_switch = None
        elif isinstance(raw, bool):
            should_switch = raw
        else:
            raise HTTPException(status_code=400, detail="should_switch must be boolean or null")
        try:
            annotation = store.save_annotation(sample_id, should_switch)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"sample not found: {sample_id}") from exc
        return {"annotation": annotation, "status": store.status()}

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
