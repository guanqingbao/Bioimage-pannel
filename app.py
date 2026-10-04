from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel


APP_DIR = Path(__file__).resolve().parent
DATA_DIR = Path(
    os.environ.get("IMAGE_BATCH_DATA_DIR", str(APP_DIR / "data"))
).expanduser().resolve()
BATCHES_DIR = DATA_DIR / "_batches"
BATCH_DB_PATH = DATA_DIR / "batches.sqlite3"
_documents_dir = os.environ.get("IMAGE_BATCH_DOCUMENTS_DIR", "").strip()
DOCUMENTS_DIR: Path | None = (
    Path(_documents_dir).expanduser().resolve() if _documents_dir else None
)
_pdf_source_roots = os.environ.get("IMAGE_BATCH_PDF_ROOTS", "").strip()
PDF_SOURCE_ROOTS: tuple[Path, ...] = tuple(
    dict.fromkeys(
        Path(value.strip()).expanduser().resolve()
        for value in _pdf_source_roots.split(os.pathsep)
        if value.strip()
    )
)
STATIC_DIR = APP_DIR / "static"
MAX_UPLOAD_BYTES = 80 * 1024 * 1024
MAX_BATCH_FILES = 2000
DEFAULT_BATCH_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 2))
MAX_BATCH_WORKERS = max(DEFAULT_BATCH_WORKERS, min(32, os.cpu_count() or 4))
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
ALLOWED_SUFFIXES = IMAGE_SUFFIXES | {".pdf"}
EDITOR_STATE_NAME = "editor_state.json"
PANEL_LABEL_RE = re.compile(r"^[A-Z]$")
EDITOR_STATE_LOCK = threading.Lock()
BATCH_STATE_LOCK = threading.Lock()
TRAINING_COLLECTION_LOCK = threading.Lock()
ACTIVE_BATCH_LOCK = threading.Lock()
ACTIVE_BATCH_IDS: set[str] = set()
# Empty in standalone mode.  The BioMat web server sets this to
# ``/image-processing`` when mounting this FastAPI app as a sub-application.
PUBLIC_PATH_PREFIX = ""

from image_batch.services import extract_pdf_figures, split_figure_panels
from image_batch.training_collection import (
    effective_panels,
    load_collection_summary,
    rebuild_training_collection,
    sync_training_record,
)


app = FastAPI(title="Image Batch", version="1.0.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
DATA_DIR.mkdir(parents=True, exist_ok=True)
BATCHES_DIR.mkdir(parents=True, exist_ok=True)


def initialize_batch_database() -> None:
    connection = sqlite3.connect(BATCH_DB_PATH, timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                source_kind TEXT NOT NULL,
                source_label TEXT NOT NULL,
                concurrency INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                updated_at TEXT NOT NULL,
                total INTEGER NOT NULL DEFAULT 0,
                completed_count INTEGER NOT NULL DEFAULT 0,
                failed_count INTEGER NOT NULL DEFAULT 0,
                progress REAL NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS batch_items (
                batch_id TEXT NOT NULL,
                item_id TEXT NOT NULL,
                job_id TEXT NOT NULL UNIQUE,
                filename TEXT NOT NULL,
                source_path TEXT NOT NULL,
                size_bytes INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                metrics_json TEXT NOT NULL DEFAULT '{}',
                message TEXT NOT NULL DEFAULT '',
                error TEXT NOT NULL DEFAULT '',
                PRIMARY KEY (batch_id, item_id),
                FOREIGN KEY (batch_id) REFERENCES batches(batch_id)
            );
            CREATE INDEX IF NOT EXISTS idx_batch_items_status
                ON batch_items(batch_id, status);
            """
        )
        connection.commit()
    finally:
        connection.close()


initialize_batch_database()


def safe_name(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    stem = "".join(char if char.isalnum() or char in "-_" else "_" for char in Path(filename).stem)
    return f"{stem[:80] or 'upload'}{suffix}"


def document_slug(name: str) -> str:
    """Return a stable filesystem segment for one uploaded literature item."""
    slug = Path(safe_name(f"{name}.pdf")).stem or "document"
    # Keep enough headroom for ``panels/<figure>/manual_versions/<label>`` and
    # JSON/image version files on Windows installations without long-path mode.
    if len(slug) <= 48:
        return slug
    digest = hashlib.sha1(str(name).encode("utf-8")).hexdigest()[:8]
    prefix = slug[:39].rstrip(" ._-") or "document"
    return f"{prefix}_{digest}"


def document_job_id(document_name: str) -> str:
    """Return the stable editor-session id for one parsed document."""
    digest = hashlib.sha256(str(document_name).encode("utf-8")).hexdigest()[:32]
    return f"doc{digest}"


def _document_visual_root(job_id: str) -> Path | None:
    documents_root = DOCUMENTS_DIR.resolve() if DOCUMENTS_DIR else None
    if documents_root is None or not documents_root.is_dir() or not job_id.startswith("doc"):
        return None
    for document_dir in documents_root.iterdir():
        if not document_dir.is_dir() or document_job_id(document_dir.name) != job_id:
            continue
        visual_root = (document_dir / "visual").resolve()
        if visual_root.is_relative_to(documents_root) and visual_root.is_dir():
            return visual_root
    return None


def file_url(job_id: str, path: Path) -> str:
    root = job_root(job_id)
    relative = path.resolve().relative_to(root)
    return f"{PUBLIC_PATH_PREFIX}/files/{job_id}/{relative.as_posix()}"


def image_artifact(job_id: str, path: Path, kind: str, label: str) -> dict[str, str]:
    return {"kind": kind, "label": label, "url": file_url(job_id, path)}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A UUID appended to the full destination name can push otherwise valid
    # panel paths beyond the classic Windows MAX_PATH boundary.
    temporary = path.with_name(f".tmp_{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def job_root(job_id: str, *, must_exist: bool = True) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9]{1,64}", job_id):
        raise HTTPException(status_code=404, detail="任务不存在")
    data_root = DATA_DIR.resolve()
    root = (data_root / job_id).resolve()
    if root.is_relative_to(data_root) and (root.is_dir() or not must_exist):
        return root
    document_root = _document_visual_root(job_id)
    if document_root is not None:
        return document_root
    raise HTTPException(status_code=404, detail="任务不存在")


def relative_job_path(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Editor asset is outside the job directory: {resolved}")
    return resolved.relative_to(root).as_posix()


def resolve_job_path(root: Path, relative_path: str, *, must_exist: bool = True) -> Path:
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root) or (must_exist and not path.exists()):
        raise HTTPException(status_code=404, detail="编辑资源不存在")
    return path


def normalize_panel_label(value: str) -> str:
    label = str(value or "").strip().upper()
    if not PANEL_LABEL_RE.fullmatch(label):
        raise HTTPException(status_code=400, detail="面板标签必须是单个 A-Z 字母")
    return label


def image_dimensions(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        normalized = ImageOps.exif_transpose(image)
        return int(normalized.width), int(normalized.height)


def write_panel_version_manifest(root: Path, target: dict[str, Any]) -> None:
    manifest_path = resolve_job_path(
        root,
        target["manifest_path"],
        must_exist=False,
    )
    write_json_atomic(
        manifest_path,
        {
            "label": target["label"],
            "current_version": target["current_version"],
            "current_path": target["current_path"],
            "active": bool(target.get("active", True)),
            "deleted_at": target.get("deleted_at"),
            "versions": target["versions"],
        },
    )


def write_understanding_version_manifest(root: Path, target: dict[str, Any]) -> None:
    """Persist the complete human-written understanding history for one panel."""
    manifest_path = resolve_job_path(
        root,
        target["understanding_manifest_path"],
        must_exist=False,
    )
    write_json_atomic(
        manifest_path,
        {
            "label": target["label"],
            "current_version": int(target.get("understanding_current_version", 0)),
            "current_text": str(target.get("understanding_text", "")),
            "versions": target.get("understanding_versions", []),
        },
    )


def serialize_version(job_id: str, root: Path, version: dict[str, Any]) -> dict[str, Any]:
    path = resolve_job_path(root, version["path"])
    return {
        **version,
        "url": file_url(job_id, path),
    }


def serialize_panel_target(
    job_id: str,
    root: Path,
    target: dict[str, Any],
) -> dict[str, Any]:
    current_path = resolve_job_path(root, target["current_path"])
    current_version = target["current_version"]
    return {
        "label": target["label"],
        "active": bool(target.get("active", True)),
        "current_version": current_version,
        "current_url": f"{file_url(job_id, current_path)}?v={current_version}",
        "versions": [
            serialize_version(job_id, root, version)
            for version in target["versions"]
        ],
        "understanding_text": str(target.get("understanding_text", "")),
        "understanding_current_version": int(
            target.get("understanding_current_version", 0)
        ),
        "understanding_versions": [
            {
                "version": int(version["version"]),
                "created_at": version["created_at"],
                "text": str(version.get("text", "")),
            }
            for version in target.get("understanding_versions", [])
        ],
    }


def serialize_editor_figure(
    job_id: str,
    root: Path,
    figure: dict[str, Any],
) -> dict[str, Any]:
    source_path = resolve_job_path(root, figure["source_path"])
    _ensure_figure_annotation_state(figure)
    return {
        "record_id": figure["record_id"],
        "whole_image_url": file_url(job_id, source_path),
        "whole_image_width_px": figure["source_width_px"],
        "whole_image_height_px": figure["source_height_px"],
        "editable_labels": figure["editable_labels"],
        "review_status": figure["review_status"],
        "annotation_mode": figure["annotation_mode"],
        "annotation_complete": figure["annotation_complete"],
        "annotation_version": figure["annotation_version"],
        "reviewer": figure["reviewer"],
        "reviewed_at": figure["reviewed_at"],
        "panel_targets": [
            serialize_panel_target(job_id, root, figure["panels"][label])
            for label in sorted(figure["panels"])
            if figure["panels"][label].get("active", True)
        ],
        "deleted_labels": [
            label
            for label in sorted(figure["panels"])
            if not figure["panels"][label].get("active", True)
        ],
    }


def _ensure_figure_annotation_state(figure: dict[str, Any]) -> None:
    figure.setdefault("review_status", "proposed")
    if figure.get("annotation_mode") not in {"panel_boxes", "no_split"}:
        figure["annotation_mode"] = "panel_boxes"
    figure.setdefault("annotation_complete", False)
    figure.setdefault("annotation_version", 0)
    figure.setdefault("reviewer", None)
    figure.setdefault("reviewed_at", None)
    for target in (figure.get("panels") or {}).values():
        target.setdefault("active", True)
        target.setdefault("deleted_at", None)


def _invalidate_figure_review(figure: dict[str, Any]) -> None:
    _ensure_figure_annotation_state(figure)
    figure["review_status"] = "proposed"
    figure["annotation_mode"] = "panel_boxes"
    figure["annotation_complete"] = False
    figure["reviewer"] = None
    figure["reviewed_at"] = None
    figure["annotation_version"] = int(figure.get("annotation_version", 0)) + 1


def _sync_training_record_safely(job_id: str, record_id: str) -> dict[str, Any] | None:
    try:
        with TRAINING_COLLECTION_LOCK:
            return sync_training_record(DATA_DIR, BATCH_DB_PATH, job_id, record_id)
    except Exception:
        # The editor state remains authoritative. A full collection rebuild
        # reports collection failures without making an annotation save fail.
        return None


def _figure_review_panels(root: Path, figure: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    source_path = resolve_job_path(root, figure["source_path"])
    width, height = image_dimensions(source_path)
    panel_dir = resolve_job_path(root, figure["panel_dir"])
    layout_path = panel_dir / "layout.json"
    layout: dict[str, Any] = {}
    if layout_path.is_file():
        try:
            layout = json.loads(layout_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            layout = {}
    return effective_panels(figure, layout, width, height)


def initialize_editor_state(
    job_id: str,
    records: list[dict[str, Any]],
    *,
    state_metadata: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Create immutable automatic versions and return editor data per record."""
    root = job_root(job_id)
    created_at = utc_now()
    state: dict[str, Any] = {
        "job_id": job_id,
        "created_at": created_at,
        "updated_at": created_at,
        "figures": {},
    }
    if state_metadata:
        state.update(state_metadata)

    for index, record in enumerate(records, 1):
        record_id = f"r{index:03d}"
        source_path = Path(record["source"]).resolve()
        preview_path = Path(record["preview"]).resolve()
        panel_dir = preview_path.parent
        source_width, source_height = image_dimensions(source_path)
        layout_panels: dict[str, Any] = {}
        layout_path = panel_dir / "layout.json"
        if layout_path.is_file():
            try:
                layout_payload = json.loads(layout_path.read_text(encoding="utf-8"))
                layout_panels = {
                    str(label).upper(): panel
                    for label, panel in (layout_payload.get("panels") or {}).items()
                }
            except (OSError, json.JSONDecodeError):
                layout_panels = {}
        labels: set[str] = set()
        for value in (
            record.get("labels", ""),
            record.get("detected_labels", ""),
            "".join(record.get("figure_constraints", {}).get("expected_labels", [])),
        ):
            labels.update(character.upper() for character in value if character.isalpha())

        figure: dict[str, Any] = {
            "record_id": record_id,
            "document_name": record.get("document_name"),
            "figure_id": record.get("figure_id", source_path.stem),
            "figure_number": record.get("figure_number"),
            "caption_kind": record.get("caption_kind", "figure"),
            "caption": record.get("caption", ""),
            "source_name": source_path.name,
            "source_path": relative_job_path(root, source_path),
            "source_width_px": source_width,
            "source_height_px": source_height,
            "panel_dir": relative_job_path(root, panel_dir),
            "editable_labels": [],
            "panels": {},
            "review_status": "proposed",
            "annotation_mode": "panel_boxes",
            "annotation_complete": False,
            "annotation_version": 0,
            "reviewer": None,
            "reviewed_at": None,
        }

        for panel_path in sorted(panel_dir.glob("panel_*.png")):
            label = panel_path.stem.removeprefix("panel_").upper()
            if not PANEL_LABEL_RE.fullmatch(label):
                continue
            labels.add(label)
            versions_dir = panel_dir / "manual_versions" / label
            versions_dir.mkdir(parents=True, exist_ok=True)
            automatic_path = versions_dir / "v0000_automatic.png"
            shutil.copy2(panel_path, automatic_path)
            width, height = image_dimensions(automatic_path)
            target = {
                "label": label,
                "current_version": 0,
                "current_path": relative_job_path(root, panel_path),
                "manifest_path": relative_job_path(root, versions_dir / "versions.json"),
                "versions": [
                    {
                        "version": 0,
                        "kind": "automatic",
                        "created_at": created_at,
                        "path": relative_job_path(root, automatic_path),
                        "bbox_px": (layout_panels.get(label) or {}).get("rect"),
                        "width_px": width,
                        "height_px": height,
                    }
                ],
                "active": True,
                "deleted_at": None,
                "understanding_current_version": 0,
                "understanding_text": "",
                "understanding_manifest_path": relative_job_path(
                    root,
                    versions_dir / "understanding_versions.json",
                ),
                "understanding_versions": [],
            }
            figure["panels"][label] = target
            write_panel_version_manifest(root, target)
            write_understanding_version_manifest(root, target)

        figure["editable_labels"] = sorted(
            label for label in labels if PANEL_LABEL_RE.fullmatch(label)
        )
        state["figures"][record_id] = figure

    write_json_atomic(root / EDITOR_STATE_NAME, state)
    return {
        record_id: serialize_editor_figure(job_id, root, figure)
        for record_id, figure in state["figures"].items()
    }


def load_editor_state(job_id: str) -> tuple[Path, dict[str, Any]]:
    root = job_root(job_id)
    state_path = root / EDITOR_STATE_NAME
    if not state_path.is_file():
        raise HTTPException(status_code=404, detail="该任务没有可编辑结果")
    try:
        return root, json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="版本状态文件损坏") from exc


class ManualPanelCropRequest(BaseModel):
    label: str
    x0: int
    y0: int
    x1: int
    y1: int
    expected_current_version: int | None = None
    confirmed: bool = False


class PanelUnderstandingRequest(BaseModel):
    label: str
    text: str = ""
    expected_current_version: int = 0
    confirmed: bool = False


class PanelDeleteRequest(BaseModel):
    label: str
    expected_current_version: int
    confirmed: bool = False


class FigureReviewRequest(BaseModel):
    status: str
    annotation_mode: str | None = None
    reviewer: str = ""
    confirmed: bool = False


class DocumentVisualOpenRequest(BaseModel):
    document_name: str


class FolderBatchRequest(BaseModel):
    folder: str
    recursive: bool = False
    concurrency: int = DEFAULT_BATCH_WORKERS


def _document_manifest_records(
    document_dir: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Translate a document-owned visual manifest into editor records."""
    visual_root = (document_dir / "visual").resolve()

    def visual_path(value: Any, fallback: Path | None = None) -> Path:
        text = str(value or "").replace("\\", "/").strip()
        if text:
            path = (document_dir / text).resolve()
        elif fallback is not None:
            path = fallback.resolve()
        else:
            raise HTTPException(status_code=422, detail="视觉清单缺少图片路径")
        if not path.is_relative_to(visual_root):
            raise HTTPException(status_code=422, detail="视觉清单包含越界路径")
        return path

    records: list[dict[str, Any]] = []
    for figure in manifest.get("figures") or []:
        if not isinstance(figure, dict):
            continue
        figure_id = str(figure.get("figure_id") or "").strip()
        if not figure_id:
            continue
        source_path = visual_path(figure.get("whole_path"))
        if not source_path.is_file():
            continue
        panel_dir = visual_root / "panels" / figure_id
        preview_path = visual_path(
            figure.get("preview_path"),
            panel_dir / "preview.png",
        )
        panels = [
            panel
            for panel in (figure.get("panels") or [])
            if isinstance(panel, dict) and panel.get("label")
        ]
        labels = [
            str(panel.get("label") or "").strip().upper()
            for panel in panels
            if PANEL_LABEL_RE.fullmatch(
                str(panel.get("label") or "").strip().upper()
            )
        ]
        split = figure.get("split") if isinstance(figure.get("split"), dict) else {}
        low_confidence = [
            str(panel.get("label") or "").strip().upper()
            for panel in panels
            if panel.get("needs_review")
        ]
        low_confidence.extend(
            str(value).strip().upper()
            for value in (split.get("low_confidence_panels") or [])
        )
        expected = [
            str(value).strip().upper()
            for value in (figure.get("expected_panel_labels") or [])
            if PANEL_LABEL_RE.fullmatch(str(value).strip().upper())
        ]
        records.append(
            {
                "source": str(source_path),
                "preview": str(preview_path),
                "labels": "".join(dict.fromkeys(labels)),
                "detected_labels": "".join(dict.fromkeys(labels)),
                "panel_count": len(labels),
                "mean_confidence": float(split.get("mean_confidence") or 0.0),
                "low_confidence_panels": sorted(set(low_confidence)),
                "split_decision": {
                    "classification": str(split.get("decision") or "unknown"),
                    "reasons": list(split.get("reasons") or []),
                },
                "figure_constraints": {"expected_labels": expected},
                "document_name": document_dir.name,
                "document_slug": document_slug(document_dir.name),
                "figure_id": figure_id,
                "figure_number": figure.get("figure_number"),
                "caption_kind": figure.get("caption_kind") or "figure",
                "caption": str(figure.get("caption") or ""),
                "page_number": figure.get("page_number"),
                "extraction_method": figure.get("extraction_method") or "unknown",
                "extraction_quality_flags": list(figure.get("quality_flags") or []),
                "extraction_needs_review": bool(figure.get("needs_review", False)),
            }
        )
    return records


def _document_editor_records(
    job_id: str,
    document_name: str,
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    root = job_root(job_id)
    state_path = root / EDITOR_STATE_NAME
    if not state_path.is_file():
        return initialize_editor_state(
            job_id,
            records,
            state_metadata={
                "scope": "document",
                "document_name": document_name,
                "manifest_path": "manifest.json",
            },
        )

    _, state = load_editor_state(job_id)
    stored = state.get("figures", {})
    result: dict[str, dict[str, Any]] = {}
    changed = False
    for index, record in enumerate(records, 1):
        record_id = f"r{index:03d}"
        figure = stored.get(record_id)
        if not figure or str(figure.get("figure_id") or "") != str(record.get("figure_id") or ""):
            raise HTTPException(
                status_code=409,
                detail="视觉结果已更新，请先备份人工版本后重新初始化编辑状态",
            )
        metadata = {
            "document_name": document_name,
            "figure_number": record.get("figure_number"),
            "caption_kind": record.get("caption_kind", "figure"),
            "caption": record.get("caption", ""),
        }
        if any(figure.get(key) != value for key, value in metadata.items()):
            figure.update(metadata)
            changed = True
        panel_dir = resolve_job_path(root, figure["panel_dir"])
        targets = figure.setdefault("panels", {})
        for label, target in list(targets.items()):
            current_path = resolve_job_path(
                root,
                target["current_path"],
                must_exist=False,
            )
            if current_path.is_file():
                continue
            if int(target.get("current_version") or 0) <= 0:
                targets.pop(label)
                changed = True
                continue
            current_version = int(target["current_version"])
            version = next(
                (
                    item
                    for item in (target.get("versions") or [])
                    if int(item.get("version") or 0) == current_version
                ),
                None,
            )
            if isinstance(version, dict):
                version_path = resolve_job_path(root, version["path"])
                current_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(version_path, current_path)
                changed = True

        for panel_path in sorted(panel_dir.glob("panel_*.png")):
            label = panel_path.stem.removeprefix("panel_").upper()
            if not PANEL_LABEL_RE.fullmatch(label) or label in targets:
                continue
            versions_dir = panel_dir / "manual_versions" / label
            versions_dir.mkdir(parents=True, exist_ok=True)
            automatic_path = versions_dir / "v0000_automatic.png"
            if not automatic_path.is_file():
                shutil.copy2(panel_path, automatic_path)
            width, height = image_dimensions(automatic_path)
            target = {
                "label": label,
                "current_version": 0,
                "current_path": relative_job_path(root, panel_path),
                "manifest_path": relative_job_path(root, versions_dir / "versions.json"),
                "versions": [
                    {
                        "version": 0,
                        "kind": "automatic",
                        "created_at": utc_now(),
                        "path": relative_job_path(root, automatic_path),
                        "bbox_px": None,
                        "width_px": width,
                        "height_px": height,
                    }
                ],
                "understanding_current_version": 0,
                "understanding_text": "",
                "understanding_manifest_path": relative_job_path(
                    root,
                    versions_dir / "understanding_versions.json",
                ),
                "understanding_versions": [],
            }
            targets[label] = target
            write_panel_version_manifest(root, target)
            write_understanding_version_manifest(root, target)
            changed = True

        labels = set(figure.get("editable_labels") or [])
        labels.update(targets)
        labels.update(
            label
            for label in record.get("figure_constraints", {}).get("expected_labels", [])
            if PANEL_LABEL_RE.fullmatch(str(label).upper())
        )
        editable_labels = sorted(str(label).upper() for label in labels)
        if figure.get("editable_labels") != editable_labels:
            figure["editable_labels"] = editable_labels
            changed = True
        result[record_id] = serialize_editor_figure(job_id, root, figure)
    if changed:
        state["updated_at"] = utc_now()
        write_json_atomic(root / EDITOR_STATE_NAME, state)
    return result


def _document_visual_response(document_dir: Path) -> dict[str, Any]:
    manifest_path = document_dir / "visual" / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="该文献尚未生成视觉结果") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="文献视觉清单损坏") from exc

    records = _document_manifest_records(document_dir, manifest)
    if not records:
        raise HTTPException(status_code=404, detail="该文献没有可编辑的整图")
    job_id = document_job_id(document_dir.name)
    editor_records = _document_editor_records(job_id, document_dir.name, records)

    response_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        editor = editor_records[f"r{index:03d}"]
        panel_targets = editor.get("panel_targets") or []
        response_records.append(
            {
                "source": Path(record["source"]).name,
                **{key: value for key, value in record.items() if key not in {"source", "preview", "figure_constraints", "split_decision"}},
                "expected_labels": "".join(record["figure_constraints"]["expected_labels"]),
                "decision": record["split_decision"]["classification"],
                "reasons": record["split_decision"]["reasons"],
                **editor,
                "labels": (
                    ""
                    if editor.get("annotation_mode") == "no_split"
                    else "".join(target["label"] for target in panel_targets)
                ),
                "panel_count": (
                    0 if editor.get("annotation_mode") == "no_split" else len(panel_targets)
                ),
            }
        )

    total_panels = sum(record["panel_count"] for record in response_records)
    low_confidence = sum(len(record["low_confidence_panels"]) for record in records)
    extraction_review = sum(bool(record["extraction_needs_review"]) for record in records)
    mean_confidence = sum(record["mean_confidence"] for record in records) / len(records)
    return {
        "job_id": job_id,
        "scope": "document",
        "document_name": document_dir.name,
        "message": (
            f"已载入文献视觉结果：{len(records)} 张整图、{total_panels} 个当前子图。"
            "人工框选和图片理解将直接保存到该文献的 visual 目录。"
        ),
        "metrics": {
            "figures": len(records),
            "processed": len(records),
            "panels": total_panels,
            "preserved": sum(record["panel_count"] == 0 for record in response_records),
            "mean_confidence": round(mean_confidence, 3),
            "low_confidence": low_confidence,
            "extraction_review": extraction_review,
            "needs_review": low_confidence + extraction_review,
        },
        "artifacts": [],
        "documents": [
            {
                "name": document_dir.name,
                "slug": document_slug(document_dir.name),
                "record_ids": [record["record_id"] for record in response_records],
                "figure_count": len(records),
                "panel_count": total_panels,
                "needs_review": low_confidence + extraction_review,
            }
        ],
        "records": response_records,
    }


def _sync_document_visual_manifest(
    root: Path,
    state: dict[str, Any],
    figure: dict[str, Any],
    target: dict[str, Any],
) -> None:
    """Publish the current manual panel and understanding back to visual/manifest.json."""
    if state.get("scope") != "document":
        return
    manifest_path = root / str(state.get("manifest_path") or "manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="无法同步文献视觉清单") from exc

    figure_id = str(figure.get("figure_id") or "")
    manifest_figure = next(
        (
            item
            for item in (manifest.get("figures") or [])
            if isinstance(item, dict) and str(item.get("figure_id") or "") == figure_id
        ),
        None,
    )
    if manifest_figure is None:
        raise HTTPException(status_code=409, detail="视觉清单中的整图记录已经变化")

    label = str(target["label"])
    panels = manifest_figure.setdefault("panels", [])
    panel = next(
        (
            item
            for item in panels
            if isinstance(item, dict) and str(item.get("label") or "").upper() == label
        ),
        None,
    )
    if panel is None:
        panel = {"label": label}
        panels.append(panel)
    current_version = int(target.get("current_version") or 0)
    current_entry = next(
        (
            item
            for item in (target.get("versions") or [])
            if int(item.get("version") or 0) == current_version
        ),
        {},
    )
    panel.update(
        {
            "label": label,
            "path": f"visual/{target['current_path']}",
            "bbox_px": current_entry.get("bbox_px"),
            "confidence": 1.0,
            "needs_review": False,
            "source": "manual" if current_version else "automatic",
            "current_version": current_version,
            "understanding": str(target.get("understanding_text") or ""),
            "understanding_current_version": int(
                target.get("understanding_current_version") or 0
            ),
            "updated_at": state.get("updated_at") or utc_now(),
        }
    )
    panels.sort(key=lambda item: str(item.get("label") or ""))
    manifest_figure["detected_panel_labels"] = [
        str(item.get("label") or "") for item in panels if item.get("label")
    ]
    manifest_figure["manual_revision_count"] = sum(
        int(item.get("current_version") or 0) > 0 for item in panels
    )
    metrics = manifest.setdefault("metrics", {})
    metrics["panels"] = sum(
        len(item.get("panels") or [])
        for item in (manifest.get("figures") or [])
        if isinstance(item, dict)
    )
    manifest["manual_updated_at"] = state.get("updated_at") or utc_now()
    write_json_atomic(manifest_path, manifest)


def run_pipeline(
    job_id: str,
    upload_path: Path,
    original_filename: str | None = None,
) -> dict:
    job_dir = DATA_DIR / job_id
    output_dir = job_dir / "results"
    artifacts: list[dict[str, str]] = []
    overview_artifacts: list[dict[str, str]] = []
    records: list[dict] = []
    extraction_review_items: list[dict] = []
    figure_count = 1
    uploaded_document_name = Path(original_filename or upload_path.name).stem

    if upload_path.suffix.lower() == ".pdf":
        extraction = json.loads(
            extract_pdf_figures(str(upload_path), str(output_dir / "extracted"))
        )
        figure_count = sum(article["figure_count"] for article in extraction["articles"])
        for article_index, article in enumerate(extraction["articles"], 1):
            literature_name = (
                uploaded_document_name
                if len(extraction["articles"]) == 1
                else Path(article["pdf_path"]).stem
            )
            literature_slug = document_slug(literature_name)
            extraction_review_items.extend(article.get("review_items", []))
            split_dir = output_dir / "documents" / literature_slug / "panels"
            split_result = json.loads(
                split_figure_panels(article["figures_dir"], str(split_dir))
            )
            figure_quality = article.get("figure_quality", {})
            for record in split_result["results"]:
                quality = figure_quality.get(Path(record["source"]).name, {})
                record["document_name"] = literature_name
                record["document_slug"] = literature_slug
                record["figure_id"] = quality.get("figure_id", Path(record["source"]).stem)
                record["figure_number"] = quality.get("figure_number")
                record["caption_kind"] = quality.get("caption_kind") or "figure"
                record["caption"] = quality.get("caption", "")
                record["page_number"] = quality.get("page_number")
                record["extraction_method"] = quality.get("extraction_method", "unknown")
                record["extraction_quality_flags"] = quality.get("quality_flags", [])
                record["extraction_needs_review"] = bool(quality.get("needs_review", False))
            records.extend(split_result["results"])
            overview = Path(split_result["overview_path"])
            if overview.is_file():
                overview_artifacts.append(
                    image_artifact(job_id, overview, "overview", f"{literature_name} · 总览")
                )
    else:
        literature_name = uploaded_document_name
        literature_slug = document_slug(literature_name)
        split_result = json.loads(
            split_figure_panels(
                str(upload_path),
                str(output_dir / "documents" / literature_slug / "panels"),
            )
        )
        records = split_result["results"]
        for record in records:
            record.update(
                document_name=literature_name,
                document_slug=literature_slug,
                figure_id=Path(record["source"]).stem,
                figure_number=None,
                caption_kind="image",
                caption="",
                page_number=None,
            )

    for index, record in enumerate(records, 1):
        preview = Path(record["preview"])
        if preview.is_file():
            if record.get("extraction_needs_review", False):
                artifacts.append(
                    image_artifact(job_id, preview, "review", f"提取复核 {index}")
                )
            elif record["panel_count"]:
                artifacts.append(
                    image_artifact(job_id, preview, "preview", f"切分预览 {index}")
                )
            else:
                artifacts.append(
                    image_artifact(job_id, preview, "preserved", f"提取整图 {index}（未切分）")
                )
        for panel_path in sorted(preview.parent.glob("panel_*.png")):
            label = panel_path.stem.removeprefix("panel_")
            artifacts.append(image_artifact(job_id, panel_path, "panel", f"面板 {label}"))

    artifacts.extend(overview_artifacts)

    total_panels = sum(record["panel_count"] for record in records)
    preserved_figures = sum(record["panel_count"] == 0 for record in records)
    low_confidence = sum(len(record["low_confidence_panels"]) for record in records)
    mean_confidence = (
        sum(record["mean_confidence"] for record in records) / len(records)
        if records
        else 0.0
    )
    extraction_review = len(extraction_review_items)
    needs_review = extraction_review + low_confidence
    review_suffix = (
        f"，{extraction_review} 个整图提取项需要复核"
        if extraction_review
        else ""
    )
    editor_records = initialize_editor_state(job_id, records)
    response_records: list[dict[str, Any]] = []
    for index, record in enumerate(records, 1):
        editor = editor_records[f"r{index:03d}"]
        response_records.append(
            {
                "source": Path(record["source"]).name,
                "document_name": record.get("document_name", upload_path.stem),
                "document_slug": record.get("document_slug", document_slug(upload_path.stem)),
                "figure_id": record.get("figure_id", Path(record["source"]).stem),
                "figure_number": record.get("figure_number"),
                "caption_kind": record.get("caption_kind", "figure"),
                "caption": record.get("caption", ""),
                "page_number": record.get("page_number"),
                "labels": record["labels"],
                "detected_labels": record.get("detected_labels", ""),
                "panel_count": record["panel_count"],
                "mean_confidence": round(record["mean_confidence"], 3),
                "low_confidence_panels": record["low_confidence_panels"],
                "expected_labels": "".join(
                    record.get("figure_constraints", {}).get("expected_labels", [])
                ),
                "decision": record.get("split_decision", {}).get(
                    "classification", "unknown"
                ),
                "reasons": record.get("split_decision", {}).get("reasons", []),
                "extraction_method": record.get("extraction_method", "direct_upload"),
                "extraction_quality_flags": record.get("extraction_quality_flags", []),
                "extraction_needs_review": record.get("extraction_needs_review", False),
                **editor,
            }
        )
    documents: list[dict[str, Any]] = []
    documents_by_name: dict[str, dict[str, Any]] = {}
    for record in response_records:
        name = record["document_name"]
        document = documents_by_name.get(name)
        if document is None:
            document = {
                "name": name,
                "slug": record["document_slug"],
                "record_ids": [],
                "figure_count": 0,
                "panel_count": 0,
                "needs_review": 0,
            }
            documents_by_name[name] = document
            documents.append(document)
        document["record_ids"].append(record["record_id"])
        document["figure_count"] += 1
        document["panel_count"] += record["panel_count"]
        document["needs_review"] += int(
            record["extraction_needs_review"]
            or bool(record["low_confidence_panels"])
        )
    return {
        "job_id": job_id,
        "message": (
            f"处理完成：识别到 {figure_count} 张整图，切分出 {total_panels} 个面板，"
            f"{preserved_figures} 张因证据不足保留整图{review_suffix}。"
            if total_panels
            else (
                f"处理完成：{preserved_figures} 张图片因切分证据不足保留整图"
                f"{review_suffix}，"
                "未执行强制切分。"
            )
        ),
        "metrics": {
            "figures": figure_count,
            "processed": len(records),
            "panels": total_panels,
            "preserved": preserved_figures,
            "mean_confidence": round(mean_confidence, 3),
            "low_confidence": low_confidence,
            "extraction_review": extraction_review,
            "needs_review": needs_review,
        },
        "artifacts": artifacts,
        "documents": documents,
        "records": response_records,
    }


@contextmanager
def _batch_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(BATCH_DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA busy_timeout=30000")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def _validate_batch_id(batch_id: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{32}", batch_id):
        raise HTTPException(status_code=404, detail="批次不存在")


def _decode_batch_item(row: sqlite3.Row) -> dict[str, Any]:
    payload = dict(row)
    try:
        payload["metrics"] = json.loads(payload.pop("metrics_json") or "{}")
    except json.JSONDecodeError:
        payload["metrics"] = {}
    return payload


def _matching_pdf_source_root(path: Path) -> Path | None:
    resolved = path.expanduser().resolve()
    for root in PDF_SOURCE_ROOTS:
        try:
            resolved.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def _add_batch_item_source_group(
    state: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, Any]:
    if state.get("source_kind") != "folder":
        item["source_group"] = "浏览器上传"
        item["source_relative_path"] = item.get("filename", "")
        return item

    source_path = Path(str(item.get("source_path") or "")).expanduser().resolve()
    source_root = Path(str(state.get("source_label") or "")).expanduser().resolve()
    try:
        relative_path = source_path.relative_to(source_root)
        relative_parent = relative_path.parent.as_posix()
        item["source_group"] = "根目录" if relative_parent == "." else relative_parent
        item["source_relative_path"] = relative_path.as_posix()
    except ValueError:
        item["source_group"] = source_path.parent.name or "其他目录"
        item["source_relative_path"] = source_path.name
    return item


def _job_review_summary(job_id: str) -> dict[str, Any]:
    """Return lightweight annotation progress for the history navigation."""
    summary = {
        "figure_count": 0,
        "verified_count": 0,
        "ambiguous_count": 0,
        "review_state": "unreviewed",
    }
    try:
        root = job_root(job_id)
        state_path = root / EDITOR_STATE_NAME
        if not state_path.is_file():
            return summary
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (HTTPException, OSError, json.JSONDecodeError):
        return summary

    figures = list((state.get("figures") or {}).values())
    verified = 0
    ambiguous = 0
    for figure in figures:
        _ensure_figure_annotation_state(figure)
        if figure["review_status"] == "verified" and figure["annotation_complete"]:
            verified += 1
        elif figure["review_status"] == "ambiguous":
            ambiguous += 1

    total = len(figures)
    if total > 0 and verified == total:
        review_state = "verified"
    elif verified > 0:
        review_state = "partial"
    elif ambiguous > 0:
        review_state = "ambiguous"
    else:
        review_state = "unreviewed"
    return {
        "figure_count": total,
        "verified_count": verified,
        "ambiguous_count": ambiguous,
        "review_state": review_state,
    }


def _batch_counts(connection: sqlite3.Connection, batch_id: str) -> dict[str, int]:
    counts = {name: 0 for name in ("queued", "processing", "completed", "failed")}
    for row in connection.execute(
        "SELECT status, COUNT(*) AS count FROM batch_items WHERE batch_id = ? GROUP BY status",
        (batch_id,),
    ):
        counts[str(row["status"])] = int(row["count"])
    return counts


def _refresh_batch_metrics_db(connection: sqlite3.Connection, batch_id: str) -> dict[str, int]:
    counts = _batch_counts(connection, batch_id)
    total = sum(counts.values())
    completed_count = counts.get("completed", 0)
    failed_count = counts.get("failed", 0)
    progress = (completed_count + failed_count) / max(1, total)
    connection.execute(
        """
        UPDATE batches
        SET total = ?, completed_count = ?, failed_count = ?, progress = ?, updated_at = ?
        WHERE batch_id = ?
        """,
        (total, completed_count, failed_count, progress, utc_now(), batch_id),
    )
    return counts


def load_batch_state(
    batch_id: str,
    *,
    offset: int = 0,
    limit: int = 200,
) -> dict[str, Any]:
    _validate_batch_id(batch_id)
    with _batch_connection() as connection:
        row = connection.execute(
            "SELECT * FROM batches WHERE batch_id = ?",
            (batch_id,),
        ).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="批次不存在")
        state = dict(row)
        state["counts"] = _batch_counts(connection, batch_id)
        start = max(0, offset)
        size = max(1, min(limit, 1000))
        rows = connection.execute(
            """
            SELECT * FROM batch_items
            WHERE batch_id = ?
            ORDER BY item_id
            LIMIT ? OFFSET ?
            """,
            (batch_id, size, start),
        ).fetchall()
        state["items"] = [
            _add_batch_item_source_group(state, _decode_batch_item(item))
            for item in rows
        ]
        state["items_offset"] = start
        state["items_returned"] = len(rows)
        state["items_total"] = int(state.get("total") or 0)
        return state


def load_batch_item(batch_id: str, item_id: str) -> dict[str, Any] | None:
    with _batch_connection() as connection:
        row = connection.execute(
            "SELECT * FROM batch_items WHERE batch_id = ? AND item_id = ?",
            (batch_id, item_id),
        ).fetchone()
    return _decode_batch_item(row) if row is not None else None


def update_batch_item(batch_id: str, item_id: str, **changes: Any) -> None:
    allowed = {
        "status",
        "started_at",
        "completed_at",
        "message",
        "error",
        "metrics",
    }
    assignments: list[str] = []
    values: list[Any] = []
    for key, value in changes.items():
        if key not in allowed:
            continue
        column = "metrics_json" if key == "metrics" else key
        assignments.append(f"{column} = ?")
        values.append(json.dumps(value, ensure_ascii=False) if key == "metrics" else value)
    if not assignments:
        return
    with BATCH_STATE_LOCK, _batch_connection() as connection:
        values.extend([batch_id, item_id])
        cursor = connection.execute(
            f"UPDATE batch_items SET {', '.join(assignments)} WHERE batch_id = ? AND item_id = ?",
            values,
        )
        if cursor.rowcount != 1:
            raise RuntimeError(f"Batch item not found: {item_id}")
        _refresh_batch_metrics_db(connection, batch_id)


def _process_batch_item(batch_id: str, item_id: str) -> None:
    item = load_batch_item(batch_id, item_id)
    if item is None:
        return
    job_id = str(item["job_id"])
    source_path = Path(item["source_path"])
    update_batch_item(
        batch_id,
        item_id,
        status="processing",
        started_at=utc_now(),
        error="",
    )
    try:
        result = run_pipeline(
            job_id,
            source_path,
            str(item.get("filename") or source_path.name),
        )
        write_json_atomic(DATA_DIR / job_id / "job_result.json", result)
        for record in result.get("records", []):
            record_id = str(record.get("record_id") or "")
            if record_id:
                _sync_training_record_safely(job_id, record_id)
        update_batch_item(
            batch_id,
            item_id,
            status="completed",
            completed_at=utc_now(),
            message=result.get("message", ""),
            metrics=result.get("metrics", {}),
        )
    except Exception as exc:
        update_batch_item(
            batch_id,
            item_id,
            status="failed",
            completed_at=utc_now(),
            error=str(exc),
        )


def _execute_batch(batch_id: str, item_ids: list[str], concurrency: int) -> None:
    with _batch_connection() as connection:
        connection.execute(
            "UPDATE batches SET status = 'running', started_at = COALESCE(started_at, ?), updated_at = ? WHERE batch_id = ?",
            (utc_now(), utc_now(), batch_id),
        )
    worker_count = max(1, min(concurrency, MAX_BATCH_WORKERS, len(item_ids)))
    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="image-batch") as pool:
        futures = [pool.submit(_process_batch_item, batch_id, item_id) for item_id in item_ids]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception:
                pass
    with BATCH_STATE_LOCK, _batch_connection() as connection:
        counts = _refresh_batch_metrics_db(connection, batch_id)
        status = "completed" if not counts.get("failed") else "completed_with_errors"
        connection.execute(
            "UPDATE batches SET status = ?, completed_at = ?, updated_at = ? WHERE batch_id = ?",
            (status, utc_now(), utc_now(), batch_id),
        )


def _run_batch_thread(batch_id: str, item_ids: list[str], concurrency: int) -> None:
    try:
        _execute_batch(batch_id, item_ids, concurrency)
    finally:
        with ACTIVE_BATCH_LOCK:
            ACTIVE_BATCH_IDS.discard(batch_id)


def _start_batch(state: dict[str, Any]) -> None:
    batch_id = str(state["batch_id"])
    with ACTIVE_BATCH_LOCK:
        if batch_id in ACTIVE_BATCH_IDS:
            return
        ACTIVE_BATCH_IDS.add(batch_id)
    with _batch_connection() as connection:
        item_ids = [
            str(row["item_id"])
            for row in connection.execute(
                "SELECT item_id FROM batch_items WHERE batch_id = ? AND status = 'queued' ORDER BY item_id",
                (batch_id,),
            )
        ]
    thread = threading.Thread(
        target=_run_batch_thread,
        args=(batch_id, item_ids, int(state["concurrency"])),
        name=f"batch-{batch_id[:8]}",
        daemon=True,
    )
    try:
        thread.start()
    except Exception:
        with ACTIVE_BATCH_LOCK:
            ACTIVE_BATCH_IDS.discard(batch_id)
        raise


def create_batch_state(
    items: list[dict[str, Any]],
    *,
    concurrency: int,
    source_kind: str,
    source_label: str,
) -> dict[str, Any]:
    batch_id = uuid.uuid4().hex
    created_at = utc_now()
    worker_count = max(1, min(concurrency, MAX_BATCH_WORKERS))
    with _batch_connection() as connection:
        connection.execute(
            """
            INSERT INTO batches (
                batch_id, status, source_kind, source_label, concurrency,
                created_at, updated_at, total
            ) VALUES (?, 'queued', ?, ?, ?, ?, ?, ?)
            """,
            (batch_id, source_kind, source_label, worker_count, created_at, created_at, len(items)),
        )
        connection.executemany(
            """
            INSERT INTO batch_items (
                batch_id, item_id, job_id, filename, source_path, size_bytes,
                status, created_at, metrics_json, message, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    batch_id,
                    item["item_id"],
                    item["job_id"],
                    item["filename"],
                    item["source_path"],
                    int(item.get("size_bytes") or 0),
                    "queued",
                    item.get("created_at") or created_at,
                    "{}",
                    "",
                    "",
                )
                for item in items
            ],
        )
        _refresh_batch_metrics_db(connection, batch_id)
    return load_batch_state(batch_id)


def recover_interrupted_batches() -> int:
    """Put interrupted work back in the queue and resume it after a restart."""
    with BATCH_STATE_LOCK, _batch_connection() as connection:
        rows = connection.execute(
            "SELECT batch_id FROM batches WHERE status IN ('queued', 'running') ORDER BY created_at"
        ).fetchall()
        batch_ids = [str(row["batch_id"]) for row in rows]
        for batch_id in batch_ids:
            connection.execute(
                """
                UPDATE batch_items
                SET status = 'queued', started_at = NULL, completed_at = NULL,
                    error = CASE WHEN status = 'processing' THEN '服务重启后自动恢复' ELSE error END
                WHERE batch_id = ? AND status = 'processing'
                """,
                (batch_id,),
            )
            connection.execute(
                "UPDATE batches SET status = 'queued', completed_at = NULL, updated_at = ? WHERE batch_id = ?",
                (utc_now(), batch_id),
            )
            _refresh_batch_metrics_db(connection, batch_id)
    for batch_id in batch_ids:
        _start_batch(load_batch_state(batch_id, limit=1))
    return len(batch_ids)


@app.on_event("startup")
def resume_batches_after_restart() -> None:
    recover_interrupted_batches()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(
        STATIC_DIR / "index.html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "image-batch",
        "dpi": 300,
        "default_workers": DEFAULT_BATCH_WORKERS,
        "max_workers": MAX_BATCH_WORKERS,
        "pdf_source_roots": [str(root) for root in PDF_SOURCE_ROOTS],
    }


@app.get("/api/server-folders")
def list_server_folders(path: str = "") -> dict[str, Any]:
    """List folders below explicitly configured PDF source roots."""
    roots = [
        {
            "name": root.name or str(root),
            "path": str(root),
            "available": root.is_dir(),
        }
        for root in PDF_SOURCE_ROOTS
    ]
    available_roots = [root for root in PDF_SOURCE_ROOTS if root.is_dir()]
    if not PDF_SOURCE_ROOTS:
        return {
            "configured": False,
            "roots": [],
            "current": None,
            "parent": None,
            "directories": [],
            "pdf_count": 0,
            "message": "尚未配置 IMAGE_BATCH_PDF_ROOTS，可继续手动输入服务器路径。",
        }
    if not available_roots:
        return {
            "configured": True,
            "roots": roots,
            "current": None,
            "parent": None,
            "directories": [],
            "pdf_count": 0,
            "message": "已配置的 PDF 根目录均不存在或不可访问。",
        }

    current = Path(path).expanduser().resolve() if path.strip() else available_roots[0]
    matching_root = _matching_pdf_source_root(current)
    if matching_root is None:
        raise HTTPException(status_code=403, detail="只能浏览已配置的 PDF 根目录")
    if not current.is_dir():
        raise HTTPException(status_code=404, detail=f"服务器目录不存在：{current}")

    try:
        children = list(current.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=f"没有权限读取目录：{current}") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=f"目录读取失败：{current}") from exc

    directory_entries: list[dict[str, Any]] = []
    direct_pdf_count = 0
    for child in children:
        try:
            resolved_child = child.resolve()
            if child.is_file() and child.suffix.lower() == ".pdf":
                direct_pdf_count += 1
                continue
            if not child.is_dir() or _matching_pdf_source_root(resolved_child) is None:
                continue
            child_pdf_count = sum(
                1
                for candidate in child.iterdir()
                if candidate.is_file() and candidate.suffix.lower() == ".pdf"
            )
            directory_entries.append(
                {
                    "name": child.name,
                    "path": str(resolved_child),
                    "pdf_count": child_pdf_count,
                }
            )
        except (OSError, PermissionError):
            continue

    parent: str | None = None
    if current != matching_root:
        candidate_parent = current.parent.resolve()
        if _matching_pdf_source_root(candidate_parent) is not None:
            parent = str(candidate_parent)
    return {
        "configured": True,
        "roots": roots,
        "current": str(current),
        "current_name": current.name or str(current),
        "parent": parent,
        "directories": sorted(directory_entries, key=lambda entry: entry["name"].casefold()),
        "pdf_count": direct_pdf_count,
        "message": "",
    }


@app.get("/api/training-collection")
def get_training_collection() -> dict[str, Any]:
    return load_collection_summary(DATA_DIR)


@app.post("/api/training-collection/rebuild")
def rebuild_training_data_collection() -> dict[str, Any]:
    with TRAINING_COLLECTION_LOCK:
        return rebuild_training_collection(DATA_DIR, BATCH_DB_PATH)


@app.get("/api/batches")
def list_batches(limit: int = 30) -> dict[str, Any]:
    with _batch_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM batches ORDER BY created_at DESC LIMIT ?",
            (max(1, min(limit, 200)),),
        ).fetchall()
        summaries = []
        for row in rows:
            summary = dict(row)
            summary["counts"] = _batch_counts(connection, str(row["batch_id"]))
            summaries.append(summary)
    return {"batches": summaries}


@app.get("/api/batches/{batch_id}")
def get_batch(
    batch_id: str,
    offset: int = 0,
    limit: int = 200,
    include_review: bool = False,
) -> dict[str, Any]:
    state = load_batch_state(batch_id, offset=offset, limit=limit)
    if include_review:
        for item in state["items"]:
            item.update(_job_review_summary(str(item["job_id"])))
    return state


@app.post("/api/batches/upload")
async def create_upload_batch(
    files: list[UploadFile] = File(...),
    concurrency: int = Form(DEFAULT_BATCH_WORKERS),
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="请选择至少一个 PDF")
    if len(files) > MAX_BATCH_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"单批最多上传 {MAX_BATCH_FILES} 个文件；更大批次请使用服务器文件夹模式",
        )
    invalid = [file.filename or "upload" for file in files if Path(file.filename or "").suffix.lower() != ".pdf"]
    if invalid:
        raise HTTPException(status_code=400, detail=f"批量任务只接受 PDF：{', '.join(invalid[:5])}")

    staged: list[dict[str, Any]] = []
    staged_job_ids: list[str] = []
    try:
        for index, file in enumerate(files, 1):
            filename = file.filename or f"document_{index}.pdf"
            job_id = uuid.uuid4().hex
            item_id = f"i{index:06d}"
            upload_dir = DATA_DIR / job_id / "upload"
            upload_dir.mkdir(parents=True)
            staged_job_ids.append(job_id)
            upload_path = upload_dir / safe_name(filename)
            size = 0
            with upload_path.open("wb") as destination:
                while chunk := await file.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise HTTPException(
                            status_code=413,
                            detail=f"{filename} 超过 {MAX_UPLOAD_BYTES // (1024 * 1024)} MB",
                        )
                    destination.write(chunk)
            if not size:
                raise HTTPException(status_code=400, detail=f"{filename} 是空文件")
            staged.append(
                {
                    "item_id": item_id,
                    "job_id": job_id,
                    "filename": filename,
                    "source_path": str(upload_path),
                    "size_bytes": size,
                    "status": "queued",
                    "created_at": utc_now(),
                    "started_at": None,
                    "completed_at": None,
                    "metrics": {},
                    "message": "",
                    "error": "",
                }
            )
    except Exception:
        for job_id in staged_job_ids:
            shutil.rmtree(DATA_DIR / job_id, ignore_errors=True)
        raise
    finally:
        for file in files:
            await file.close()

    state = create_batch_state(
        staged,
        concurrency=concurrency,
        source_kind="upload",
        source_label=f"上传 {len(staged)} 篇 PDF",
    )
    _start_batch(state)
    return state


@app.post("/api/batches/folder")
def create_folder_batch(request: FolderBatchRequest) -> dict[str, Any]:
    folder = Path(request.folder).expanduser().resolve()
    if PDF_SOURCE_ROOTS and _matching_pdf_source_root(folder) is None:
        raise HTTPException(status_code=403, detail="只能处理已配置 PDF 根目录中的文件夹")
    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"PDF 文件夹不存在：{folder}")
    iterator = folder.rglob("*.pdf") if request.recursive else folder.glob("*.pdf")
    pdf_paths = sorted(path for path in iterator if path.is_file())
    if not pdf_paths:
        raise HTTPException(status_code=404, detail="该文件夹中没有 PDF")

    items = [
        {
            "item_id": f"i{index:06d}",
            "job_id": uuid.uuid4().hex,
            "filename": path.name,
            "source_path": str(path),
            "size_bytes": path.stat().st_size,
            "status": "queued",
            "created_at": utc_now(),
            "started_at": None,
            "completed_at": None,
            "metrics": {},
            "message": "",
            "error": "",
        }
        for index, path in enumerate(pdf_paths, 1)
    ]
    state = create_batch_state(
        items,
        concurrency=request.concurrency,
        source_kind="folder",
        source_label=str(folder),
    )
    _start_batch(state)
    return state


@app.get("/api/jobs/{job_id}/result")
def get_job_result(job_id: str) -> dict[str, Any]:
    root = job_root(job_id)
    path = root / "job_result.json"
    if not path.is_file():
        raise HTTPException(status_code=404, detail="任务结果尚未生成")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        state_path = root / EDITOR_STATE_NAME
        if state_path.is_file():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for record in payload.get("records", []):
                figure = (state.get("figures") or {}).get(record.get("record_id"))
                if figure:
                    record.update(serialize_editor_figure(job_id, root, figure))
                    record["panel_count"] = (
                        0
                        if record.get("annotation_mode") == "no_split"
                        else len(record.get("panel_targets", []))
                    )
        return payload
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail="任务结果损坏") from exc


@app.get("/api/documents")
def list_document_visuals() -> dict[str, Any]:
    """List parsed documents whose visual sidecar can be edited in place."""
    documents_root = DOCUMENTS_DIR.resolve() if DOCUMENTS_DIR else None
    documents: list[dict[str, Any]] = []
    if documents_root is None or not documents_root.is_dir():
        return {"documents": documents}
    for document_dir in sorted(documents_root.iterdir(), key=lambda path: path.name.lower()):
        manifest_path = document_dir / "visual" / "manifest.json"
        if not document_dir.is_dir() or not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        metrics = manifest.get("metrics") if isinstance(manifest.get("metrics"), dict) else {}
        documents.append(
            {
                "name": document_dir.name,
                "job_id": document_job_id(document_dir.name),
                "figures": int(metrics.get("figures") or len(manifest.get("figures") or [])),
                "panels": int(metrics.get("panels") or 0),
                "needs_review": int(metrics.get("figures_needing_review") or 0)
                + int(metrics.get("low_confidence_panels") or 0),
                "updated_at": manifest.get("manual_updated_at") or manifest.get("completed_at"),
            }
        )
    return {"documents": documents}


@app.post("/api/documents/open")
def open_document_visual(request: DocumentVisualOpenRequest) -> dict[str, Any]:
    documents_root = DOCUMENTS_DIR.resolve() if DOCUMENTS_DIR else None
    if documents_root is None or not documents_root.is_dir():
        raise HTTPException(status_code=404, detail="文献目录尚未配置")
    document_dir = next(
        (
            path
            for path in documents_root.iterdir()
            if path.is_dir() and path.name == request.document_name
        ),
        None,
    )
    if document_dir is None:
        raise HTTPException(status_code=404, detail="文献不存在")
    resolved = document_dir.resolve()
    if not resolved.is_relative_to(documents_root):
        raise HTTPException(status_code=404, detail="文献不存在")
    return _document_visual_response(resolved)


@app.get("/api/jobs/{job_id}/figures/{record_id}/panel-versions")
def get_panel_versions(job_id: str, record_id: str, label: str) -> dict[str, Any]:
    root, state = load_editor_state(job_id)
    figure = state.get("figures", {}).get(record_id)
    if not figure:
        raise HTTPException(status_code=404, detail="整图记录不存在")
    normalized_label = normalize_panel_label(label)
    target = figure.get("panels", {}).get(normalized_label)
    if not target:
        return {
            "record_id": record_id,
            "label": normalized_label,
            "current_version": None,
            "current_url": None,
            "versions": [],
        }
    return {
        "record_id": record_id,
        **serialize_panel_target(job_id, root, target),
    }


@app.post("/api/jobs/{job_id}/figures/{record_id}/panel-versions")
def create_panel_version(
    job_id: str,
    record_id: str,
    request: ManualPanelCropRequest,
) -> dict[str, Any]:
    """Crop one panel from the immutable whole figure and confirm it as current."""
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="请先确认后再保存新版本")
    label = normalize_panel_label(request.label)

    with EDITOR_STATE_LOCK:
        root, state = load_editor_state(job_id)
        figure = state.get("figures", {}).get(record_id)
        if not figure:
            raise HTTPException(status_code=404, detail="整图记录不存在")
        _ensure_figure_annotation_state(figure)
        target = figure.get("panels", {}).get(label)
        current_version = (
            target.get("current_version")
            if target and target.get("active", True)
            else None
        )
        if request.expected_current_version != current_version:
            raise HTTPException(
                status_code=409,
                detail="当前图片已经被其他操作更新，请刷新版本后重试",
            )

        source_path = resolve_job_path(root, figure["source_path"])
        with Image.open(source_path) as opened:
            source = ImageOps.exif_transpose(opened)
            width, height = source.size
            if not (
                0 <= request.x0 < request.x1 <= width
                and 0 <= request.y0 < request.y1 <= height
            ):
                raise HTTPException(status_code=400, detail="框选区域超出整图范围")
            if request.x1 - request.x0 < 8 or request.y1 - request.y0 < 8:
                raise HTTPException(status_code=400, detail="框选区域太小")
            crop = source.crop((request.x0, request.y0, request.x1, request.y1)).copy()
            source_dpi = opened.info.get("dpi") or (300, 300)

        panel_dir = resolve_job_path(root, figure["panel_dir"])
        versions_dir = panel_dir / "manual_versions" / label
        versions_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = target.get("versions", []) if target else []
        version_number = max(
            (int(version["version"]) for version in existing_versions),
            default=0,
        ) + 1
        version_path = versions_dir / f"v{version_number:04d}_manual.png"
        crop.save(version_path, format="PNG", dpi=source_dpi)

        current_path = panel_dir / f"panel_{label}.png"
        shutil.copy2(version_path, current_path)
        version = {
            "version": version_number,
            "kind": "manual_crop",
            "created_at": utc_now(),
            "path": relative_job_path(root, version_path),
            "bbox_px": [request.x0, request.y0, request.x1, request.y1],
            "width_px": request.x1 - request.x0,
            "height_px": request.y1 - request.y0,
        }
        if target is None:
            target = {
                "label": label,
                "current_version": version_number,
                "current_path": relative_job_path(root, current_path),
                "manifest_path": relative_job_path(root, versions_dir / "versions.json"),
                "versions": [version],
                "understanding_current_version": 0,
                "understanding_text": "",
                "understanding_manifest_path": relative_job_path(
                    root,
                    versions_dir / "understanding_versions.json",
                ),
                "understanding_versions": [],
                "active": True,
                "deleted_at": None,
            }
            figure.setdefault("panels", {})[label] = target
        else:
            target["current_version"] = version_number
            target["current_path"] = relative_job_path(root, current_path)
            target.setdefault("versions", []).append(version)
            target["active"] = True
            target["deleted_at"] = None

        editable_labels = set(figure.get("editable_labels", []))
        editable_labels.add(label)
        figure["editable_labels"] = sorted(editable_labels)
        _invalidate_figure_review(figure)
        state["updated_at"] = utc_now()
        write_panel_version_manifest(root, target)
        if not target.get("understanding_manifest_path"):
            target["understanding_current_version"] = int(
                target.get("understanding_current_version", 0)
            )
            target["understanding_text"] = str(target.get("understanding_text", ""))
            target["understanding_versions"] = target.get(
                "understanding_versions",
                [],
            )
            target["understanding_manifest_path"] = relative_job_path(
                root,
                versions_dir / "understanding_versions.json",
            )
        if not resolve_job_path(
            root,
            target["understanding_manifest_path"],
            must_exist=False,
        ).exists():
            write_understanding_version_manifest(root, target)
        write_json_atomic(root / EDITOR_STATE_NAME, state)
        _sync_document_visual_manifest(root, state, figure, target)

    _sync_training_record_safely(job_id, record_id)
    return {
        "record_id": record_id,
        **serialize_panel_target(job_id, root, target),
    }


@app.delete("/api/jobs/{job_id}/figures/{record_id}/panels")
def delete_panel_target(
    job_id: str,
    record_id: str,
    request: PanelDeleteRequest,
) -> dict[str, Any]:
    """Deactivate a wrong panel without deleting its immutable history."""
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="请先确认后再删除面板")
    label = normalize_panel_label(request.label)
    with EDITOR_STATE_LOCK:
        root, state = load_editor_state(job_id)
        figure = state.get("figures", {}).get(record_id)
        if not figure:
            raise HTTPException(status_code=404, detail="整图记录不存在")
        _ensure_figure_annotation_state(figure)
        target = figure.get("panels", {}).get(label)
        if not target or not target.get("active", True):
            raise HTTPException(status_code=404, detail="当前面板不存在")
        if int(target.get("current_version", 0)) != request.expected_current_version:
            raise HTTPException(status_code=409, detail="当前图片已经更新，请刷新后重试")
        target["active"] = False
        target["deleted_at"] = utc_now()
        _invalidate_figure_review(figure)
        state["updated_at"] = utc_now()
        write_panel_version_manifest(root, target)
        write_json_atomic(root / EDITOR_STATE_NAME, state)
        response = serialize_editor_figure(job_id, root, figure)
    _sync_training_record_safely(job_id, record_id)
    return response


@app.post("/api/jobs/{job_id}/figures/{record_id}/review")
def review_figure_annotation(
    job_id: str,
    record_id: str,
    request: FigureReviewRequest,
) -> dict[str, Any]:
    """Record whether the complete Figure has been human reviewed."""
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="请先确认整图审核结果")
    status = request.status.strip().lower()
    if status not in {"verified", "ambiguous", "proposed"}:
        raise HTTPException(status_code=400, detail="审核状态无效")
    requested_mode = (request.annotation_mode or "").strip().lower()
    if requested_mode and requested_mode not in {"panel_boxes", "no_split"}:
        raise HTTPException(status_code=400, detail="标注类型无效")
    with EDITOR_STATE_LOCK:
        root, state = load_editor_state(job_id)
        figure = state.get("figures", {}).get(record_id)
        if not figure:
            raise HTTPException(status_code=404, detail="整图记录不存在")
        _ensure_figure_annotation_state(figure)
        annotation_mode = requested_mode or str(figure.get("annotation_mode") or "panel_boxes")
        if status == "verified":
            if annotation_mode == "panel_boxes":
                panels, issues = _figure_review_panels(root, figure)
                if issues:
                    raise HTTPException(
                        status_code=400,
                        detail=f"仍有无效框：{', '.join(issues[:5])}",
                    )
                if not panels:
                    raise HTTPException(
                        status_code=400,
                        detail="当前没有有效面板框；若整图无需划分，请点击“确认整图无需划分”",
                    )
            figure["annotation_complete"] = True
            figure["reviewed_at"] = utc_now()
        else:
            figure["annotation_complete"] = False
            figure["reviewed_at"] = utc_now() if status == "ambiguous" else None
        figure["review_status"] = status
        figure["annotation_mode"] = annotation_mode
        figure["reviewer"] = request.reviewer.strip()[:200] or None
        figure["annotation_version"] = int(figure.get("annotation_version", 0)) + 1
        state["updated_at"] = utc_now()
        write_json_atomic(root / EDITOR_STATE_NAME, state)
        response = serialize_editor_figure(job_id, root, figure)
    collection_record = _sync_training_record_safely(job_id, record_id)
    response["ready_for_training"] = bool(
        collection_record and collection_record.get("ready_for_training")
    )
    return response


@app.post("/api/jobs/{job_id}/figures/{record_id}/understanding-versions")
def create_panel_understanding_version(
    job_id: str,
    record_id: str,
    request: PanelUnderstandingRequest,
) -> dict[str, Any]:
    """Save a confirmed, immutable revision of a panel's human understanding."""
    if not request.confirmed:
        raise HTTPException(status_code=400, detail="请先确认后再保存图片理解")
    label = normalize_panel_label(request.label)
    text = request.text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if "\x00" in text:
        raise HTTPException(status_code=400, detail="图片理解包含无效字符")
    if len(text) > 20000:
        raise HTTPException(status_code=400, detail="图片理解不能超过 20000 个字符")

    with EDITOR_STATE_LOCK:
        root, state = load_editor_state(job_id)
        figure = state.get("figures", {}).get(record_id)
        if not figure:
            raise HTTPException(status_code=404, detail="整图记录不存在")
        target = figure.get("panels", {}).get(label)
        if not target:
            raise HTTPException(status_code=404, detail="请先生成该标签对应的子图")

        current_version = int(target.get("understanding_current_version", 0))
        if request.expected_current_version != current_version:
            raise HTTPException(
                status_code=409,
                detail="图片理解已经被其他操作更新，请刷新后重试",
            )

        versions_dir = resolve_job_path(root, figure["panel_dir"]) / "manual_versions" / label
        versions_dir.mkdir(parents=True, exist_ok=True)
        versions = target.setdefault("understanding_versions", [])
        version_number = max(
            (int(version["version"]) for version in versions),
            default=0,
        ) + 1
        created_at = utc_now()
        version_path = versions_dir / f"understanding_v{version_number:04d}.json"
        version = {
            "version": version_number,
            "created_at": created_at,
            "text": text,
            "path": relative_job_path(root, version_path),
        }
        write_json_atomic(
            version_path,
            {
                "record_id": record_id,
                "label": label,
                "version": version_number,
                "created_at": created_at,
                "text": text,
            },
        )

        target["understanding_current_version"] = version_number
        target["understanding_text"] = text
        target["understanding_manifest_path"] = relative_job_path(
            root,
            versions_dir / "understanding_versions.json",
        )
        versions.append(version)
        state["updated_at"] = created_at
        write_understanding_version_manifest(root, target)
        write_json_atomic(root / EDITOR_STATE_NAME, state)
        _sync_document_visual_manifest(root, state, figure, target)

    return {
        "record_id": record_id,
        **serialize_panel_target(job_id, root, target),
    }


@app.get("/files/{job_id}/{relative_path:path}")
def generated_file(job_id: str, relative_path: str) -> FileResponse:
    root = job_root(job_id)
    target = (root / relative_path).resolve()
    if not target.is_relative_to(root) or not target.is_file():
        raise HTTPException(status_code=404, detail="文件不存在")
    return FileResponse(target)


@app.post("/api/process")
async def process_file(file: UploadFile = File(...)) -> dict:
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=400, detail="仅支持 PDF、PNG、JPG、WEBP、BMP 和 TIFF")

    job_id = uuid.uuid4().hex
    job_dir = DATA_DIR / job_id
    upload_dir = job_dir / "upload"
    upload_dir.mkdir(parents=True)
    upload_path = upload_dir / safe_name(filename)
    size = 0
    try:
        with upload_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="文件不能超过 80 MB")
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=400, detail="上传文件为空")
        result = await asyncio.to_thread(run_pipeline, job_id, upload_path, filename)
        write_json_atomic(job_dir / "job_result.json", result)
        for record in result.get("records", []):
            record_id = str(record.get("record_id") or "")
            if record_id:
                _sync_training_record_safely(job_id, record_id)
        return result
    except HTTPException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"处理失败：{exc}") from exc
    finally:
        await file.close()
