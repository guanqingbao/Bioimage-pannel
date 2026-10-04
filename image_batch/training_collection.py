"""Build a split-agnostic, training-ready panel annotation collection.

The collection deliberately keeps rule/model proposals separate from samples
that a person has confirmed as complete.  Verified samples also receive an
unsplit YOLO label, so train/val/test grouping can be performed later without
re-reading the editor's internal files.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
from typing import Any
import uuid

from PIL import Image, ImageOps


SCHEMA_VERSION = "1.1"
COLLECTION_DIRNAME = "training_collection"
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".tmp_{path.name}_{uuid.uuid4().hex}"
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    _write_text_atomic(path, json.dumps(payload, ensure_ascii=False, indent=2))


def sha256_file(path: Path) -> str:
    stat = path.stat()
    key = (str(path.resolve()), int(stat.st_size), int(stat.st_mtime_ns))
    cached = _HASH_CACHE.get(key)
    if cached:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    _HASH_CACHE[key] = value
    return value


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _resolve_inside(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve()
    resolved = (resolved_root / relative_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError(f"Path escapes job root: {relative_path}")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _source_context(data_dir: Path, database_path: Path, job_id: str) -> dict[str, Any]:
    filename = ""
    source_path: Path | None = None
    if database_path.is_file():
        connection = sqlite3.connect(database_path, timeout=30)
        try:
            row = connection.execute(
                "SELECT filename, source_path FROM batch_items WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row:
                filename = str(row[0] or "")
                source_path = Path(str(row[1])).expanduser().resolve()
        finally:
            connection.close()
    if source_path is None or not source_path.is_file():
        upload_dir = data_dir / job_id / "upload"
        candidates = sorted(path for path in upload_dir.glob("*") if path.is_file())
        if candidates:
            source_path = candidates[0].resolve()
            filename = filename or source_path.name

    source_hash = sha256_file(source_path) if source_path and source_path.is_file() else None
    is_pdf = bool(source_path and source_path.suffix.lower() == ".pdf")
    source_group_id = (
        f"paper_{source_hash}" if source_hash and is_pdf else f"source_{source_hash}"
        if source_hash
        else f"job_{job_id}"
    )
    return {
        "paper_id": f"paper_{source_hash}" if source_hash and is_pdf else None,
        "source_group_id": source_group_id,
        "source_file_name": filename or (source_path.name if source_path else ""),
        "source_file_path": str(source_path) if source_path else None,
        "source_file_sha256": source_hash,
    }


def _layout_for_figure(job_root: Path, figure: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    panel_dir = _resolve_inside(job_root, str(figure["panel_dir"]))
    layout_path = panel_dir / "layout.json"
    if not layout_path.is_file():
        return layout_path, {}
    return layout_path, _load_json(layout_path)


def _current_version(target: dict[str, Any]) -> dict[str, Any] | None:
    number = int(target.get("current_version", 0))
    return next(
        (
            version
            for version in target.get("versions", [])
            if int(version.get("version", -1)) == number
        ),
        None,
    )


def _valid_bbox(value: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        bbox = [float(number) for number in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(number) for number in bbox):
        return None
    x1, y1, x2, y2 = bbox
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return None
    return bbox


def effective_panels(
    figure: dict[str, Any],
    layout: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Return active current boxes in original-Figure pixels."""
    layout_panels = {
        str(label).upper(): panel
        for label, panel in (layout.get("panels") or {}).items()
    }
    panels: list[dict[str, Any]] = []
    issues: list[str] = []
    targets = figure.get("panels") if isinstance(figure.get("panels"), dict) else {}
    for label, target in sorted(targets.items(), key=lambda item: str(item[0]).upper()):
        normalized_label = str(label).upper()
        if target.get("active", True) is False:
            continue
        version_number = int(target.get("current_version", 0))
        version = _current_version(target)
        bbox = _valid_bbox(version.get("bbox_px") if version else None, width, height)
        source = "human_manual"
        if bbox is None and version_number == 0:
            bbox = _valid_bbox(
                (layout_panels.get(normalized_label) or {}).get("rect"),
                width,
                height,
            )
            source = "rule_proposal"
        if bbox is None:
            issues.append(f"panel_{normalized_label}_missing_or_invalid_bbox")
            continue
        panels.append(
            {
                "instance_id": f"panel_{normalized_label}",
                "category": "main_panel",
                "class_id": 0,
                "level": 1,
                "displayed_label": normalized_label,
                "bbox_xyxy": [int(number) if number.is_integer() else number for number in bbox],
                "annotation_source": source,
                "panel_version": version_number,
                "panel_version_created_at": version.get("created_at") if version else None,
            }
        )
    return panels, issues


def _sample_id(source_group_id: str, figure_id: str, image_id: str) -> str:
    key = f"{source_group_id}\0{figure_id}\0{image_id}".encode("utf-8")
    return f"sample_{hashlib.sha256(key).hexdigest()[:24]}"


def _yolo_lines(record: dict[str, Any]) -> str:
    width = float(record["image"]["width"])
    height = float(record["image"]["height"])
    lines: list[str] = []
    for panel in record["panels"]:
        x1, y1, x2, y2 = (float(value) for value in panel["bbox_xyxy"])
        xc = (x1 + x2) / (2.0 * width)
        yc = (y1 + y2) / (2.0 * height)
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        values = (xc, yc, box_width, box_height)
        if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"Invalid normalized box: {panel['bbox_xyxy']}")
        lines.append("0 " + " ".join(f"{value:.8f}" for value in values))
    return "\n".join(lines) + ("\n" if lines else "")


def build_training_record(
    data_dir: Path,
    database_path: Path,
    job_id: str,
    record_id: str,
) -> tuple[dict[str, Any], Path]:
    job_root = (data_dir / job_id).resolve()
    state_path = job_root / "editor_state.json"
    state = _load_json(state_path)
    figure = (state.get("figures") or {}).get(record_id)
    if not isinstance(figure, dict):
        raise KeyError(f"Figure record not found: {job_id}/{record_id}")
    source_path = _resolve_inside(job_root, str(figure["source_path"]))
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened)
        width, height = image.size
    image_hash = sha256_file(source_path)
    image_id = f"img_{image_hash}"
    source_context = _source_context(data_dir, database_path, job_id)
    figure_id = str(figure.get("figure_id") or source_path.stem)
    sample_id = _sample_id(source_context["source_group_id"], figure_id, image_id)
    layout_path, layout = _layout_for_figure(job_root, figure)
    panels, issues = effective_panels(figure, layout, width, height)
    review_status = str(figure.get("review_status") or "proposed")
    annotation_mode = str(figure.get("annotation_mode") or "panel_boxes")
    if annotation_mode not in {"panel_boxes", "no_split"}:
        annotation_mode = "panel_boxes"
    if annotation_mode == "no_split":
        panels = []
        issues = []
    annotation_complete = bool(figure.get("annotation_complete", False))
    has_valid_truth = annotation_mode == "no_split" or bool(panels)
    ready = review_status == "verified" and annotation_complete and has_valid_truth and not issues
    collection_root = data_dir / COLLECTION_DIRNAME
    asset_path = collection_root / "assets" / "images" / f"{image_id}{source_path.suffix.lower()}"
    record = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "image_id": image_id,
        "paper_id": source_context["paper_id"],
        "source_group_id": source_context["source_group_id"],
        "figure_id": figure_id,
        "figure_number": figure.get("figure_number"),
        "caption_kind": figure.get("caption_kind"),
        "caption": figure.get("caption", ""),
        "image": {
            "file_name": source_path.name,
            "collection_path": asset_path.relative_to(collection_root).as_posix(),
            "width": width,
            "height": height,
            "file_sha256": image_hash,
            "dpi": 300,
        },
        "coordinate_space": "original_figure_pixels",
        "bbox_format": "xyxy",
        "coordinate_convention": "zero_based_half_open",
        "review_status": review_status,
        "annotation_mode": annotation_mode,
        "annotation_complete": annotation_complete,
        "annotation_version": int(figure.get("annotation_version", 0)),
        "human_panel_count": len(panels) if ready else None,
        "reviewer": figure.get("reviewer"),
        "reviewed_at": figure.get("reviewed_at"),
        "ready_for_training": ready,
        "panels": panels,
        "issues": issues,
        "provenance": {
            "job_id": job_id,
            "record_id": record_id,
            "editor_state": str(state_path),
            "layout_json": str(layout_path) if layout_path.is_file() else None,
            "source_figure": str(source_path),
            **source_context,
        },
        "collected_at": _utc_now(),
    }
    return record, source_path


def sync_training_record(
    data_dir: Path,
    database_path: Path,
    job_id: str,
    record_id: str,
) -> dict[str, Any]:
    record, source_path = build_training_record(data_dir, database_path, job_id, record_id)
    collection_root = data_dir / COLLECTION_DIRNAME
    asset_path = collection_root / record["image"]["collection_path"]
    annotation_path = collection_root / "annotations" / f"{record['sample_id']}.json"
    ready_image = collection_root / "ready" / "images" / f"{record['sample_id']}{source_path.suffix.lower()}"
    ready_label = collection_root / "ready" / "labels" / f"{record['sample_id']}.txt"
    _link_or_copy(source_path, asset_path)
    _write_json_atomic(annotation_path, record)
    if record["ready_for_training"]:
        _link_or_copy(asset_path, ready_image)
        _write_text_atomic(ready_label, _yolo_lines(record))
    else:
        if ready_image.exists():
            ready_image.unlink()
        if ready_label.exists():
            ready_label.unlink()
    record["annotation_path"] = annotation_path.relative_to(collection_root).as_posix()
    record["ready_image_path"] = (
        ready_image.relative_to(collection_root).as_posix()
        if record["ready_for_training"]
        else None
    )
    record["ready_label_path"] = (
        ready_label.relative_to(collection_root).as_posix()
        if record["ready_for_training"]
        else None
    )
    return record


def _manifest_entry(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": record["sample_id"],
        "image_id": record["image_id"],
        "paper_id": record["paper_id"],
        "source_group_id": record["source_group_id"],
        "figure_id": record["figure_id"],
        "review_status": record["review_status"],
        "annotation_mode": record["annotation_mode"],
        "annotation_complete": record["annotation_complete"],
        "ready_for_training": record["ready_for_training"],
        "panel_count": len(record["panels"]),
        "annotation_version": record["annotation_version"],
        "image_path": record["image"]["collection_path"],
        "annotation_path": record["annotation_path"],
        "ready_image_path": record["ready_image_path"],
        "ready_label_path": record["ready_label_path"],
        "issues": record["issues"],
    }


def rebuild_training_collection(data_dir: Path, database_path: Path) -> dict[str, Any]:
    collection_root = data_dir / COLLECTION_DIRNAME
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for state_path in sorted(data_dir.glob("*/editor_state.json")):
        job_id = state_path.parent.name
        try:
            state = _load_json(state_path)
        except Exception as exc:
            failures.append({"job_id": job_id, "record_id": "", "error": str(exc)})
            continue
        for record_id in sorted((state.get("figures") or {})):
            try:
                records.append(sync_training_record(data_dir, database_path, job_id, record_id))
            except Exception as exc:
                failures.append({"job_id": job_id, "record_id": record_id, "error": str(exc)})

    entries = [_manifest_entry(record) for record in records]
    ready_entries = [entry for entry in entries if entry["ready_for_training"]]
    manifest_text = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)
    ready_text = "".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in ready_entries)
    _write_text_atomic(collection_root / "manifests" / "images.jsonl", manifest_text)
    _write_text_atomic(collection_root / "manifests" / "ready.jsonl", ready_text)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "built_at": _utc_now(),
        "collection_root": str(collection_root.resolve()),
        "samples": len(entries),
        "unique_images": len({entry["image_id"] for entry in entries}),
        "source_groups": len({entry["source_group_id"] for entry in entries}),
        "proposed": sum(entry["review_status"] == "proposed" for entry in entries),
        "verified": sum(entry["review_status"] == "verified" for entry in entries),
        "ambiguous": sum(entry["review_status"] == "ambiguous" for entry in entries),
        "ready_for_training": len(ready_entries),
        "no_split": sum(entry["annotation_mode"] == "no_split" for entry in ready_entries),
        "panel_box_samples": sum(
            entry["annotation_mode"] == "panel_boxes" for entry in ready_entries
        ),
        "candidate_panels": sum(entry["panel_count"] for entry in entries),
        "verified_panels": sum(entry["panel_count"] for entry in ready_entries),
        "failures": failures,
    }
    _write_json_atomic(collection_root / "collection_summary.json", summary)
    return summary


def load_collection_summary(data_dir: Path) -> dict[str, Any]:
    path = data_dir / COLLECTION_DIRNAME / "collection_summary.json"
    if not path.is_file():
        return {
            "schema_version": SCHEMA_VERSION,
            "collection_root": str((data_dir / COLLECTION_DIRNAME).resolve()),
            "samples": 0,
            "unique_images": 0,
            "source_groups": 0,
            "proposed": 0,
            "verified": 0,
            "ambiguous": 0,
            "ready_for_training": 0,
            "no_split": 0,
            "panel_box_samples": 0,
            "candidate_panels": 0,
            "verified_panels": 0,
            "failures": [],
            "built_at": None,
        }
    return _load_json(path)
