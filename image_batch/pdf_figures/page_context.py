from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Collection

import fitz  # PyMuPDF

"""Page text access, geometry helpers, validation, and path utilities."""

from .models import (
    BBox,
    DEFAULT_SCHEME_MIN_HEIGHT_PT,
    DEFAULT_SCHEME_MIN_WIDTH_PT,
    TextBlock,
)

def collect_text_blocks(page: fitz.Page, page_number: int) -> list[TextBlock]:
    """Collect text blocks with dominant font metadata for caption continuity."""
    blocks: list[TextBlock] = []
    try:
        page_dict = page.get_text("dict", sort=True)
    except TypeError:
        page_dict = page.get_text("dict")
    except Exception:
        page_dict = None

    if isinstance(page_dict, dict):
        for raw_block in page_dict.get("blocks", []):
            if raw_block.get("type") != 0 or "bbox" not in raw_block:
                continue
            line_texts: list[str] = []
            weighted_size = 0.0
            weight_total = 0
            font_weights: Counter[str] = Counter()
            for line in raw_block.get("lines", []):
                span_texts: list[str] = []
                for span in line.get("spans", []):
                    span_text = str(span.get("text", ""))
                    if not span_text.strip():
                        continue
                    span_texts.append(span_text)
                    weight = max(1, len(span_text.strip()))
                    size = float(span.get("size", 0.0) or 0.0)
                    weighted_size += size * weight
                    weight_total += weight
                    font_weights[str(span.get("font", "") or "")] += weight
                line_text = clean_text(" ".join(span_texts))
                if line_text:
                    line_texts.append(line_text)
            block_text = clean_text(" ".join(line_texts))
            if not block_text:
                continue
            dominant_font = font_weights.most_common(1)[0][0] if font_weights else ""
            font_size = weighted_size / weight_total if weight_total else 0.0
            blocks.append(
                TextBlock(
                    page_number=page_number,
                    text=block_text,
                    bbox_pt=tuple(float(value) for value in raw_block["bbox"]),
                    font_size_pt=font_size,
                    font_name=dominant_font,
                    bold=bool(re.search(r"bold|semibold|demibold", dominant_font, re.I)),
                    line_count=max(1, len(line_texts)),
                )
            )
    else:
        for raw_block in page.get_text("blocks"):
            if len(raw_block) < 5:
                continue
            if len(raw_block) >= 7 and int(raw_block[6]) != 0:
                continue
            block_text = clean_text(raw_block[4])
            if not block_text:
                continue
            blocks.append(
                TextBlock(
                    page_number=page_number,
                    text=block_text,
                    bbox_pt=tuple(float(value) for value in raw_block[:4]),
                )
            )
    return sorted(blocks, key=lambda block: (block.bbox_pt[1], block.bbox_pt[0]))

def bbox_coverage_ratio(inner: BBox, outer: BBox) -> float:
    intersection = fitz.Rect(inner) & fitz.Rect(outer)
    inner_rect = fitz.Rect(inner)
    if inner_rect.is_empty or intersection.is_empty:
        return 0.0
    return float(intersection.get_area() / max(1.0, inner_rect.get_area()))

def parse_page_selector(value: str) -> set[int] | None:
    value = clean_text(value)
    if not value:
        return None

    pages: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start <= 0 or end <= 0 or end < start:
                raise ValueError(f"Invalid page range: {part}")
            pages.update(range(start, end + 1))
        else:
            page = int(part)
            if page <= 0:
                raise ValueError(f"Invalid page number: {part}")
            pages.add(page)

    return pages or None

def validate_inputs(
    pdf_path: Path,
    dpi: int,
    min_width_pt: float,
    min_height_pt: float,
    pages: set[int] | None,
    *,
    scheme_min_width_pt: float = DEFAULT_SCHEME_MIN_WIDTH_PT,
    scheme_min_height_pt: float = DEFAULT_SCHEME_MIN_HEIGHT_PT,
) -> None:
    if not pdf_path.exists():
        raise FileNotFoundError(f"Input PDF not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Input file must be a PDF: {pdf_path}")
    if dpi <= 0:
        raise ValueError("dpi must be a positive integer.")
    if min_width_pt < 0 or min_height_pt < 0:
        raise ValueError("min_width_pt and min_height_pt must be non-negative.")
    if scheme_min_width_pt < 0 or scheme_min_height_pt < 0:
        raise ValueError(
            "scheme_min_width_pt and scheme_min_height_pt must be non-negative."
        )
    if pages is not None and any(page <= 0 for page in pages):
        raise ValueError("pages must use 1-based positive page numbers.")

def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()

def sanitize_name(value: Any) -> str:
    text = clean_text(value) or "item"
    text = re.sub(r'[<>:"/\\|?*]+', "_", text)
    text = re.sub(r"\s+", "_", text)
    return text.strip(" ._") or "item"

def unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    index = 2
    while True:
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1

def union_bbox(left: BBox, right: BBox) -> BBox:
    return (
        min(left[0], right[0]),
        min(left[1], right[1]),
        max(left[2], right[2]),
        max(left[3], right[3]),
    )

def bbox_center_x(bbox: BBox) -> float:
    return (bbox[0] + bbox[2]) / 2

def bbox_center_y(bbox: BBox) -> float:
    return (bbox[1] + bbox[3]) / 2

def horizontal_overlap_ratio(left: BBox, right: BBox) -> float:
    overlap = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    width = max(1.0, min(left[2] - left[0], right[2] - right[0]))
    return overlap / width

def looks_like_heading(text: str) -> bool:
    compact = clean_text(text)
    if len(compact) > 90:
        return False
    if re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", compact):
        return True
    return compact.isupper() and any(character.isalpha() for character in compact)
