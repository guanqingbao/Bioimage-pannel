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

"""Safe figure-region expansion, native-label mapping, 300 DPI output metadata."""

from .captions import CAPTION_START_RE, parse_expected_panel_labels
from .models import (
    BBox,
    Caption,
    ExtractionResult,
    MAX_FIGURE_CONTEXT_EXPANSION_PT,
    MAX_SCHEME_CONTEXT_EXPANSION_PT,
    caption_segments_on_page,
    normalize_caption_kind,
)
from .page_context import clean_text, collect_text_blocks, looks_like_heading, sanitize_name

def _rect_axis_overlap(first: fitz.Rect, second: fitz.Rect, horizontal: bool) -> float:
    if horizontal:
        return max(0.0, min(first.x1, second.x1) - max(first.x0, second.x0))
    return max(0.0, min(first.y1, second.y1) - max(first.y0, second.y0))

def _rect_axis_gap(first: fitz.Rect, second: fitz.Rect, horizontal: bool) -> float:
    if horizontal:
        return max(0.0, first.x0 - second.x1, second.x0 - first.x1)
    return max(0.0, first.y0 - second.y1, second.y0 - first.y1)

def _caption_rect_intersects(
    rect: fitz.Rect,
    caption: Caption,
    page_number: int,
    tolerance: float = 1.0,
) -> bool:
    padded = fitz.Rect(
        rect.x0 - tolerance,
        rect.y0 - tolerance,
        rect.x1 + tolerance,
        rect.y1 + tolerance,
    )
    return any(
        not (padded & fitz.Rect(segment.bbox_pt)).is_empty
        for segment in caption_segments_on_page(caption, page_number)
    )

def _is_likely_native_figure_label(text: str, expected_labels: Collection[str]) -> bool:
    compact = clean_text(text).strip()
    stripped = compact.strip("()[]{}.:; ")
    if not stripped:
        return False
    expected = {str(label).upper() for label in expected_labels}
    if stripped.upper() in expected:
        return True
    if re.fullmatch(r"[A-Za-z]", stripped):
        return True
    if re.fullmatch(r"[A-Za-z]\d{0,2}", stripped):
        return True
    if re.fullmatch(r"\d+(?:\.\d+)?%?", stripped):
        return True
    return False

def _is_short_figure_context_text(text: str) -> bool:
    compact = clean_text(text)
    if not compact or CAPTION_START_RE.match(compact) or looks_like_heading(compact):
        return False
    words = compact.split()
    if len(compact) > 72 or len(words) > 9:
        return False
    # A sentence-like block several words long is more likely body prose than a
    # native axis / legend / panel label. Very close blocks can still be added by
    # the caller through the strict-gap path below.
    if len(words) >= 7 and re.search(r"[.!?]$", compact):
        return False
    return True

def _clip_expansion_away_from_caption(
    rect: fitz.Rect,
    base_rect: fitz.Rect,
    caption: Caption,
    page_number: int,
) -> fitz.Rect:
    """Prevent context expansion from swallowing a same-page caption."""
    clipped = fitz.Rect(rect)
    segments = caption_segments_on_page(caption, page_number)
    if not segments:
        return clipped

    caption_union = fitz.Rect(segments[0].bbox_pt)
    for segment in segments[1:]:
        caption_union |= fitz.Rect(segment.bbox_pt)

    if caption_union.y0 >= base_rect.y1 - 4.0:
        clipped.y1 = min(clipped.y1, caption_union.y0 - 2.0)
    elif caption_union.y1 <= base_rect.y0 + 4.0:
        clipped.y0 = max(clipped.y0, caption_union.y1 + 2.0)
    elif caption_union.x0 >= base_rect.x1 - 4.0:
        clipped.x1 = min(clipped.x1, caption_union.x0 - 2.0)
    elif caption_union.x1 <= base_rect.x0 + 4.0:
        clipped.x0 = max(clipped.x0, caption_union.x1 + 2.0)
    return clipped

def expand_figure_rect_with_nearby_content(
    page: fitz.Page,
    figure_rect: fitz.Rect,
    caption: Caption,
    source_block_indices: Collection[int],
) -> fitz.Rect:
    """Include native labels and vector fragments just outside a raster cluster.

    PDF figures are often assembled from one or more raster XObjects while panel
    letters, axis titles, legends, arrows, or borders remain native PDF text/vector
    objects. Expansion is therefore applied to *all* matched figures, not only
    four-or-more-tile mosaics. The search is bounded relative to the original
    raster rectangle so nearby article prose cannot trigger runaway chaining.
    """
    del source_block_indices  # Kept in the public helper signature for compatibility.

    page_number = page.number + 1
    page_rect = page.rect
    base = fitz.Rect(figure_rect)
    expanded = fitz.Rect(base)
    expected_labels = parse_expected_panel_labels(caption.text)

    context_limit = (
        MAX_SCHEME_CONTEXT_EXPANSION_PT
        if normalize_caption_kind(caption.caption_kind) == "scheme"
        else MAX_FIGURE_CONTEXT_EXPANSION_PT
    )
    horizontal_limit = min(
        context_limit,
        max(26.0, 0.14 * max(1.0, base.width)),
    )
    vertical_limit = min(
        context_limit,
        max(26.0, 0.14 * max(1.0, base.height)),
    )
    allowed = fitz.Rect(
        max(page_rect.x0, base.x0 - horizontal_limit),
        max(page_rect.y0, base.y0 - vertical_limit),
        min(page_rect.x1, base.x1 + horizontal_limit),
        min(page_rect.y1, base.y1 + vertical_limit),
    )
    allowed = _clip_expansion_away_from_caption(allowed, base, caption, page_number)

    text_blocks = collect_text_blocks(page, page_number)
    text_rects: list[tuple[fitz.Rect, bool, bool]] = []
    for block in text_blocks:
        text = block.text.strip()
        if not text or CAPTION_START_RE.match(text):
            continue
        block_rect = fitz.Rect(block.bbox_pt)
        if (block_rect & allowed).is_empty:
            continue
        if _caption_rect_intersects(block_rect, caption, page_number, tolerance=2.0):
            continue
        likely_label = _is_likely_native_figure_label(text, expected_labels)
        short_context = _is_short_figure_context_text(text)
        if not likely_label and not short_context:
            continue
        text_rects.append((block_rect, likely_label, short_context))

    drawing_rects: list[fitz.Rect] = []
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for drawing in drawings:
        raw_rect = drawing.get("rect")
        if raw_rect is None:
            continue
        drawing_rect = fitz.Rect(raw_rect)
        if drawing_rect.is_empty or drawing_rect.get_area() < 2.0:
            continue
        if (drawing_rect & allowed).is_empty:
            continue
        # Ignore page frames/backgrounds unless they substantially overlap the
        # raster figure itself.
        if drawing_rect.get_area() >= 0.65 * max(1.0, page_rect.get_area()):
            overlap = (drawing_rect & base).get_area() / max(1.0, base.get_area())
            if overlap < 0.50:
                continue
        drawing_rects.append(drawing_rect)

    # Two bounded passes allow a label immediately adjacent to a vector border to
    # join the same crop without permitting arbitrary page-wide transitive merges.
    for _ in range(2):
        changed = False
        for block_rect, likely_label, short_context in text_rects:
            if not (block_rect & expanded).is_empty:
                continue
            horizontal_gap = _rect_axis_gap(expanded, block_rect, horizontal=True)
            vertical_gap = _rect_axis_gap(expanded, block_rect, horizontal=False)
            horizontal_overlap = _rect_axis_overlap(expanded, block_rect, horizontal=True)
            vertical_overlap = _rect_axis_overlap(expanded, block_rect, horizontal=False)

            label_margin = min(60.0, horizontal_limit if vertical_overlap > 0 else vertical_limit)
            context_margin = 24.0
            permitted_margin = label_margin if likely_label else context_margin
            orthogonally_aligned = horizontal_overlap >= 1.0 or vertical_overlap >= 1.0
            extremely_close = horizontal_gap <= 5.0 and vertical_gap <= 5.0
            if not orthogonally_aligned and not extremely_close:
                continue
            if horizontal_gap > permitted_margin or vertical_gap > permitted_margin:
                continue
            if not likely_label and short_context and not extremely_close:
                # Longer native legend/axis text must align strongly with one edge.
                overlap_ratio = max(
                    horizontal_overlap / max(1.0, min(expanded.width, block_rect.width)),
                    vertical_overlap / max(1.0, min(expanded.height, block_rect.height)),
                )
                if overlap_ratio < 0.30:
                    continue
            candidate_union = expanded | block_rect
            if not allowed.contains(candidate_union):
                continue
            expanded = candidate_union
            changed = True

        for drawing_rect in drawing_rects:
            if not (drawing_rect & expanded).is_empty:
                # Intersecting drawing content is already represented by the crop.
                continue
            horizontal_gap = _rect_axis_gap(expanded, drawing_rect, horizontal=True)
            vertical_gap = _rect_axis_gap(expanded, drawing_rect, horizontal=False)
            horizontal_overlap = _rect_axis_overlap(expanded, drawing_rect, horizontal=True)
            vertical_overlap = _rect_axis_overlap(expanded, drawing_rect, horizontal=False)
            if horizontal_gap > 14.0 or vertical_gap > 14.0:
                continue
            if horizontal_overlap < 1.0 and vertical_overlap < 1.0:
                continue
            candidate_union = expanded | drawing_rect
            if not allowed.contains(candidate_union):
                continue
            expanded = candidate_union
            changed = True
        if not changed:
            break

    if (
        abs(expanded.x0 - base.x0) < 0.5
        and abs(expanded.y0 - base.y0) < 0.5
        and abs(expanded.x1 - base.x1) < 0.5
        and abs(expanded.y1 - base.y1) < 0.5
    ):
        return base

    padding = max(4.0, min(8.0, 0.012 * min(base.width, base.height)))
    expanded = fitz.Rect(
        max(page_rect.x0, expanded.x0 - padding),
        max(page_rect.y0, expanded.y0 - padding),
        min(page_rect.x1, expanded.x1 + padding),
        min(page_rect.y1, expanded.y1 + padding),
    )
    expanded = _clip_expansion_away_from_caption(expanded, base, caption, page_number)
    return expanded

def expand_tiled_figure_rect_with_nearby_text(
    page: fitz.Page,
    figure_rect: fitz.Rect,
    caption: Caption,
    source_block_indices: Collection[int],
) -> fitz.Rect:
    """Backward-compatible alias for the generalized context expansion."""
    return expand_figure_rect_with_nearby_content(
        page,
        figure_rect,
        caption,
        source_block_indices,
    )

def extract_native_panel_labels(
    page: fitz.Page,
    figure_rect: fitz.Rect,
    expected_labels: list[str],
    dpi: int,
    figure_pixmap: fitz.Pixmap,
) -> list[dict[str, Any]]:
    """Locate expected panel letters in the PDF text layer and map them to crop pixels."""
    if not expected_labels or page.rotation != 0:
        return []

    expected_upper = {label.upper() for label in expected_labels}
    candidates: dict[str, list[dict[str, Any]]] = {
        label.upper(): [] for label in expected_labels
    }
    # Exact one-letter PDF words avoid matching A/B inside axis titles or legends.
    for word in page.get_text("words", clip=figure_rect):
        if len(word) < 5:
            continue
        value = str(word[4]).strip().strip("().:;")
        if len(value) != 1 or value.upper() not in expected_upper:
            continue
        bbox = tuple(float(item) for item in word[:4])
        candidates[value.upper()].append(
            {
                "label": value.upper(),
                "bbox_pt": list(bbox),
                "font_size_pt": bbox[3] - bbox[1],
                "font": "",
                "bold": False,
            }
        )

    selected: list[dict[str, Any]] = []
    render_matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    for expected in expected_labels:
        options = candidates.get(expected.upper(), [])
        if not options:
            continue
        best = max(
            options,
            key=lambda item: (
                item["font_size_pt"],
                -item["bbox_pt"][1],
                -item["bbox_pt"][0],
            ),
        )
        x0, y0, x1, y1 = best["bbox_pt"]
        best["label"] = expected
        label_pixmap = page.get_pixmap(
            matrix=render_matrix,
            clip=fitz.Rect(x0, y0, x1, y1),
            alpha=False,
        )
        best["bbox_px"] = [
            int(label_pixmap.x - figure_pixmap.x),
            int(label_pixmap.y - figure_pixmap.y),
            int(label_pixmap.x - figure_pixmap.x + label_pixmap.width),
            int(label_pixmap.y - figure_pixmap.y + label_pixmap.height),
        ]
        best["anchor_px"] = [
            0.5 * (best["bbox_px"][0] + best["bbox_px"][2]),
            0.5 * (best["bbox_px"][1] + best["bbox_px"][3]),
        ]
        selected.append(best)
    return selected

def rects_almost_equal(
    first: fitz.Rect,
    second: fitz.Rect,
    tolerance: float = 0.75,
) -> bool:
    return max(
        abs(first.x0 - second.x0),
        abs(first.y0 - second.y0),
        abs(first.x1 - second.x1),
        abs(first.y1 - second.y1),
    ) <= tolerance

def quality_flags_require_review(flags: Collection[str]) -> bool:
    """Distinguish diagnostic provenance flags from actual review failures."""
    informational = {
        "caption_guided_component_merge",
        "native_context_expanded",
        "caption_spans_pages",
        "cross_page_caption_match",
        "compact_scheme_candidate",
    }
    return any(flag not in informational for flag in flags)

def build_figure_id(
    page_number: int,
    block_index: int,
    caption: Caption | None,
    used_ids: set[str],
) -> str:
    if caption is None:
        prefix = "figure"
    elif normalize_caption_kind(caption.caption_kind) == "scheme":
        prefix = f"scheme{caption.figure_number}"
    else:
        prefix = f"fig{caption.figure_number}"
    base = sanitize_name(f"{prefix}_p{page_number:02d}_b{block_index:02d}")
    candidate = base
    suffix = 2
    while candidate in used_ids:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used_ids.add(candidate)
    return candidate

def write_metadata(result: ExtractionResult) -> None:
    result.output_dir.mkdir(parents=True, exist_ok=True)
    result.metadata_path.write_text(
        json.dumps(result.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

def expand_tiled_figure_rect_with_nearby_text_conservative(
    page: fitz.Page,
    figure_rect: fitz.Rect,
    caption: Caption,
    source_block_indices: Collection[int],
) -> fitz.Rect:
    """Include short native labels placed just outside a tiled raster figure."""
    if len(source_block_indices) < 4:
        return fitz.Rect(figure_rect)

    horizontal_margin = max(12.0, min(30.0, 0.06 * figure_rect.width))
    vertical_margin = max(12.0, min(30.0, 0.06 * figure_rect.height))
    nearby: list[fitz.Rect] = []

    for block in collect_text_blocks(page, page.number + 1):
        text = block.text.strip()
        if not text or CAPTION_START_RE.match(text):
            continue
        word_count = len(text.split())
        if len(text) > 160 or word_count > 18:
            continue

        block_rect = fitz.Rect(block.bbox_pt)
        if (
            caption.page_number == page.number + 1
            and block_rect.y0 >= caption.bbox_pt[1] - 2.0
        ):
            continue

        horizontal_gap = max(
            figure_rect.x0 - block_rect.x1,
            block_rect.x0 - figure_rect.x1,
            0.0,
        )
        vertical_gap = max(
            figure_rect.y0 - block_rect.y1,
            block_rect.y0 - figure_rect.y1,
            0.0,
        )
        horizontal_overlap = max(
            0.0,
            min(figure_rect.x1, block_rect.x1)
            - max(figure_rect.x0, block_rect.x0),
        )
        vertical_overlap = max(
            0.0,
            min(figure_rect.y1, block_rect.y1)
            - max(figure_rect.y0, block_rect.y0),
        )
        if (
            horizontal_gap > horizontal_margin
            or vertical_gap > vertical_margin
            or (horizontal_overlap < 1.0 and vertical_overlap < 1.0)
        ):
            continue
        nearby.append(block_rect)

    if not nearby:
        return fitz.Rect(figure_rect)

    expanded = fitz.Rect(figure_rect)
    for block_rect in nearby:
        expanded |= block_rect
    raw_expanded = fitz.Rect(expanded)
    did_expand = (
        raw_expanded.x0 < figure_rect.x0 - 0.5
        or raw_expanded.y0 < figure_rect.y0 - 0.5
        or raw_expanded.x1 > figure_rect.x1 + 0.5
        or raw_expanded.y1 > figure_rect.y1 + 0.5
    )
    if not did_expand:
        return fitz.Rect(figure_rect)

    padding = max(4.0, min(8.0, 0.015 * min(
        figure_rect.width,
        figure_rect.height,
    )))
    expanded = fitz.Rect(
        max(page.rect.x0, raw_expanded.x0 - padding),
        max(page.rect.y0, raw_expanded.y0 - padding),
        min(page.rect.x1, raw_expanded.x1 + padding),
        min(page.rect.y1, raw_expanded.y1 + padding),
    )
    if caption.page_number == page.number + 1:
        expanded.y1 = min(expanded.y1, caption.bbox_pt[1] - 2.0)
    return expanded
