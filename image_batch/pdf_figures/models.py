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

"""Stable data models and extraction defaults."""

DEFAULT_DPI = 300
DEFAULT_MIN_WIDTH_PT = 180.0
DEFAULT_MIN_HEIGHT_PT = 120.0
DEFAULT_CAPTION_MERGE_GAP_PT = 8.0
DEFAULT_SCHEME_MIN_WIDTH_PT = 96.0
DEFAULT_SCHEME_MIN_HEIGHT_PT = 54.0
MIN_CLUSTER_BLOCK_SIDE_PT = 24.0
MIN_CLUSTER_BLOCK_AREA_PT2 = 576.0
MAX_CAPTION_GUIDED_VERTICAL_GAP_PT = 240.0
MAX_FIGURE_CONTEXT_EXPANSION_PT = 72.0
MAX_SCHEME_CONTEXT_EXPANSION_PT = 96.0
MAX_CROSS_PAGE_DISTANCE = 1

BBox = tuple[float, float, float, float]

def normalize_caption_kind(value: str) -> str:
    """Return the stable metadata class for a caption label or kind value."""
    compact = re.sub(r"[.\s]+", "", str(value or "")).lower()
    if "scheme" in compact or compact.startswith("sch"):
        return "scheme"
    return "figure"

def normalize_figure_number(number: str) -> str:
    """Backward-compatible caption-number normalizer."""
    return re.sub(r"\s+", "", str(number or "")).upper()

def caption_kind_from_label(label: str) -> str:
    return normalize_caption_kind(label)

@dataclass(slots=True)
class CaptionSegment:
    """One physical caption segment on one PDF page."""

    page_number: int
    text: str
    bbox_pt: BBox

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "text": self.text,
            "bbox_pt": list(self.bbox_pt),
        }

@dataclass(slots=True)
class Caption:
    """A caption text block detected on one PDF page."""

    page_number: int
    figure_number: str
    text: str
    bbox_pt: BBox
    caption_kind: str = "figure"
    caption_label: str = ""
    segments: tuple[CaptionSegment, ...] = ()
    font_size_pt: float = 0.0
    font_name: str = ""
    bold: bool = False

    @property
    def page_numbers(self) -> tuple[int, ...]:
        if self.segments:
            return tuple(dict.fromkeys(segment.page_number for segment in self.segments))
        return (self.page_number,)

    @property
    def end_page_number(self) -> int:
        return max(self.page_numbers)

    def segments_or_primary(self) -> tuple[CaptionSegment, ...]:
        if self.segments:
            return self.segments
        return (
            CaptionSegment(
                page_number=self.page_number,
                text=self.text,
                bbox_pt=self.bbox_pt,
            ),
        )

    @property
    def identity(self) -> tuple[str, str]:
        return (
            normalize_caption_kind(self.caption_kind),
            normalize_figure_number(self.figure_number),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "end_page_number": self.end_page_number,
            "page_numbers": list(self.page_numbers),
            "caption_kind": normalize_caption_kind(self.caption_kind),
            "caption_label": self.caption_label,
            "caption_number": self.figure_number,
            # Backward-compatible key used by existing consumers.
            "figure_number": self.figure_number,
            "text": self.text,
            "bbox_pt": list(self.bbox_pt),
            "segments": [segment.to_dict() for segment in self.segments_or_primary()],
        }

@dataclass(slots=True)
class ExtractedFigure:
    """One PDF image block rendered to a PNG file."""

    figure_id: str
    page_number: int
    block_index: int
    source_block_indices: list[int]
    bbox_pt: BBox
    image_path: Path
    width_px: int
    height_px: int
    dpi: int
    caption: str
    caption_figure_number: str | None
    caption_bbox_pt: BBox | None
    caption_page_number: int | None
    caption_end_page_number: int | None
    caption_segments: list[dict[str, Any]]
    expected_panel_labels: list[str]
    native_panel_labels: list[dict[str, Any]]
    extraction_method: str
    quality_flags: list[str]
    needs_review: bool
    # Appended defaults preserve positional construction used by v2 callers.
    caption_kind: str | None = None
    caption_label: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "figure_id": self.figure_id,
            "page_number": self.page_number,
            "block_index": self.block_index,
            "source_block_indices": self.source_block_indices,
            "bbox_pt": list(self.bbox_pt),
            "image_path": str(self.image_path),
            "width_px": self.width_px,
            "height_px": self.height_px,
            "dpi": self.dpi,
            "caption": self.caption,
            "caption_figure_number": self.caption_figure_number,
            "caption_number": self.caption_figure_number,
            "caption_kind": self.caption_kind,
            "caption_label": self.caption_label,
            "asset_type": self.caption_kind or "unmatched",
            "caption_bbox_pt": list(self.caption_bbox_pt) if self.caption_bbox_pt else None,
            "caption_page_number": self.caption_page_number,
            "caption_end_page_number": self.caption_end_page_number,
            "caption_segments": self.caption_segments,
            "expected_panel_labels": self.expected_panel_labels,
            "native_panel_labels": self.native_panel_labels,
            "extraction_method": self.extraction_method,
            "quality_flags": self.quality_flags,
            "needs_review": self.needs_review,
        }

@dataclass(slots=True)
class ExtractionResult:
    """Return value for the public extraction API."""

    pdf_path: Path
    output_dir: Path
    figures_dir: Path
    metadata_path: Path
    dpi: int
    min_width_pt: float
    min_height_pt: float
    pages: set[int] | None
    figures: list[ExtractedFigure]
    captions: list[Caption]
    review_items: list[dict[str, Any]]
    # Appended defaults preserve positional construction used by v2 callers.
    scheme_min_width_pt: float = DEFAULT_SCHEME_MIN_WIDTH_PT
    scheme_min_height_pt: float = DEFAULT_SCHEME_MIN_HEIGHT_PT

    def to_dict(self) -> dict[str, Any]:
        review_items = list(self.review_items)
        review_items.extend(
            {
                "page_number": figure.page_number,
                "figure_number": figure.caption_figure_number,
                "caption_kind": figure.caption_kind,
                "figure_id": figure.figure_id,
                "code": "visual_extraction_review",
                "quality_flags": figure.quality_flags,
                "message": "Extracted figure or scheme requires review before batch use.",
            }
            for figure in self.figures
            if figure.needs_review
        )
        matched_kind_counts = Counter(
            figure.caption_kind for figure in self.figures if figure.caption_kind
        )
        caption_kind_counts = Counter(
            normalize_caption_kind(caption.caption_kind) for caption in self.captions
        )
        return {
            "pdf_path": str(self.pdf_path),
            "output_dir": str(self.output_dir),
            "figures_dir": str(self.figures_dir),
            "metadata_path": str(self.metadata_path),
            "dpi": self.dpi,
            "min_width_pt": self.min_width_pt,
            "min_height_pt": self.min_height_pt,
            "scheme_min_width_pt": self.scheme_min_width_pt,
            "scheme_min_height_pt": self.scheme_min_height_pt,
            "pages": sorted(self.pages) if self.pages else None,
            "asset_count": len(self.figures),
            # Backward-compatible total count. It now includes Scheme assets.
            "figure_count": len(self.figures),
            "standard_figure_count": matched_kind_counts.get("figure", 0),
            "scheme_count": matched_kind_counts.get("scheme", 0),
            "unmatched_asset_count": sum(
                1 for figure in self.figures if not figure.caption_kind
            ),
            "caption_count": len(self.captions),
            "figure_caption_count": caption_kind_counts.get("figure", 0),
            "scheme_caption_count": caption_kind_counts.get("scheme", 0),
            "review_required_count": len(review_items),
            "review_items": review_items,
            "figures": [figure.to_dict() for figure in self.figures],
            "captions": [caption.to_dict() for caption in self.captions],
        }

@dataclass(slots=True)
class TextBlock:
    page_number: int
    text: str
    bbox_pt: BBox
    font_size_pt: float = 0.0
    font_name: str = ""
    bold: bool = False
    line_count: int = 1

@dataclass(slots=True)
class RenderItem:
    page_number: int
    block_index: int
    rect: fitz.Rect
    caption: Caption | None
    source_block_indices: tuple[int, ...]
    quality_flags: tuple[str, ...] = field(default_factory=tuple)

@dataclass(slots=True)
class ImageBlockCandidate:
    page_number: int
    block_index: int
    bbox_pt: BBox
    member_block_indices: tuple[int, ...] = ()

def image_block_indices(candidate: ImageBlockCandidate) -> tuple[int, ...]:
    """Return every PDF image-block index represented by one candidate."""
    return candidate.member_block_indices or (candidate.block_index,)

def image_block_is_consumed(
    candidate: ImageBlockCandidate,
    consumed: set[tuple[int, int]],
) -> bool:
    return any(
        (candidate.page_number, block_index) in consumed
        for block_index in image_block_indices(candidate)
    )

def consume_image_block(
    candidate: ImageBlockCandidate,
    consumed: set[tuple[int, int]],
) -> None:
    consumed.update(
        (candidate.page_number, block_index)
        for block_index in image_block_indices(candidate)
    )

def caption_segments_on_page(caption: Caption, page_number: int) -> list[CaptionSegment]:
    """Return the physical caption segments located on ``page_number``."""
    return [
        segment
        for segment in caption.segments_or_primary()
        if segment.page_number == page_number
    ]
