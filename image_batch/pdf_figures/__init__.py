"""Public API for complete PDF Figure/Scheme extraction."""

from .models import (
    DEFAULT_CAPTION_MERGE_GAP_PT,
    DEFAULT_DPI,
    DEFAULT_MIN_HEIGHT_PT,
    DEFAULT_MIN_WIDTH_PT,
    DEFAULT_SCHEME_MIN_HEIGHT_PT,
    DEFAULT_SCHEME_MIN_WIDTH_PT,
    Caption,
    CaptionSegment,
    ExtractedFigure,
    ExtractionResult,
    ImageBlockCandidate,
    RenderItem,
)
from .page_context import parse_page_selector, sanitize_name
from .pdf_figure_extractor import extract_pdf_figures

__all__ = [
    "Caption",
    "CaptionSegment",
    "ExtractedFigure",
    "ExtractionResult",
    "ImageBlockCandidate",
    "RenderItem",
    "DEFAULT_CAPTION_MERGE_GAP_PT",
    "DEFAULT_DPI",
    "DEFAULT_MIN_HEIGHT_PT",
    "DEFAULT_MIN_WIDTH_PT",
    "DEFAULT_SCHEME_MIN_HEIGHT_PT",
    "DEFAULT_SCHEME_MIN_WIDTH_PT",
    "extract_pdf_figures",
    "parse_page_selector",
    "sanitize_name",
]
