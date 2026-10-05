from __future__ import annotations

"""High-precision hybrid PDF whole-figure extractor.

This is the single production entry for whole-figure extraction. It combines
conservative caption ownership with Figure/Scheme recognition, multi-block
completion, native text/vector expansion, and DPI-controlled rendering.

The design favours precision over speculative cross-page recall:

* body references such as ``Figure 3B shows ...`` are hard-rejected;
* the legacy same-page / guarded-previous-page matcher owns figure identity;
* V3 context expansion is bounded by the raster plus exact expected labels;
* nearly identical page regions cannot be emitted for multiple captions.
"""

import argparse
import math
import re
from pathlib import Path
from typing import Any, Collection

import fitz  # PyMuPDF

from . import captions as caption_ops
from . import matching as matching_ops
from . import models as model_types
from . import page_context as page_ops
from . import rendering as rendering_ops


DEFAULT_DPI = model_types.DEFAULT_DPI
DEFAULT_MIN_WIDTH_PT = model_types.DEFAULT_MIN_WIDTH_PT
DEFAULT_MIN_HEIGHT_PT = model_types.DEFAULT_MIN_HEIGHT_PT
DEFAULT_CAPTION_MERGE_GAP_PT = model_types.DEFAULT_CAPTION_MERGE_GAP_PT
DEFAULT_SCHEME_MIN_WIDTH_PT = model_types.DEFAULT_SCHEME_MIN_WIDTH_PT
DEFAULT_SCHEME_MIN_HEIGHT_PT = model_types.DEFAULT_SCHEME_MIN_HEIGHT_PT

Caption = model_types.Caption
CaptionSegment = model_types.CaptionSegment
ExtractedFigure = model_types.ExtractedFigure
ExtractionResult = model_types.ExtractionResult
ImageBlockCandidate = model_types.ImageBlockCandidate
RenderItem = model_types.RenderItem
BBox = model_types.BBox
sanitize_name = page_ops.sanitize_name

REFERENTIAL_CAPTION_BODY_RE = re.compile(
    r"^(?:shows?|showed|presents?|illustrates?|depicts?|demonstrates?|"
    r"indicates?|compares?|summarizes?|describes?|reveals?|suggests?)\b",
    re.IGNORECASE,
)


def _caption_rank(caption: Caption) -> tuple[float, int, int, int]:
    """Rank already-filtered formal-caption candidates deterministically."""
    match = caption_ops.CAPTION_START_RE.match(caption.text)
    if match is None:
        return (-math.inf, 0, 0, 0)
    delimiter = page_ops.clean_text(match.group("delimiter"))
    body = page_ops.clean_text(match.group("body"))
    score = 0.0
    if delimiter:
        score += 6.0
    if 8 <= len(body) <= 1600:
        score += 2.0
    if caption.bold:
        score += 0.4
    if caption.font_size_pt > 0:
        score += 0.2
    # Cross-page continuation is useful metadata but should never outrank a
    # complete same-page caption merely because it absorbed more text.
    page_span = caption.end_page_number - caption.page_number
    score -= 0.75 * page_span
    return score, -page_span, len(caption.text), -caption.page_number


def is_formal_caption(caption: Caption) -> bool:
    """Hard precision gate for Figure/Scheme captions.

    Numeric captions without punctuation remain supported because many journal
    PDFs use ``Figure 1 Description``.  Alphanumeric identifiers are accepted
    only with explicit caption punctuation; this rejects body references such as
    ``Figure 3B shows`` while retaining a genuine ``Figure 3B. Description``.
    """
    match = caption_ops.CAPTION_START_RE.match(caption.text)
    if match is None:
        return False
    number = model_types.normalize_figure_number(match.group("number"))
    delimiter = page_ops.clean_text(match.group("delimiter"))
    body = page_ops.clean_text(match.group("body"))
    if not number or not body:
        return False
    if REFERENTIAL_CAPTION_BODY_RE.match(body) and not delimiter:
        return False
    if not re.fullmatch(r"\d+", number) and not delimiter:
        return False
    # Very short unpunctuated fragments are normally cross-references or running
    # headers rather than captions.
    if not delimiter and len(body.split()) < 3:
        return False
    return True


def select_formal_captions(captions: list[Caption]) -> list[Caption]:
    """Keep one high-confidence formal caption per kind/number identity."""
    selected: dict[tuple[str, str], Caption] = {}
    for caption in captions:
        if not is_formal_caption(caption):
            continue
        key = caption.identity
        current = selected.get(key)
        if current is None or _caption_rank(caption) > _caption_rank(current):
            selected[key] = caption
    return sorted(
        selected.values(),
        key=lambda item: (item.page_number, item.bbox_pt[1], item.bbox_pt[0]),
    )


def collect_formal_captions(
    document: fitz.Document,
    *,
    pages: set[int] | None,
    caption_merge_gap_pt: float,
) -> list[Caption]:
    """Use V3's rich text collection, followed by the hybrid precision gate."""
    return select_formal_captions(
        caption_ops.collect_captions(
            document,
            pages=pages,
            caption_merge_gap_pt=caption_merge_gap_pt,
        )
    )


def _exact_expected_label_rects(
    page: fitz.Page,
    base_rect: fitz.Rect,
    caption: Caption,
    all_captions: Collection[Caption],
) -> list[fitz.Rect]:
    """Find exact expected panel letters around a raster before final clipping."""
    expected_labels = caption_ops.parse_expected_panel_labels(caption.text)
    if not expected_labels or page.rotation != 0:
        return []
    expected = {label.upper() for label in expected_labels}
    page_rect = page.rect
    search_limit = 36.0
    search_rect = fitz.Rect(
        max(page_rect.x0, base_rect.x0 - search_limit),
        max(page_rect.y0, base_rect.y0 - search_limit),
        min(page_rect.x1, base_rect.x1 + search_limit),
        min(page_rect.y1, base_rect.y1 + search_limit),
    )
    prose_rects = [
        fitz.Rect(block.bbox_pt)
        for block in page_ops.collect_text_blocks(page, page.number + 1)
        if len(block.text) >= 85 and len(block.text.split()) >= 12
    ]
    candidates: dict[str, list[fitz.Rect]] = {label: [] for label in expected}
    for word in page.get_text("words", clip=search_rect):
        if len(word) < 5:
            continue
        value = str(word[4]).strip().strip("()[]{}.:; ")
        if len(value) != 1 or value.upper() not in expected:
            continue
        rect = fitz.Rect(tuple(float(value) for value in word[:4]))
        # A standalone "A" inside an article sentence is still returned as one
        # PDF word. Reject words covered by a paragraph-sized text block.
        if any(
            (rect & prose_rect).get_area() >= 0.80 * max(1.0, rect.get_area())
            for prose_rect in prose_rects
        ):
            continue
        if any(
            rendering_ops._caption_rect_intersects(
                rect,
                other,
                page.number + 1,
                tolerance=2.0,
            )
            for other in all_captions
        ):
            continue
        # Top-level native labels are normally substantially larger than body
        # glyphs; retain smaller labels only when they are immediately adjacent.
        gap_x = rendering_ops._rect_axis_gap(base_rect, rect, horizontal=True)
        gap_y = rendering_ops._rect_axis_gap(base_rect, rect, horizontal=False)
        if max(gap_x, gap_y) > 24.0:
            continue
        if max(rect.width, rect.height) < 7.0 and max(gap_x, gap_y) > 6.0:
            continue
        candidates[value.upper()].append(rect)

    selected: list[fitz.Rect] = []
    for label in expected_labels:
        options = candidates.get(label.upper(), [])
        if not options:
            continue
        selected.append(
            min(
                options,
                key=lambda rect: (
                    rendering_ops._rect_axis_gap(base_rect, rect, horizontal=True)
                    + rendering_ops._rect_axis_gap(base_rect, rect, horizontal=False),
                    -rect.height,
                    rect.y0,
                    rect.x0,
                ),
            )
        )
    return selected


def expand_figure_rect_safely(
    page: fitz.Page,
    figure_rect: fitz.Rect,
    caption: Caption,
    source_block_indices: Collection[int],
    all_captions: Collection[Caption],
) -> fitz.Rect:
    """Apply V3 native-content expansion inside a label-anchored safety envelope."""
    base_rect = fitz.Rect(figure_rect)
    expanded = rendering_ops.expand_figure_rect_with_nearby_content(
        page,
        base_rect,
        caption,
        source_block_indices,
    )
    anchors = _exact_expected_label_rects(
        page,
        base_rect,
        caption,
        all_captions,
    )
    anchor_union = fitz.Rect(base_rect)
    for rect in anchors:
        anchor_union |= rect

    # Enough room for axis titles/ticks connected to a detected panel label, but
    # much less than V3's page-header-reaching transitive expansion.
    horizontal_padding = 18.0 if anchors else 14.0
    # Panel letters are included in ``anchor_union`` already.  A smaller upward
    # margin keeps running headers out, while the larger downward margin retains
    # x-axis tick labels that often sit just below the raster XObject.
    upward_padding = 10.0 if anchors else 12.0
    downward_padding = 18.0 if anchors else 14.0
    page_rect = page.rect
    envelope = fitz.Rect(
        max(page_rect.x0, anchor_union.x0 - horizontal_padding),
        max(page_rect.y0, anchor_union.y0 - upward_padding),
        min(page_rect.x1, anchor_union.x1 + horizontal_padding),
        min(page_rect.y1, anchor_union.y1 + downward_padding),
    )
    clipped = expanded & envelope
    if clipped.is_empty:
        clipped = anchor_union
    else:
        clipped |= anchor_union
    clipped = rendering_ops._clip_expansion_away_from_caption(
        clipped,
        base_rect,
        caption,
        page.number + 1,
    )
    # Every other formal caption is also a hard page-layout boundary. This is
    # essential on dense pages containing several small figures in one row.
    for other in all_captions:
        if other.identity == caption.identity:
            continue
        segments = model_types.caption_segments_on_page(other, page.number + 1)
        for segment in segments:
            other_rect = fitz.Rect(segment.bbox_pt)
            horizontal_overlap = rendering_ops._rect_axis_overlap(
                base_rect,
                other_rect,
                horizontal=True,
            )
            vertical_overlap = rendering_ops._rect_axis_overlap(
                base_rect,
                other_rect,
                horizontal=False,
            )
            if other_rect.y1 <= base_rect.y0 + 4.0 and horizontal_overlap >= 1.0:
                clipped.y0 = max(clipped.y0, other_rect.y1 + 2.0)
            elif other_rect.y0 >= base_rect.y1 - 4.0 and horizontal_overlap >= 1.0:
                clipped.y1 = min(clipped.y1, other_rect.y0 - 2.0)
            elif other_rect.x1 <= base_rect.x0 + 4.0 and vertical_overlap >= 1.0:
                clipped.x0 = max(clipped.x0, other_rect.x1 + 2.0)
            elif other_rect.x0 >= base_rect.x1 - 4.0 and vertical_overlap >= 1.0:
                clipped.x1 = min(clipped.x1, other_rect.x0 - 2.0)
    return fitz.Rect(
        max(page_rect.x0, clipped.x0),
        max(page_rect.y0, clipped.y0),
        min(page_rect.x1, clipped.x1),
        min(page_rect.y1, clipped.y1),
    )


def rect_overlap_metrics(first: fitz.Rect, second: fitz.Rect) -> tuple[float, float]:
    """Return intersection-over-union and smaller-region coverage."""
    intersection = first & second
    if intersection.is_empty:
        return 0.0, 0.0
    intersection_area = intersection.get_area()
    union_area = first.get_area() + second.get_area() - intersection_area
    iou = intersection_area / max(1.0, union_area)
    smaller_coverage = intersection_area / max(
        1.0,
        min(first.get_area(), second.get_area()),
    )
    return float(iou), float(smaller_coverage)


def find_region_conflict(
    items: Collection[RenderItem],
    page_number: int,
    rect: fitz.Rect,
) -> RenderItem | None:
    """Reject duplicate visual regions assigned to different formal captions."""
    for item in items:
        if item.page_number != page_number:
            continue
        iou, smaller_coverage = rect_overlap_metrics(item.rect, rect)
        if iou >= 0.80 or smaller_coverage >= 0.92:
            return item
    return None


def _review_item(
    caption: Caption,
    code: str,
    flags: list[str],
    message: str,
) -> dict[str, Any]:
    return {
        "page_number": caption.page_number,
        "figure_number": caption.figure_number,
        "caption_kind": model_types.normalize_caption_kind(caption.caption_kind),
        "figure_id": None,
        "code": code,
        "quality_flags": flags,
        "message": message,
    }


def _append_unique_render_item(
    render_items: list[RenderItem],
    item: RenderItem,
    review_items: list[dict[str, Any]],
) -> bool:
    conflict = find_region_conflict(render_items, item.page_number, item.rect)
    if conflict is None:
        render_items.append(item)
        return True
    if item.caption is not None:
        owner = conflict.caption
        owner_text = (
            f"{model_types.normalize_caption_kind(owner.caption_kind)} {owner.figure_number}"
            if owner is not None
            else "an unmatched visual"
        )
        review_items.append(
            _review_item(
                item.caption,
                "duplicate_visual_region_rejected",
                ["duplicate_visual_region"],
                f"Candidate region substantially duplicates {owner_text}; duplicate output was rejected.",
            )
        )
    return False


def _match_caption_conservatively(
    document: fitz.Document,
    caption: Caption,
    candidates: list[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
) -> ImageBlockCandidate | None:
    """Delegate ownership to the proven legacy high-precision matcher."""
    matched = matching_ops.match_caption_to_image_block_conservative(
        document,
        caption,
        candidates,
        consumed,
    )
    if matched is not None:
        return matched
    return matching_ops.match_caption_to_previous_page_figure(
        document,
        caption,
        candidates,
        consumed,
    )


def _source_image_clipped_by_page(page: fitz.Page, figure_rect: fitz.Rect) -> bool:
    """Detect an XObject extending beyond the printable page boundary."""
    if figure_rect.y1 < page.rect.y1 - 3.0:
        return False
    for image in page.get_images(full=True):
        for placement in page.get_image_rects(image[0]):
            if placement.y1 <= page.rect.y1 + 2.0:
                continue
            visible = placement & page.rect
            if not visible.is_empty and (visible & figure_rect).get_area() >= 0.70 * visible.get_area():
                return True
    return False


def _find_conservative_caption_crop(
    document: fitz.Document,
    caption: Caption,
    min_height_pt: float,
) -> tuple[int, fitz.Rect | None]:
    """Use legacy figure fallback; keep Scheme fallback on its caption page only."""
    if model_types.normalize_caption_kind(caption.caption_kind) == "figure":
        return matching_ops.find_caption_crop_conservative(document, caption, min_height_pt)

    page = document[caption.page_number - 1]
    rect = matching_ops.find_caption_figure_rect(page, caption, min_height_pt)
    if not matching_ops._caption_crop_is_safe(page, rect, caption, min_height_pt):
        return caption.page_number, None
    return caption.page_number, rect


def complete_caption_crop_with_visual_anchors(
    document: fitz.Document,
    caption: Caption,
    crop_page_number: int,
    crop_rect: fitz.Rect,
    candidates: Collection[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
    all_captions: Collection[Caption],
) -> tuple[fitz.Rect, tuple[ImageBlockCandidate, ...]]:
    """Recover an upper raster omitted from a caption-relative fallback crop.

    Some journal figures combine an embedded raster at the top with native PDF
    charts below it. The legacy fallback can mistake long legend text inside the
    charts for article prose and move the crop top below the raster. Reconnect an
    omitted raster only when it reveals additional expected native panel letters.
    The label-gain requirement prevents a nearby figure from being absorbed.
    """
    expected_labels = caption_ops.parse_expected_panel_labels(caption.text)
    if len(expected_labels) < 2:
        return fitz.Rect(crop_rect), ()
    if crop_page_number <= 0 or crop_page_number > len(document):
        return fitz.Rect(crop_rect), ()

    page = document[crop_page_number - 1]
    current = fitz.Rect(crop_rect)
    current_anchor_count = len(
        _exact_expected_label_rects(page, current, caption, all_captions)
    )
    merged: list[ImageBlockCandidate] = []
    max_vertical_gap = max(150.0, 0.24 * page.rect.height)

    candidate_pool = list(candidates)
    represented_indices = {
        block_index
        for candidate in candidate_pool
        if candidate.page_number == crop_page_number
        for block_index in model_types.image_block_indices(candidate)
    }
    # The normal candidate collector filters by the final cluster height. A
    # shallow but wide western-blot panel can therefore be absent even though it
    # is the missing top panel of a larger native-PDF figure. Include meaningful
    # raw image blocks here; the expected-label gain below remains the acceptance
    # gate, so these relaxed candidates cannot be emitted on their own.
    try:
        page_dict = page.get_text("dict")
    except Exception:
        page_dict = {}
    for block_index, block in enumerate(page_dict.get("blocks", []), 1):
        if (
            block_index in represented_indices
            or block.get("type") != 1
            or "bbox" not in block
        ):
            continue
        raw_rect = fitz.Rect(block["bbox"])
        if (
            raw_rect.width < model_types.MIN_CLUSTER_BLOCK_SIDE_PT
            or raw_rect.height < model_types.MIN_CLUSTER_BLOCK_SIDE_PT
            or raw_rect.get_area() < model_types.MIN_CLUSTER_BLOCK_AREA_PT2
        ):
            continue
        candidate_pool.append(
            ImageBlockCandidate(
                page_number=crop_page_number,
                block_index=block_index,
                bbox_pt=tuple(float(value) for value in raw_rect),
                member_block_indices=(block_index,),
            )
        )

    eligible: list[tuple[float, ImageBlockCandidate]] = []
    for candidate in candidate_pool:
        if (
            candidate.page_number != crop_page_number
            or model_types.image_block_is_consumed(candidate, consumed)
        ):
            continue
        candidate_rect = fitz.Rect(candidate.bbox_pt)
        if candidate_rect.y1 > current.y0 + 8.0:
            continue
        vertical_gap = max(0.0, current.y0 - candidate_rect.y1)
        if vertical_gap > max_vertical_gap:
            continue
        overlap = rendering_ops._rect_axis_overlap(current, candidate_rect, horizontal=True)
        overlap_ratio = overlap / max(
            1.0,
            min(current.width, candidate_rect.width),
        )
        if overlap_ratio < 0.65 or candidate_rect.width < 0.45 * current.width:
            continue

        proposed = current | candidate_rect
        separated = False
        for other in all_captions:
            if other.identity == caption.identity:
                continue
            for segment in model_types.caption_segments_on_page(other, crop_page_number):
                separator = fitz.Rect(segment.bbox_pt)
                if separator.y0 < candidate_rect.y1 - 3.0:
                    continue
                if separator.y1 > current.y0 + 3.0:
                    continue
                separator_overlap = rendering_ops._rect_axis_overlap(
                    proposed,
                    separator,
                    horizontal=True,
                )
                if separator_overlap >= 0.30 * min(proposed.width, separator.width):
                    separated = True
                    break
            if separated:
                break
        if not separated:
            eligible.append((vertical_gap, candidate))

    for _, candidate in sorted(eligible, key=lambda item: item[0]):
        candidate_rect = fitz.Rect(candidate.bbox_pt)
        proposed = current | candidate_rect
        proposed_anchor_count = len(
            _exact_expected_label_rects(
                page,
                proposed,
                caption,
                all_captions,
            )
        )
        if proposed_anchor_count <= current_anchor_count:
            continue
        current = proposed
        current_anchor_count = proposed_anchor_count
        merged.append(candidate)
        if current_anchor_count >= len(expected_labels):
            break

    if not merged:
        return fitz.Rect(crop_rect), ()

    source_indices = tuple(
        sorted(
            {
                block_index
                for candidate in merged
                for block_index in model_types.image_block_indices(candidate)
            }
        )
    )
    completed = expand_figure_rect_safely(
        page,
        current,
        caption,
        source_indices,
        all_captions,
    )
    completed = _clip_completed_crop_away_from_prose_columns(
        page,
        completed,
    )
    return completed, tuple(merged)


def _clip_completed_crop_away_from_prose_columns(
    page: fitz.Page,
    crop_rect: fitz.Rect,
) -> fitz.Rect:
    """Trim a completed narrow-column crop before adjacent article prose."""
    clipped = fitz.Rect(crop_rect)
    for block in page_ops.collect_text_blocks(page, page.number + 1):
        if len(block.text) < 120 or len(block.text.split()) < 18:
            continue
        prose_rect = fitz.Rect(block.bbox_pt)
        vertical_overlap = rendering_ops._rect_axis_overlap(
            clipped,
            prose_rect,
            horizontal=False,
        )
        if vertical_overlap < 24.0:
            continue
        # The prose column may overlap a padded crop by a few points. Require
        # its leading edge to sit in the outer quarter of the current crop.
        if (
            prose_rect.x0 >= clipped.x0 + 0.75 * clipped.width
            and prose_rect.x0 <= clipped.x1 + 12.0
        ):
            clipped.x1 = min(clipped.x1, prose_rect.x0 - 4.0)
        elif (
            prose_rect.x1 <= clipped.x1 - 0.75 * clipped.width
            and prose_rect.x1 >= clipped.x0 - 12.0
        ):
            clipped.x0 = max(clipped.x0, prose_rect.x1 + 4.0)
    return clipped


def extract_pdf_figures(
    pdf_path: str | Path,
    output_dir: str | Path | None = None,
    *,
    dpi: int = DEFAULT_DPI,
    min_width_pt: float = DEFAULT_MIN_WIDTH_PT,
    min_height_pt: float = DEFAULT_MIN_HEIGHT_PT,
    scheme_min_width_pt: float = DEFAULT_SCHEME_MIN_WIDTH_PT,
    scheme_min_height_pt: float = DEFAULT_SCHEME_MIN_HEIGHT_PT,
    pages: Collection[int] | None = None,
    caption_merge_gap_pt: float = DEFAULT_CAPTION_MERGE_GAP_PT,
    overwrite: bool = True,
) -> ExtractionResult:
    """Extract complete 300-DPI Figure/Scheme assets with conservative ownership."""
    pdf_path = Path(pdf_path).expanduser().resolve()
    if output_dir is None:
        output_dir = Path("outputs") / "pdf_figures_hybrid_300dpi" / page_ops.sanitize_name(pdf_path.stem)
    output_dir = Path(output_dir).expanduser().resolve()
    page_filter = set(pages) if pages else None
    page_ops.validate_inputs(
        pdf_path,
        dpi,
        min_width_pt,
        min_height_pt,
        page_filter,
        scheme_min_width_pt=scheme_min_width_pt,
        scheme_min_height_pt=scheme_min_height_pt,
    )

    figures_dir = output_dir / "figures"
    metadata_path = output_dir / "figures_metadata.json"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for stale_path in figures_dir.glob("*.png"):
            stale_path.unlink()

    captions: list[Caption] = []
    figures: list[ExtractedFigure] = []
    review_items: list[dict[str, Any]] = []
    used_ids: set[str] = set()
    zoom = dpi / 72.0

    with fitz.open(pdf_path) as document:
        captions = collect_formal_captions(
            document,
            pages=page_filter,
            caption_merge_gap_pt=caption_merge_gap_pt,
        )
        image_blocks = matching_ops.collect_image_block_candidates(
            document,
            page_filter,
            min_width_pt,
            min_height_pt,
            captions,
            scheme_min_width_pt=scheme_min_width_pt,
            scheme_min_height_pt=scheme_min_height_pt,
        )
        consumed_blocks: set[tuple[int, int]] = set()
        render_items: list[RenderItem] = []

        for caption_index, caption in enumerate(captions, 1):
            matched = _match_caption_conservatively(
                document,
                caption,
                image_blocks,
                consumed_blocks,
            )
            if matched is not None:
                grouped = matching_ops.merge_caption_owned_figure_parts(
                    document,
                    caption,
                    matched,
                    image_blocks,
                    consumed_blocks,
                    captions,
                )
                source_indices = model_types.image_block_indices(grouped)
                initial_rect = fitz.Rect(grouped.bbox_pt)
                matched_page = document[grouped.page_number - 1]
                matched_rect = expand_figure_rect_safely(
                    matched_page,
                    initial_rect,
                    caption,
                    source_indices,
                    captions,
                )
                flags: list[str] = []
                if grouped.page_number != caption.page_number:
                    flags.append("cross_page_caption_match")
                if _source_image_clipped_by_page(matched_page, initial_rect):
                    flags.append("source_image_clipped_by_page")
                if caption.end_page_number != caption.page_number:
                    flags.append("caption_spans_pages")
                if set(source_indices) != set(model_types.image_block_indices(matched)):
                    flags.append("caption_guided_component_merge")
                if not rendering_ops.rects_almost_equal(initial_rect, matched_rect):
                    flags.append("native_context_expanded")
                if (
                    model_types.normalize_caption_kind(caption.caption_kind) == "scheme"
                    and (
                        initial_rect.width < min_width_pt
                        or initial_rect.height < min_height_pt
                    )
                ):
                    flags.append("compact_scheme_candidate")
                item = RenderItem(
                    page_number=grouped.page_number,
                    block_index=grouped.block_index,
                    rect=matched_rect,
                    caption=caption,
                    source_block_indices=source_indices,
                    quality_flags=tuple(flags),
                )
                if _append_unique_render_item(render_items, item, review_items):
                    model_types.consume_image_block(grouped, consumed_blocks)
                continue

            effective_height = caption_ops.effective_caption_min_height(
                caption,
                min_height_pt,
                scheme_min_height_pt,
            )
            crop_page_number, crop_rect = _find_conservative_caption_crop(
                document,
                caption,
                effective_height,
            )
            if crop_rect is None or (page_filter and crop_page_number not in page_filter):
                review_items.append(
                    _review_item(
                        caption,
                        "caption_not_safely_matched",
                        ["unsafe_caption_crop_rejected"],
                        "Caption could not be matched to a safe visual region.",
                    )
                )
                continue

            crop_rect, anchored_blocks = complete_caption_crop_with_visual_anchors(
                document,
                caption,
                crop_page_number,
                crop_rect,
                image_blocks,
                consumed_blocks,
                captions,
            )

            covered_indices: list[int] = [
                block_index
                for block in anchored_blocks
                for block_index in model_types.image_block_indices(block)
            ]
            covered_blocks: list[ImageBlockCandidate] = []
            for block in image_blocks:
                if (
                    model_types.image_block_is_consumed(block, consumed_blocks)
                    or block.page_number != crop_page_number
                ):
                    continue
                if page_ops.bbox_coverage_ratio(block.bbox_pt, tuple(crop_rect)) >= 0.65:
                    covered_blocks.append(block)
                    covered_indices.extend(model_types.image_block_indices(block))
            item = RenderItem(
                page_number=crop_page_number,
                block_index=10000 + caption_index,
                rect=crop_rect,
                caption=caption,
                source_block_indices=tuple(sorted(set(covered_indices))),
                quality_flags=tuple(
                    flag
                    for flag, enabled in (
                        ("cross_page_caption_crop", crop_page_number != caption.page_number),
                        ("caption_spans_pages", caption.end_page_number != caption.page_number),
                        ("caption_crop_visual_anchor_merge", bool(anchored_blocks)),
                        ("native_context_expanded", bool(anchored_blocks)),
                    )
                    if enabled
                ),
            )
            if _append_unique_render_item(render_items, item, review_items):
                for block in covered_blocks:
                    model_types.consume_image_block(block, consumed_blocks)

        for block in image_blocks:
            if model_types.image_block_is_consumed(block, consumed_blocks):
                continue
            item = RenderItem(
                page_number=block.page_number,
                block_index=block.block_index,
                rect=fitz.Rect(block.bbox_pt),
                caption=None,
                source_block_indices=model_types.image_block_indices(block),
            )
            _append_unique_render_item(render_items, item, review_items)

        render_items.sort(key=lambda item: (item.page_number, item.rect.y0, item.rect.x0))
        for item in render_items:
            page = document[item.page_number - 1]
            caption = item.caption
            figure_id = rendering_ops.build_figure_id(
                item.page_number,
                item.block_index,
                caption,
                used_ids,
            )
            output_path = figures_dir / f"{figure_id}.png"
            if not overwrite:
                output_path = page_ops.unique_path(output_path)
                figure_id = output_path.stem

            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(zoom, zoom),
                clip=item.rect,
                alpha=False,
            )
            if hasattr(pixmap, "set_dpi"):
                pixmap.set_dpi(dpi, dpi)
            pixmap.save(output_path)

            expected_labels = (
                caption_ops.parse_expected_panel_labels(caption.text) if caption else []
            )
            if caption is not None and 10000 <= item.block_index < 100000:
                method = "caption_crop"
                flags = ["synthetic_caption_crop", *item.quality_flags]
            elif caption is not None and len(item.source_block_indices) > 1:
                method = "tiled_raster"
                flags = list(item.quality_flags)
            elif caption is not None:
                method = "embedded_raster"
                flags = list(item.quality_flags)
            elif len(item.source_block_indices) > 1:
                method = "unmatched_tiled_raster"
                flags = ["caption_unmatched", *item.quality_flags]
            else:
                method = "unmatched_raster"
                flags = ["caption_unmatched", *item.quality_flags]
            flags = list(dict.fromkeys(flags))
            native_labels = rendering_ops.extract_native_panel_labels(
                page,
                item.rect,
                expected_labels,
                dpi,
                pixmap,
            )
            figures.append(
                ExtractedFigure(
                    figure_id=figure_id,
                    page_number=item.page_number,
                    block_index=item.block_index,
                    source_block_indices=list(item.source_block_indices),
                    bbox_pt=tuple(float(value) for value in item.rect),
                    image_path=output_path,
                    width_px=int(pixmap.width),
                    height_px=int(pixmap.height),
                    dpi=dpi,
                    caption=caption.text if caption else "",
                    caption_figure_number=caption.figure_number if caption else None,
                    caption_kind=(
                        model_types.normalize_caption_kind(caption.caption_kind) if caption else None
                    ),
                    caption_label=caption.caption_label if caption else None,
                    caption_bbox_pt=caption.bbox_pt if caption else None,
                    caption_page_number=caption.page_number if caption else None,
                    caption_end_page_number=(caption.end_page_number if caption else None),
                    caption_segments=(
                        [segment.to_dict() for segment in caption.segments_or_primary()]
                        if caption
                        else []
                    ),
                    expected_panel_labels=expected_labels,
                    native_panel_labels=native_labels,
                    extraction_method=method,
                    quality_flags=flags,
                    needs_review=rendering_ops.quality_flags_require_review(flags),
                )
            )

    result = ExtractionResult(
        pdf_path=pdf_path,
        output_dir=output_dir,
        figures_dir=figures_dir,
        metadata_path=metadata_path,
        dpi=dpi,
        min_width_pt=min_width_pt,
        min_height_pt=min_height_pt,
        scheme_min_width_pt=scheme_min_width_pt,
        scheme_min_height_pt=scheme_min_height_pt,
        pages=page_filter,
        figures=figures,
        captions=captions,
        review_items=review_items,
    )
    rendering_ops.write_metadata(result)
    return result


def parse_page_selector(value: str) -> set[int] | None:
    return page_ops.parse_page_selector(value)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract complete PDF Figures/Schemes at 300 DPI using conservative "
            "caption ownership and label-aware native-content expansion."
        )
    )
    parser.add_argument("--pdf", required=True, type=Path, help="Input PDF path.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--min-width-pt", type=float, default=DEFAULT_MIN_WIDTH_PT)
    parser.add_argument("--min-height-pt", type=float, default=DEFAULT_MIN_HEIGHT_PT)
    parser.add_argument(
        "--scheme-min-width-pt",
        type=float,
        default=DEFAULT_SCHEME_MIN_WIDTH_PT,
    )
    parser.add_argument(
        "--scheme-min-height-pt",
        type=float,
        default=DEFAULT_SCHEME_MIN_HEIGHT_PT,
    )
    parser.add_argument("--pages", default="")
    parser.add_argument(
        "--caption-merge-gap-pt",
        type=float,
        default=DEFAULT_CAPTION_MERGE_GAP_PT,
    )
    parser.add_argument("--no-overwrite", action="store_true")
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()
    try:
        result = extract_pdf_figures(
            pdf_path=args.pdf,
            output_dir=args.output_dir,
            dpi=args.dpi,
            min_width_pt=args.min_width_pt,
            min_height_pt=args.min_height_pt,
            scheme_min_width_pt=args.scheme_min_width_pt,
            scheme_min_height_pt=args.scheme_min_height_pt,
            pages=parse_page_selector(args.pages),
            caption_merge_gap_pt=args.caption_merge_gap_pt,
            overwrite=not args.no_overwrite,
        )
    except Exception as exc:
        parser.exit(2, f"error: {exc}\n")

    figures = sum(item.caption_kind == "figure" for item in result.figures)
    schemes = sum(item.caption_kind == "scheme" for item in result.figures)
    unmatched = sum(not item.caption_kind for item in result.figures)
    print(f"PDF: {result.pdf_path}")
    print(f"Output: {result.output_dir}")
    print(f"Visual assets: {len(result.figures)}")
    print(f"Figures: {figures}")
    print(f"Schemes: {schemes}")
    print(f"Unmatched assets: {unmatched}")
    print(f"Captions detected: {len(result.captions)}")
    print(f"Metadata: {result.metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
