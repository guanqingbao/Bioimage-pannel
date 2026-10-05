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

"""Raster clustering, caption ownership, safe fallbacks, and conservative matching."""

from .captions import CAPTION_START_RE, caption_identity_from_match
from .models import (
    BBox,
    Caption,
    CaptionSegment,
    DEFAULT_SCHEME_MIN_HEIGHT_PT,
    DEFAULT_SCHEME_MIN_WIDTH_PT,
    ImageBlockCandidate,
    MAX_CAPTION_GUIDED_VERTICAL_GAP_PT,
    MIN_CLUSTER_BLOCK_AREA_PT2,
    MIN_CLUSTER_BLOCK_SIDE_PT,
    caption_segments_on_page,
    image_block_indices,
    image_block_is_consumed,
    normalize_caption_kind,
)
from .page_context import (
    bbox_center_x,
    bbox_center_y,
    bbox_coverage_ratio,
    clean_text,
    collect_text_blocks,
    horizontal_overlap_ratio,
    looks_like_heading,
    union_bbox,
)
from .rendering import (
    _caption_rect_intersects,
    _rect_axis_gap,
    _rect_axis_overlap,
    expand_figure_rect_with_nearby_content,
)

# Frozen high-precision Figure-only rules retained from the validated legacy matcher.
LEGACY_CAPTION_START_RE = re.compile(
    r"^(?:fig(?:ure)?\.?)\s*"
    r"(?P<number>[A-Za-z]?\d+[A-Za-z0-9-]*(?:\.\d+[A-Za-z0-9-]*)*)"
    r"\s*[.:]?\s*(?P<body>.*)",
    re.IGNORECASE | re.DOTALL,
)

def _candidate_rect(candidate: ImageBlockCandidate) -> fitz.Rect:
    return fitz.Rect(candidate.bbox_pt)

def _rect_union(left: fitz.Rect, right: fitz.Rect) -> fitz.Rect:
    return fitz.Rect(
        min(left.x0, right.x0),
        min(left.y0, right.y0),
        max(left.x1, right.x1),
        max(left.y1, right.y1),
    )

def _axis_overlap_ratio(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    overlap = max(0.0, min(first_end, second_end) - max(first_start, second_start))
    span = max(1.0, min(first_end - first_start, second_end - second_start))
    return overlap / span

def _axis_gap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, second_start - first_end, first_start - second_end)

def _deduplicate_image_blocks(
    candidates: list[ImageBlockCandidate],
) -> list[ImageBlockCandidate]:
    """Collapse overlapping image layers while retaining all source indices."""
    deduplicated: list[ImageBlockCandidate] = []
    for candidate in sorted(
        candidates,
        key=lambda item: (
            item.page_number,
            item.bbox_pt[1],
            item.bbox_pt[0],
            item.block_index,
        ),
    ):
        rect = _candidate_rect(candidate)
        merged_at: int | None = None
        for index, current in enumerate(deduplicated):
            if current.page_number != candidate.page_number:
                continue
            current_rect = _candidate_rect(current)
            intersection = rect & current_rect
            if intersection.is_empty:
                continue
            smaller_area = min(rect.get_area(), current_rect.get_area())
            larger_area = max(rect.get_area(), current_rect.get_area())
            containment = intersection.get_area() / max(1.0, smaller_area)
            area_similarity = smaller_area / max(1.0, larger_area)
            edge_delta = max(
                abs(rect.x0 - current_rect.x0),
                abs(rect.y0 - current_rect.y0),
                abs(rect.x1 - current_rect.x1),
                abs(rect.y1 - current_rect.y1),
            )
            # A page-sized background often fully contains the scientific figure.
            # Treat layers as duplicates only when their extents are genuinely
            # similar (or nearly identical), not merely because one contains the
            # other.
            if containment < 0.92 or (area_similarity < 0.62 and edge_delta > 3.0):
                continue
            merged_at = index
            members = tuple(
                sorted(set(image_block_indices(current) + image_block_indices(candidate)))
            )
            union = _rect_union(current_rect, rect)
            deduplicated[index] = ImageBlockCandidate(
                page_number=current.page_number,
                block_index=min(members),
                bbox_pt=tuple(float(value) for value in union),
                member_block_indices=members,
            )
            break
        if merged_at is None:
            deduplicated.append(
                ImageBlockCandidate(
                    page_number=candidate.page_number,
                    block_index=candidate.block_index,
                    bbox_pt=candidate.bbox_pt,
                    member_block_indices=image_block_indices(candidate),
                )
            )
    return deduplicated

def _caption_owner(
    candidate: ImageBlockCandidate,
    captions: list[Caption],
) -> int | None:
    """Return a confident nearby caption index for anti-merge evidence."""
    rect = _candidate_rect(candidate)
    ranked: list[tuple[float, float, int]] = []
    for index, caption in enumerate(captions):
        if caption.page_number != candidate.page_number:
            continue
        gap = caption.bbox_pt[1] - rect.y1
        if gap < -8.0 or gap > 80.0:
            continue
        overlap = horizontal_overlap_ratio(candidate.bbox_pt, caption.bbox_pt)
        if overlap < 0.55:
            continue
        ranked.append(
            (
                max(0.0, gap),
                abs(bbox_center_x(candidate.bbox_pt) - bbox_center_x(caption.bbox_pt)),
                index,
            )
        )
    return min(ranked)[2] if ranked else None

def _caption_separates_vertical_blocks(
    first: ImageBlockCandidate,
    second: ImageBlockCandidate,
    captions: list[Caption],
) -> bool:
    first_rect = _candidate_rect(first)
    second_rect = _candidate_rect(second)
    if first_rect.y0 > second_rect.y0:
        first_rect, second_rect = second_rect, first_rect
    if first_rect.y1 > second_rect.y0:
        return False
    union = _rect_union(first_rect, second_rect)
    union_bbox = tuple(float(value) for value in union)
    for caption in captions:
        if caption.page_number != first.page_number:
            continue
        if caption.bbox_pt[1] < first_rect.y1 - 4.0:
            continue
        if caption.bbox_pt[3] > second_rect.y0 + 4.0:
            continue
        if horizontal_overlap_ratio(union_bbox, caption.bbox_pt) >= 0.30:
            return True
    return False

def _caption_separates_horizontal_blocks(
    first: ImageBlockCandidate,
    second: ImageBlockCandidate,
    captions: list[Caption],
) -> bool:
    """Keep side-by-side independent figures separate when two captions follow."""
    first_rect = _candidate_rect(first)
    second_rect = _candidate_rect(second)
    if first_rect.x0 > second_rect.x0:
        first_rect, second_rect = second_rect, first_rect
    if first_rect.x1 > second_rect.x0:
        return False
    vertical_overlap = _axis_overlap_ratio(
        first_rect.y0,
        first_rect.y1,
        second_rect.y0,
        second_rect.y1,
    )
    if vertical_overlap < 0.45:
        return False
    union = _rect_union(first_rect, second_rect)
    bottom = max(first_rect.y1, second_rect.y1)
    following = []
    for caption in captions:
        caption_bbox = caption_bbox_on_page(caption, first.page_number)
        if caption_bbox is None:
            continue
        caption_rect = fitz.Rect(caption_bbox)
        gap = caption_rect.y0 - bottom
        if gap < -12.0 or gap > 105.0:
            continue
        if caption_rect.x1 < union.x0 - 24.0 or caption_rect.x0 > union.x1 + 24.0:
            continue
        following.append(caption.identity)
    return len(set(following)) >= 2

def _image_blocks_are_neighbors(
    first: ImageBlockCandidate,
    second: ImageBlockCandidate,
    captions: list[Caption],
) -> bool:
    if first.page_number != second.page_number:
        return False

    first_owner = _caption_owner(first, captions)
    second_owner = _caption_owner(second, captions)
    if (
        first_owner is not None
        and second_owner is not None
        and first_owner != second_owner
    ):
        return False

    first_rect = _candidate_rect(first)
    second_rect = _candidate_rect(second)
    horizontal_gap = _axis_gap(
        first_rect.x0,
        first_rect.x1,
        second_rect.x0,
        second_rect.x1,
    )
    vertical_gap = _axis_gap(
        first_rect.y0,
        first_rect.y1,
        second_rect.y0,
        second_rect.y1,
    )
    vertical_overlap = _axis_overlap_ratio(
        first_rect.y0,
        first_rect.y1,
        second_rect.y0,
        second_rect.y1,
    )
    horizontal_overlap = _axis_overlap_ratio(
        first_rect.x0,
        first_rect.x1,
        second_rect.x0,
        second_rect.x1,
    )

    if horizontal_gap == 0.0 and vertical_gap == 0.0:
        intersection = first_rect & second_rect
        smaller_area = min(first_rect.get_area(), second_rect.get_area())
        larger_area = max(first_rect.get_area(), second_rect.get_area())
        containment = intersection.get_area() / max(1.0, smaller_area)
        area_similarity = smaller_area / max(1.0, larger_area)
        if containment >= 0.90 and area_similarity < 0.55:
            # Do not absorb a foreground scientific figure into a page-sized
            # background / watermark image.
            return False
        return True

    same_row = (
        vertical_overlap >= 0.55
        and horizontal_gap
        <= max(10.0, 0.25 * min(first_rect.width, second_rect.width))
    )
    if same_row:
        return not _caption_separates_horizontal_blocks(first, second, captions)

    same_column = (
        horizontal_overlap >= 0.45
        and vertical_gap
        <= max(10.0, 0.20 * min(first_rect.height, second_rect.height))
    )
    if not same_column:
        return False
    return not _caption_separates_vertical_blocks(first, second, captions)

def _is_caption_adjacent_compact_scheme_candidate(
    rect: fitz.Rect,
    page_number: int,
    captions: Collection[Caption],
    scheme_min_width_pt: float,
    scheme_min_height_pt: float,
) -> bool:
    """Retain a small raster only when a Scheme caption provides strong context."""
    if (
        rect.width < scheme_min_width_pt
        or rect.height < scheme_min_height_pt
        or rect.get_area() < max(2800.0, 0.55 * scheme_min_width_pt * scheme_min_height_pt)
    ):
        return False
    for caption in captions:
        if normalize_caption_kind(caption.caption_kind) != "scheme":
            continue
        caption_bbox = caption_bbox_on_page(caption, page_number)
        if caption_bbox is None:
            continue
        caption_rect = fitz.Rect(caption_bbox)
        x_overlap = _axis_overlap_ratio(
            rect.x0,
            rect.x1,
            caption_rect.x0,
            caption_rect.x1,
        )
        y_overlap = _axis_overlap_ratio(
            rect.y0,
            rect.y1,
            caption_rect.y0,
            caption_rect.y1,
        )
        center_delta = abs(
            0.5 * (rect.x0 + rect.x1) - 0.5 * (caption_rect.x0 + caption_rect.x1)
        )
        column_aligned = x_overlap >= 0.28 or center_delta <= max(
            42.0,
            0.32 * max(rect.width, caption_rect.width),
        )
        vertical_gap = _rect_axis_gap(rect, caption_rect, horizontal=False)
        # A Scheme may be assembled from several shallow raster strips separated
        # by reaction-condition whitespace. Keep the whole caption corridor
        # available for the later caption-guided merge, while other captions still
        # act as hard separators in that merge stage.
        scheme_corridor_gap = max(
            300.0,
            MAX_CAPTION_GUIDED_VERTICAL_GAP_PT + scheme_min_height_pt,
            2.5 * rect.height,
        )
        if column_aligned and vertical_gap <= scheme_corridor_gap:
            return True
        horizontal_gap = _rect_axis_gap(rect, caption_rect, horizontal=True)
        if y_overlap >= 0.42 and horizontal_gap <= 48.0:
            return True
    return False

def cluster_image_block_candidates(
    candidates: list[ImageBlockCandidate],
    min_width_pt: float,
    min_height_pt: float,
    captions: list[Caption] | None = None,
    *,
    scheme_min_width_pt: float = DEFAULT_SCHEME_MIN_WIDTH_PT,
    scheme_min_height_pt: float = DEFAULT_SCHEME_MIN_HEIGHT_PT,
) -> list[ImageBlockCandidate]:
    """Group neighboring raster blocks, then filter by the union dimensions."""
    captions = captions or []
    nodes = _deduplicate_image_blocks(candidates)
    parents = list(range(len(nodes)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            if _image_blocks_are_neighbors(nodes[left], nodes[right], captions):
                union(left, right)

    components: dict[int, list[ImageBlockCandidate]] = {}
    for index, candidate in enumerate(nodes):
        components.setdefault(find(index), []).append(candidate)

    clustered: list[ImageBlockCandidate] = []
    for members in components.values():
        rect = _candidate_rect(members[0])
        block_indices: set[int] = set()
        for member in members:
            rect = _rect_union(rect, _candidate_rect(member))
            block_indices.update(image_block_indices(member))
        ordinary_size = rect.width >= min_width_pt and rect.height >= min_height_pt
        compact_scheme = _is_caption_adjacent_compact_scheme_candidate(
            rect,
            members[0].page_number,
            captions,
            scheme_min_width_pt,
            scheme_min_height_pt,
        )
        if not ordinary_size and not compact_scheme:
            continue
        ordered_indices = tuple(sorted(block_indices))
        clustered.append(
            ImageBlockCandidate(
                page_number=members[0].page_number,
                block_index=min(ordered_indices),
                bbox_pt=tuple(float(value) for value in rect),
                member_block_indices=ordered_indices,
            )
        )
    return sorted(
        clustered,
        key=lambda item: (item.page_number, item.bbox_pt[1], item.bbox_pt[0]),
    )

def collect_image_block_candidates(
    document: fitz.Document,
    pages: set[int] | None,
    min_width_pt: float,
    min_height_pt: float,
    captions: list[Caption] | None = None,
    *,
    scheme_min_width_pt: float = DEFAULT_SCHEME_MIN_WIDTH_PT,
    scheme_min_height_pt: float = DEFAULT_SCHEME_MIN_HEIGHT_PT,
) -> list[ImageBlockCandidate]:
    """Collect small meaningful rasters and return filtered same-page clusters."""
    raw: list[ImageBlockCandidate] = []
    for page_index, page in enumerate(document):
        page_number = page_index + 1
        if pages and page_number not in pages:
            continue
        page_blocks: list[ImageBlockCandidate] = []
        for block_index, block in enumerate(page.get_text("dict").get("blocks", []), 1):
            if block.get("type") != 1 or "bbox" not in block:
                continue
            rect = fitz.Rect(block["bbox"])
            if (
                rect.width < MIN_CLUSTER_BLOCK_SIDE_PT
                or rect.height < MIN_CLUSTER_BLOCK_SIDE_PT
                or rect.get_area() < MIN_CLUSTER_BLOCK_AREA_PT2
            ):
                continue
            page_blocks.append(
                ImageBlockCandidate(
                    page_number=page_number,
                    block_index=block_index,
                    bbox_pt=tuple(float(value) for value in rect),
                    member_block_indices=(block_index,),
                )
            )
        # PyMuPDF's text dictionary omits image blocks whose placement extends
        # beyond the PDF page, even though their visible portion is rendered.
        # XObject placement rectangles retain these clipped figures.
        fallback_index = 100000
        for image in page.get_images(full=True):
            for placement in page.get_image_rects(image[0]):
                rect = placement & page.rect
                if (
                    rect.is_empty
                    or rect.width < MIN_CLUSTER_BLOCK_SIDE_PT
                    or rect.height < MIN_CLUSTER_BLOCK_SIDE_PT
                    or rect.get_area() < MIN_CLUSTER_BLOCK_AREA_PT2
                ):
                    continue
                if any(
                    (rect & fitz.Rect(block.bbox_pt)).get_area()
                    >= 0.90 * min(rect.get_area(), fitz.Rect(block.bbox_pt).get_area())
                    for block in page_blocks
                ):
                    continue
                fallback_index += 1
                page_blocks.append(
                    ImageBlockCandidate(
                        page_number=page_number,
                        block_index=fallback_index,
                        bbox_pt=tuple(float(value) for value in rect),
                        member_block_indices=(fallback_index,),
                    )
                )
        raw.extend(page_blocks)
    return cluster_image_block_candidates(
        raw,
        min_width_pt,
        min_height_pt,
        captions,
        scheme_min_width_pt=scheme_min_width_pt,
        scheme_min_height_pt=scheme_min_height_pt,
    )


def match_caption_to_previous_page_figure(
    document: fitz.Document,
    caption: Caption,
    candidates: list[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
) -> ImageBlockCandidate | None:
    """Own a dominant preceding-page raster when the caption starts next page."""
    if caption.page_number <= 1:
        return None
    caption_page = document[caption.page_number - 1]
    if caption.bbox_pt[1] > caption_page.rect.y0 + 0.28 * caption_page.rect.height:
        return None
    previous_page = document[caption.page_number - 2]
    ranked: list[tuple[float, ImageBlockCandidate]] = []
    for candidate in candidates:
        if candidate.page_number != caption.page_number - 1:
            continue
        if image_block_is_consumed(candidate, consumed):
            continue
        rect = fitz.Rect(candidate.bbox_pt)
        page_rect = previous_page.rect
        if (
            rect.width < 0.65 * page_rect.width
            or rect.get_area() < 0.20 * page_rect.get_area()
            or rect.y1 < page_rect.y0 + 0.70 * page_rect.height
        ):
            continue
        score = _cross_page_candidate_score(document, caption, candidate)
        if score >= 10.0:
            ranked.append((score, candidate))
    ranked.sort(key=lambda item: -item[0])
    if not ranked or (len(ranked) > 1 and ranked[1][0] >= ranked[0][0] - 1.0):
        return None
    return ranked[0][1]

def caption_bbox_on_page(caption: Caption, page_number: int) -> BBox | None:
    segments = caption_segments_on_page(caption, page_number)
    if not segments:
        return None
    bbox = segments[0].bbox_pt
    for segment in segments[1:]:
        bbox = union_bbox(bbox, segment.bbox_pt)
    return bbox

def _page_background_penalty(page: fitz.Page, rect: fitz.Rect) -> float:
    area_ratio = rect.get_area() / max(1.0, page.rect.get_area())
    edge_distance = max(
        abs(rect.x0 - page.rect.x0),
        abs(rect.y0 - page.rect.y0),
        abs(rect.x1 - page.rect.x1),
        abs(rect.y1 - page.rect.y1),
    )
    if area_ratio >= 0.90:
        return 10.0
    if area_ratio >= 0.76 and edge_distance <= 8.0:
        return 7.0
    if area_ratio >= 0.72:
        return 3.0
    return 0.0

def _same_page_candidate_score(
    document: fitz.Document,
    caption: Caption,
    candidate: ImageBlockCandidate,
) -> float:
    caption_bbox = caption_bbox_on_page(caption, candidate.page_number)
    if caption_bbox is None:
        return -math.inf
    page = document[candidate.page_number - 1]
    image_rect = fitz.Rect(candidate.bbox_pt)
    caption_rect = fitz.Rect(caption_bbox)
    caption_overlap = (image_rect & caption_rect).get_area() / max(
        1.0,
        caption_rect.get_area(),
    )
    if caption_overlap >= 0.18:
        return -math.inf

    x_overlap = _axis_overlap_ratio(
        image_rect.x0,
        image_rect.x1,
        caption_rect.x0,
        caption_rect.x1,
    )
    y_overlap = _axis_overlap_ratio(
        image_rect.y0,
        image_rect.y1,
        caption_rect.y0,
        caption_rect.y1,
    )
    center_x_delta = abs(
        0.5 * (image_rect.x0 + image_rect.x1)
        - 0.5 * (caption_rect.x0 + caption_rect.x1)
    ) / max(1.0, page.rect.width)
    width_ratio = min(image_rect.width, caption_rect.width) / max(
        1.0,
        max(image_rect.width, caption_rect.width),
    )

    scores: list[float] = []
    edge_delta = min(
        abs(image_rect.x0 - caption_rect.x0),
        abs(image_rect.x1 - caption_rect.x1),
    )
    strongly_aligned = x_overlap >= 0.62 and (
        center_x_delta <= 0.16
        or edge_delta <= max(24.0, 0.08 * page.rect.width)
    )
    same_page_gap_limit = 240.0 if strongly_aligned else 110.0
    above_gap = caption_rect.y0 - image_rect.y1
    if -14.0 <= above_gap <= same_page_gap_limit and x_overlap >= 0.28:
        scores.append(
            11.5
            - max(0.0, above_gap) / (28.0 if strongly_aligned else 18.0)
            + 3.6 * x_overlap
            + 1.2 * width_ratio
            - 7.0 * center_x_delta
        )

    below_gap = image_rect.y0 - caption_rect.y1
    if -10.0 <= below_gap <= same_page_gap_limit and x_overlap >= 0.28:
        scores.append(
            9.7
            - max(0.0, below_gap) / (30.0 if strongly_aligned else 20.0)
            + 3.4 * x_overlap
            + 1.0 * width_ratio
            - 7.0 * center_x_delta
        )

    side_gap = min(
        abs(caption_rect.x0 - image_rect.x1),
        abs(image_rect.x0 - caption_rect.x1),
    )
    if side_gap <= 42.0 and y_overlap >= 0.45:
        scores.append(
            8.8
            - side_gap / 14.0
            + 3.0 * y_overlap
            + 0.8 * width_ratio
        )

    if not scores:
        return -math.inf
    return max(scores) - _page_background_penalty(page, image_rect)

def _cross_page_candidate_score(
    document: fitz.Document,
    caption: Caption,
    candidate: ImageBlockCandidate,
) -> float:
    first_caption_page = min(caption.page_numbers)
    last_caption_page = max(caption.page_numbers)
    relation: str | None = None
    caption_page_number: int | None = None
    if candidate.page_number == first_caption_page - 1:
        relation = "previous"
        caption_page_number = first_caption_page
    elif candidate.page_number == last_caption_page + 1:
        relation = "next"
        caption_page_number = last_caption_page
    if relation is None or caption_page_number is None:
        return -math.inf

    caption_bbox = caption_bbox_on_page(caption, caption_page_number)
    if caption_bbox is None:
        return -math.inf
    caption_page = document[caption_page_number - 1]
    image_page = document[candidate.page_number - 1]
    image_rect = fitz.Rect(candidate.bbox_pt)
    caption_rect = fitz.Rect(caption_bbox)

    area_ratio = image_rect.get_area() / max(1.0, image_page.rect.get_area())
    width_ratio = image_rect.width / max(1.0, image_page.rect.width)
    x_overlap = _axis_overlap_ratio(
        image_rect.x0,
        image_rect.x1,
        caption_rect.x0,
        caption_rect.x1,
    )
    center_delta = abs(
        0.5 * (image_rect.x0 + image_rect.x1)
        - 0.5 * (caption_rect.x0 + caption_rect.x1)
    ) / max(1.0, image_page.rect.width)
    if x_overlap < 0.22 and center_delta > 0.18 and width_ratio < 0.72:
        return -math.inf

    score = 1.5 + 3.0 * min(1.0, x_overlap) + 2.0 * min(1.0, width_ratio)
    score += 3.0 * min(1.0, area_ratio / 0.28)
    if relation == "previous":
        caption_top_ratio = (
            caption_rect.y0 - caption_page.rect.y0
        ) / max(1.0, caption_page.rect.height)
        image_bottom_ratio = (
            image_rect.y1 - image_page.rect.y0
        ) / max(1.0, image_page.rect.height)
        score += 3.0 * max(0.0, 1.0 - caption_top_ratio / 0.35)
        score += 2.6 * max(0.0, (image_bottom_ratio - 0.45) / 0.55)
        if has_next_page_caption_marker(image_page):
            score += 5.0
        below_region = fitz.Rect(
            image_page.rect.x0,
            min(image_page.rect.y1, image_rect.y1 + 2.0),
            image_page.rect.x1,
            image_page.rect.y1,
        )
        prose_blocks, prose_words = prose_evidence_in_rect(image_page, below_region)
        if prose_blocks == 0 or prose_words < 24.0:
            score += 1.4
        # Full-page / dominant figures can legitimately have captions after body
        # text on the following page, so permit a caption lower on that page when
        # the visual evidence is strong.
        if area_ratio >= 0.24 and image_bottom_ratio >= 0.66:
            score += 2.0
    else:
        caption_bottom_ratio = (
            caption_rect.y1 - caption_page.rect.y0
        ) / max(1.0, caption_page.rect.height)
        image_top_ratio = (
            image_rect.y0 - image_page.rect.y0
        ) / max(1.0, image_page.rect.height)
        score += 3.0 * max(0.0, (caption_bottom_ratio - 0.62) / 0.38)
        score += 2.8 * max(0.0, 1.0 - image_top_ratio / 0.35)

    score -= 8.0 * center_delta
    score -= _page_background_penalty(image_page, image_rect)
    return score

def _candidate_pair_can_form_one_figure(
    first: ImageBlockCandidate,
    second: ImageBlockCandidate,
) -> bool:
    if first.page_number != second.page_number:
        return False
    first_rect = fitz.Rect(first.bbox_pt)
    second_rect = fitz.Rect(second.bbox_pt)
    horizontal_overlap = _axis_overlap_ratio(
        first_rect.x0,
        first_rect.x1,
        second_rect.x0,
        second_rect.x1,
    )
    width_ratio = min(first_rect.width, second_rect.width) / max(
        1.0,
        max(first_rect.width, second_rect.width),
    )
    vertical_gap = _rect_axis_gap(first_rect, second_rect, horizontal=False)
    center_delta = abs(
        0.5 * (first_rect.x0 + first_rect.x1)
        - 0.5 * (second_rect.x0 + second_rect.x1)
    )
    return (
        horizontal_overlap >= 0.72
        and width_ratio >= 0.60
        and center_delta <= max(24.0, 0.13 * max(first_rect.width, second_rect.width))
        and vertical_gap <= MAX_CAPTION_GUIDED_VERTICAL_GAP_PT
    )

def _ordered_caption_row_match(
    caption: Caption,
    captions: list[Caption],
    candidates: list[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
) -> ImageBlockCandidate | None:
    """Map merged side-by-side caption text to figures by left-to-right order."""
    caption_bbox = caption_bbox_on_page(caption, caption.page_number)
    if caption_bbox is None:
        return None
    caption_rect = fitz.Rect(caption_bbox)
    caption_height = max(1.0, caption_rect.height)
    siblings: list[Caption] = []
    for other in captions:
        other_bbox = caption_bbox_on_page(other, caption.page_number)
        if other_bbox is None:
            continue
        if abs(other_bbox[1] - caption_bbox[1]) <= max(18.0, 1.8 * caption_height):
            siblings.append(other)
    if len(siblings) < 2:
        return None
    siblings.sort(
        key=lambda item: (
            (caption_bbox_on_page(item, caption.page_number) or item.bbox_pt)[0],
            item.identity,
        )
    )
    try:
        caption_index = siblings.index(caption)
    except ValueError:
        return None

    row_candidates = []
    for candidate in candidates:
        if candidate.page_number != caption.page_number:
            continue
        rect = fitz.Rect(candidate.bbox_pt)
        gap = caption_rect.y0 - rect.y1
        if gap < -14.0 or gap > 125.0:
            continue
        row_candidates.append(candidate)
    if len(row_candidates) != len(siblings):
        return None
    y_centers = [bbox_center_y(candidate.bbox_pt) for candidate in row_candidates]
    if max(y_centers) - min(y_centers) > 0.35 * max(
        1.0,
        max(fitz.Rect(candidate.bbox_pt).height for candidate in row_candidates),
    ):
        return None
    row_candidates.sort(key=lambda item: item.bbox_pt[0])
    chosen = row_candidates[caption_index]
    if image_block_is_consumed(chosen, consumed):
        return None
    return chosen

def match_caption_to_image_block(
    document: fitz.Document,
    caption: Caption,
    candidates: list[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
    captions: list[Caption] | None = None,
) -> ImageBlockCandidate | None:
    """Score same-page and adjacent-page image candidates for one caption."""
    captions = captions or [caption]

    ordered_match = _ordered_caption_row_match(
        caption,
        captions,
        candidates,
        consumed,
    )
    if ordered_match is not None:
        return ordered_match

    same_page_ranked: list[tuple[float, ImageBlockCandidate]] = []
    cross_page_ranked: list[tuple[float, ImageBlockCandidate]] = []
    caption_pages = set(caption.page_numbers)
    for candidate in candidates:
        if image_block_is_consumed(candidate, consumed):
            continue
        if candidate.page_number in caption_pages:
            score = _same_page_candidate_score(document, caption, candidate)
            if math.isfinite(score):
                same_page_ranked.append((score, candidate))
        elif (
            candidate.page_number == min(caption.page_numbers) - 1
            or candidate.page_number == max(caption.page_numbers) + 1
        ):
            score = _cross_page_candidate_score(document, caption, candidate)
            if math.isfinite(score):
                cross_page_ranked.append((score, candidate))

    def choose(
        ranked: list[tuple[float, ImageBlockCandidate]],
        threshold: float,
    ) -> ImageBlockCandidate | None:
        if not ranked:
            return None
        ranked.sort(
            key=lambda item: (
                -item[0],
                abs(item[1].page_number - caption.page_number),
                item[1].bbox_pt[1],
                item[1].bbox_pt[0],
            )
        )
        best_score, best = ranked[0]
        if best_score < threshold:
            return None
        if len(ranked) >= 2:
            second_score, second = ranked[1]
            if second_score >= best_score - 0.85 and not _candidate_pair_can_form_one_figure(
                best,
                second,
            ):
                return None
        return best

    same_page = choose(same_page_ranked, threshold=6.0)
    if same_page is not None:
        return same_page
    return choose(cross_page_ranked, threshold=7.0)

def _candidate_owned_by_other_caption(
    candidate: ImageBlockCandidate,
    current_caption: Caption,
    captions: list[Caption],
) -> bool:
    rect = fitz.Rect(candidate.bbox_pt)
    current_identity = current_caption.identity
    for caption in captions:
        if caption.identity == current_identity:
            continue
        caption_bbox = caption_bbox_on_page(caption, candidate.page_number)
        if caption_bbox is None:
            continue
        caption_rect = fitz.Rect(caption_bbox)
        gap = caption_rect.y0 - rect.y1
        overlap = _axis_overlap_ratio(
            rect.x0,
            rect.x1,
            caption_rect.x0,
            caption_rect.x1,
        )
        if -10.0 <= gap <= 72.0 and overlap >= 0.48:
            return True
    return False

def _candidate_reserved_by_ordered_caption_row(
    candidate: ImageBlockCandidate,
    current_caption: Caption,
    captions: list[Caption],
    candidates: list[ImageBlockCandidate],
) -> bool:
    current_identity = current_caption.identity
    for other in captions:
        if other.identity == current_identity:
            continue
        mapped = _ordered_caption_row_match(other, captions, candidates, set())
        if mapped is candidate:
            return True
    return False

def _gap_rect_between(first: fitz.Rect, second: fitz.Rect) -> fitz.Rect:
    if first.y1 <= second.y0:
        return fitz.Rect(
            max(first.x0, second.x0),
            first.y1,
            min(first.x1, second.x1),
            second.y0,
        )
    if second.y1 <= first.y0:
        return fitz.Rect(
            max(first.x0, second.x0),
            second.y1,
            min(first.x1, second.x1),
            first.y0,
        )
    if first.x1 <= second.x0:
        return fitz.Rect(
            first.x1,
            max(first.y0, second.y0),
            second.x0,
            min(first.y1, second.y1),
        )
    if second.x1 <= first.x0:
        return fitz.Rect(
            second.x1,
            max(first.y0, second.y0),
            first.x0,
            min(first.y1, second.y1),
        )
    return fitz.Rect(0.0, 0.0, 0.0, 0.0)

def _gap_has_blocking_text(
    page: fitz.Page,
    gap_rect: fitz.Rect,
    caption: Caption,
) -> bool:
    if gap_rect.is_empty or gap_rect.width <= 1.0 or gap_rect.height <= 1.0:
        return False
    padded = fitz.Rect(
        gap_rect.x0 - 4.0,
        gap_rect.y0 - 2.0,
        gap_rect.x1 + 4.0,
        gap_rect.y1 + 2.0,
    )
    for block in collect_text_blocks(page, page.number + 1):
        block_rect = fitz.Rect(block.bbox_pt)
        intersection = block_rect & padded
        if intersection.is_empty:
            continue
        if _caption_rect_intersects(block_rect, caption, page.number + 1, tolerance=2.0):
            continue
        overlap = intersection.get_area() / max(1.0, block_rect.get_area())
        if overlap < 0.12:
            continue
        match = CAPTION_START_RE.match(block.text)
        if match is not None:
            if caption_identity_from_match(match) != caption.identity:
                return True
            continue
        words = block.text.split()
        if looks_like_heading(block.text):
            return True
        if len(block.text) >= 75 and len(words) >= 11:
            return True
    return False

def _caption_direction_allows_candidate(
    caption: Caption,
    page_number: int,
    group_rect: fitz.Rect,
    candidate_rect: fitz.Rect,
) -> bool:
    caption_bbox = caption_bbox_on_page(caption, page_number)
    if caption_bbox is None:
        return True
    caption_rect = fitz.Rect(caption_bbox)
    if caption_rect.y0 >= group_rect.y1 - 8.0:
        return candidate_rect.y1 <= caption_rect.y0 + 2.0
    if caption_rect.y1 <= group_rect.y0 + 8.0:
        return candidate_rect.y0 >= caption_rect.y1 - 2.0
    if caption_rect.x0 >= group_rect.x1 - 8.0:
        return candidate_rect.x1 <= caption_rect.x0 + 2.0
    if caption_rect.x1 <= group_rect.x0 + 8.0:
        return candidate_rect.x0 >= caption_rect.x1 - 2.0
    return True

def _figure_part_merge_rank(
    page: fitz.Page,
    caption: Caption,
    group_rect: fitz.Rect,
    candidate_rect: fitz.Rect,
) -> tuple[float, float, float] | None:
    if not _caption_direction_allows_candidate(
        caption,
        page.number + 1,
        group_rect,
        candidate_rect,
    ):
        return None

    horizontal_overlap = _axis_overlap_ratio(
        group_rect.x0,
        group_rect.x1,
        candidate_rect.x0,
        candidate_rect.x1,
    )
    vertical_overlap = _axis_overlap_ratio(
        group_rect.y0,
        group_rect.y1,
        candidate_rect.y0,
        candidate_rect.y1,
    )
    vertical_gap = _rect_axis_gap(group_rect, candidate_rect, horizontal=False)
    horizontal_gap = _rect_axis_gap(group_rect, candidate_rect, horizontal=True)
    width_ratio = min(group_rect.width, candidate_rect.width) / max(
        1.0,
        max(group_rect.width, candidate_rect.width),
    )
    height_ratio = min(group_rect.height, candidate_rect.height) / max(
        1.0,
        max(group_rect.height, candidate_rect.height),
    )
    center_x_delta = abs(
        0.5 * (group_rect.x0 + group_rect.x1)
        - 0.5 * (candidate_rect.x0 + candidate_rect.x1)
    )

    strong_column_alignment = (
        horizontal_overlap >= 0.80
        and width_ratio >= 0.68
        and center_x_delta <= max(22.0, 0.11 * max(group_rect.width, candidate_rect.width))
    )
    vertical_limit = max(
        34.0,
        min(
            MAX_CAPTION_GUIDED_VERTICAL_GAP_PT,
            20.0 + (0.48 if strong_column_alignment else 0.24)
            * min(group_rect.width, candidate_rect.width),
        ),
    )
    if (
        horizontal_overlap >= 0.62
        and width_ratio >= 0.52
        and vertical_gap <= vertical_limit
    ):
        gap_rect = _gap_rect_between(group_rect, candidate_rect)
        if _gap_has_blocking_text(page, gap_rect, caption):
            return None
        return (
            vertical_gap,
            -horizontal_overlap,
            center_x_delta / max(1.0, page.rect.width),
        )

    horizontal_limit = max(
        18.0,
        min(58.0, 0.18 * min(group_rect.width, candidate_rect.width)),
    )
    if (
        vertical_overlap >= 0.58
        and height_ratio >= 0.32
        and horizontal_gap <= horizontal_limit
    ):
        gap_rect = _gap_rect_between(group_rect, candidate_rect)
        if _gap_has_blocking_text(page, gap_rect, caption):
            return None
        return (
            horizontal_gap + 5.0,
            -vertical_overlap,
            abs(
                0.5 * (group_rect.y0 + group_rect.y1)
                - 0.5 * (candidate_rect.y0 + candidate_rect.y1)
            ) / max(1.0, page.rect.height),
        )
    return None

def merge_caption_owned_figure_parts(
    document: fitz.Document,
    caption: Caption,
    anchor: ImageBlockCandidate,
    candidates: list[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
    captions: list[Caption],
) -> ImageBlockCandidate:
    """Grow a caption-matched raster into one complete multi-block figure.

    This caption-guided pass is deliberately more permissive than global image
    clustering. It can bridge a large white gap in a long single-column figure,
    while formal captions and paragraph text inside the gap remain hard barriers.
    """
    page = document[anchor.page_number - 1]
    group_rect = fitz.Rect(anchor.bbox_pt)
    member_indices: set[int] = set(image_block_indices(anchor))
    member_candidates: set[int] = {id(anchor)}

    for _ in range(min(16, len(candidates))):
        ranked: list[
            tuple[tuple[float, float, float], ImageBlockCandidate]
        ] = []
        for candidate in candidates:
            if id(candidate) in member_candidates:
                continue
            if candidate.page_number != anchor.page_number:
                continue
            if image_block_is_consumed(candidate, consumed):
                continue
            if _candidate_reserved_by_ordered_caption_row(
                candidate,
                caption,
                captions,
                candidates,
            ):
                continue
            if _candidate_owned_by_other_caption(candidate, caption, captions):
                continue
            rank = _figure_part_merge_rank(
                page,
                caption,
                group_rect,
                fitz.Rect(candidate.bbox_pt),
            )
            if rank is not None:
                ranked.append((rank, candidate))
        if not ranked:
            break
        ranked.sort(key=lambda item: item[0])
        _, chosen = ranked[0]
        group_rect |= fitz.Rect(chosen.bbox_pt)
        member_indices.update(image_block_indices(chosen))
        member_candidates.add(id(chosen))

    ordered = tuple(sorted(member_indices))
    return ImageBlockCandidate(
        page_number=anchor.page_number,
        block_index=min(ordered),
        bbox_pt=tuple(float(value) for value in group_rect),
        member_block_indices=ordered,
    )

def has_next_page_caption_marker(page: fitz.Page) -> bool:
    return bool(re.search(r"caption\s+on\s+(?:the\s+)?next\s+page", page.get_text(), re.IGNORECASE))

def _caption_crop_is_safe(
    page: fitz.Page,
    rect: fitz.Rect,
    caption: Caption,
    min_height_pt: float,
) -> bool:
    if rect.is_empty or rect.height < min_height_pt or rect.width < 36.0:
        return False
    if is_prose_dominated_crop(page, rect):
        return False
    quality_issues = caption_crop_quality_issues(page, rect, caption)
    if caption_crop_has_blocking_issue(quality_issues):
        return False
    # A synthetic crop must contain real raster/vector evidence. This prevents a
    # short caption from producing an arbitrary column of surrounding prose.
    return crop_has_visual_evidence(page, rect)

def _boundary_pseudo_caption(
    page: fitz.Page,
    caption: Caption,
    at_bottom: bool,
) -> Caption:
    source_bbox = fitz.Rect(caption.bbox_pt)
    x0 = max(page.rect.x0 + 24.0, source_bbox.x0)
    x1 = min(page.rect.x1 - 24.0, max(source_bbox.x1, x0 + 80.0))
    if at_bottom:
        bbox = (x0, page.rect.y1 - 3.0, x1, page.rect.y1 - 1.0)
    else:
        bbox = (x0, page.rect.y0 + 1.0, x1, page.rect.y0 + 3.0)
    return Caption(
        page_number=page.number + 1,
        figure_number=caption.figure_number,
        text=caption.text,
        bbox_pt=tuple(float(value) for value in bbox),
        caption_kind=caption.caption_kind,
        caption_label=caption.caption_label,
        segments=(
            CaptionSegment(
                page_number=page.number + 1,
                text=caption.text,
                bbox_pt=tuple(float(value) for value in bbox),
            ),
        ),
        font_size_pt=caption.font_size_pt,
        font_name=caption.font_name,
        bold=caption.bold,
    )

def _cross_page_crop_score(
    caption_page: fitz.Page,
    caption_rect: fitz.Rect,
    image_page: fitz.Page,
    crop_rect: fitz.Rect,
    relation: str,
) -> float:
    area_ratio = crop_rect.get_area() / max(1.0, image_page.rect.get_area())
    score = 4.0 * min(1.0, area_ratio / 0.24)
    if relation == "previous":
        caption_top_ratio = (
            caption_rect.y0 - caption_page.rect.y0
        ) / max(1.0, caption_page.rect.height)
        crop_bottom_ratio = (
            crop_rect.y1 - image_page.rect.y0
        ) / max(1.0, image_page.rect.height)
        score += 3.0 * max(0.0, 1.0 - caption_top_ratio / 0.38)
        score += 2.5 * max(0.0, (crop_bottom_ratio - 0.45) / 0.55)
        if has_next_page_caption_marker(image_page):
            score += 4.0
    else:
        caption_bottom_ratio = (
            caption_rect.y1 - caption_page.rect.y0
        ) / max(1.0, caption_page.rect.height)
        crop_top_ratio = (
            crop_rect.y0 - image_page.rect.y0
        ) / max(1.0, image_page.rect.height)
        score += 3.0 * max(0.0, (caption_bottom_ratio - 0.62) / 0.38)
        score += 2.5 * max(0.0, 1.0 - crop_top_ratio / 0.38)
    return score

def find_caption_crop(
    document: fitz.Document,
    caption: Caption,
    min_height_pt: float,
) -> tuple[int, fitz.Rect | None]:
    """Return a safe same-page or adjacent-page visual crop for a caption."""
    # First try every physical page on which the caption itself appears. A caption
    # that starts at the bottom of one page and continues on the next should still
    # be anchored to the figure on its starting page.
    for page_number in caption.page_numbers:
        page = document[page_number - 1]
        rect = find_caption_figure_rect(page, caption, min_height_pt)
        if _caption_crop_is_safe(page, rect, caption, min_height_pt):
            return page_number, rect

    ranked_cross_page: list[tuple[float, int, fitz.Rect]] = []
    first_caption_page = min(caption.page_numbers)
    last_caption_page = max(caption.page_numbers)

    if first_caption_page > 1:
        image_page = document[first_caption_page - 2]
        pseudo = _boundary_pseudo_caption(image_page, caption, at_bottom=True)
        rect = find_caption_figure_rect(image_page, pseudo, min_height_pt)
        if _caption_crop_is_safe(image_page, rect, pseudo, min_height_pt):
            caption_page = document[first_caption_page - 1]
            caption_bbox = caption_bbox_on_page(caption, first_caption_page) or caption.bbox_pt
            score = _cross_page_crop_score(
                caption_page,
                fitz.Rect(caption_bbox),
                image_page,
                rect,
                "previous",
            )
            if score >= 4.5:
                ranked_cross_page.append((score, first_caption_page - 1, rect))

    if last_caption_page < len(document):
        image_page = document[last_caption_page]
        pseudo = _boundary_pseudo_caption(image_page, caption, at_bottom=False)
        rect = find_caption_figure_rect(image_page, pseudo, min_height_pt)
        if _caption_crop_is_safe(image_page, rect, pseudo, min_height_pt):
            caption_page = document[last_caption_page - 1]
            caption_bbox = caption_bbox_on_page(caption, last_caption_page) or caption.bbox_pt
            score = _cross_page_crop_score(
                caption_page,
                fitz.Rect(caption_bbox),
                image_page,
                rect,
                "next",
            )
            if score >= 4.5:
                ranked_cross_page.append((score, last_caption_page + 1, rect))

    if not ranked_cross_page:
        return caption.page_number, None
    ranked_cross_page.sort(key=lambda item: -item[0])
    if len(ranked_cross_page) > 1 and ranked_cross_page[1][0] >= ranked_cross_page[0][0] - 0.75:
        return caption.page_number, None
    _, page_number, rect = ranked_cross_page[0]
    return page_number, rect

def prose_evidence_in_rect(page: fitz.Page, clip: fitz.Rect) -> tuple[int, float]:
    """Return long-paragraph count and overlap-weighted word count in a region."""
    if clip.is_empty:
        return 0, 0.0

    paragraph_count = 0
    weighted_words = 0.0
    min_block_width = max(72.0, 0.18 * page.rect.width)
    for block in collect_text_blocks(page, 1):
        block_rect = fitz.Rect(block.bbox_pt)
        if block_rect.width < min_block_width:
            continue
        word_count = len(block.text.split())
        if len(block.text) < 120 or word_count < 18:
            continue
        if CAPTION_START_RE.match(block.text):
            continue
        intersection = block_rect & clip
        if intersection.is_empty:
            continue
        overlap = intersection.get_area() / max(1.0, block_rect.get_area())
        if overlap < 0.08:
            continue
        paragraph_count += 1
        weighted_words += word_count * overlap
    return paragraph_count, weighted_words

def is_prose_dominated_crop(page: fitz.Page, rect: fitz.Rect) -> bool:
    """Reject synthetic figure crops that mainly contain article paragraphs."""
    prose_blocks, prose_words = prose_evidence_in_rect(page, rect)
    return prose_blocks >= 2 and prose_words >= 80.0

def caption_crop_quality_issues(
    page: fitz.Page,
    rect: fitz.Rect,
    caption: Caption,
) -> list[str]:
    """Detect obvious page-region failures before a synthetic crop is emitted."""
    issues: list[str] = []
    page_rect = page.rect
    width_ratio = rect.width / max(1.0, page_rect.width)
    height_ratio = rect.height / max(1.0, page_rect.height)
    if width_ratio >= 0.72 and height_ratio >= 0.78:
        issues.append("page_like_crop")
        if not crop_has_visual_evidence(page, rect):
            issues.append("page_like_without_visual_evidence")

    for block in collect_text_blocks(page, caption.page_number):
        match = CAPTION_START_RE.match(block.text)
        if match is None or caption_identity_from_match(match) == caption.identity:
            continue
        block_rect = fitz.Rect(block.bbox_pt)
        intersection = block_rect & rect
        if intersection.is_empty:
            continue
        overlap = intersection.get_area() / max(1.0, block_rect.get_area())
        if overlap >= 0.15:
            issues.append("contains_other_visual_caption")
            break
    return issues

def caption_crop_has_blocking_issue(issues: Collection[str]) -> bool:
    """Return True only for failures that make a synthetic crop unsafe."""
    return bool(
        {"contains_other_visual_caption", "page_like_without_visual_evidence"}
        & set(issues)
    )

def crop_has_visual_evidence(page: fitz.Page, rect: fitz.Rect) -> bool:
    """Require meaningful raster or vector content for a page-sized fallback."""
    raster_count = 0
    raster_area = 0.0
    try:
        page_dict = page.get_text("dict")
    except Exception:
        page_dict = None
    if isinstance(page_dict, dict):
        for block in page_dict.get("blocks", []):
            if block.get("type") != 1 or "bbox" not in block:
                continue
            block_rect = fitz.Rect(block["bbox"])
            intersection = block_rect & rect
            if intersection.is_empty:
                continue
            coverage = intersection.get_area() / max(1.0, block_rect.get_area())
            if coverage < 0.50:
                continue
            raster_count += 1
            raster_area += intersection.get_area()
    if raster_count >= 2 or raster_area >= 0.04 * max(1.0, rect.get_area()):
        return True

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    drawing_count = 0
    drawing_item_count = 0
    drawing_area = 0.0
    for drawing in drawings:
        drawing_rect = fitz.Rect(drawing.get("rect", (0.0, 0.0, 0.0, 0.0)))
        intersection = drawing_rect & rect
        if intersection.is_empty:
            continue
        drawing_count += 1
        drawing_item_count += len(drawing.get("items", []))
        drawing_area += intersection.get_area()
        if drawing_count >= 3 or drawing_item_count >= 3:
            return True
    return drawing_area >= 0.02 * max(1.0, rect.get_area())

def _infer_caption_column_rect(page: fitz.Page, caption_rect: fitz.Rect) -> fitz.Rect:
    """Infer a full-width or two-column content corridor for a caption."""
    page_rect = page.rect
    default = fitz.Rect(
        page_rect.x0 + 34.0,
        page_rect.y0,
        page_rect.x1 - 34.0,
        page_rect.y1,
    )
    if caption_rect.width >= 0.52 * page_rect.width:
        return default

    prose_blocks = [
        block
        for block in collect_text_blocks(page, page.number + 1)
        if len(block.text) >= 85
        and len(block.text.split()) >= 12
        and 0.20 * page_rect.width <= fitz.Rect(block.bbox_pt).width <= 0.62 * page_rect.width
    ]
    midpoint = 0.5 * (page_rect.x0 + page_rect.x1)
    left_blocks = [
        block for block in prose_blocks
        if bbox_center_x(block.bbox_pt) <= midpoint - 0.07 * page_rect.width
    ]
    right_blocks = [
        block for block in prose_blocks
        if bbox_center_x(block.bbox_pt) >= midpoint + 0.07 * page_rect.width
    ]
    if len(left_blocks) < 2 or len(right_blocks) < 2:
        # With no strong two-column evidence, a short left-aligned caption must
        # not force a narrow crop. Full-width single-column figures are common.
        return default

    use_left = 0.5 * (caption_rect.x0 + caption_rect.x1) < midpoint
    selected = left_blocks if use_left else right_blocks
    x0 = max(page_rect.x0 + 24.0, min(block.bbox_pt[0] for block in selected) - 8.0)
    x1 = min(page_rect.x1 - 24.0, max(block.bbox_pt[2] for block in selected) + 8.0)
    return fitz.Rect(x0, page_rect.y0, x1, page_rect.y1)

def _collect_visual_evidence_rects(
    page: fitz.Page,
    search_rect: fitz.Rect,
    caption: Caption | None = None,
) -> list[fitz.Rect]:
    """Collect raster/vector bounds and, for Schemes, compact native text."""
    visual: list[fitz.Rect] = []
    try:
        page_dict = page.get_text("dict")
    except Exception:
        page_dict = None
    if isinstance(page_dict, dict):
        for block in page_dict.get("blocks", []):
            if block.get("type") != 1 or "bbox" not in block:
                continue
            rect = fitz.Rect(block["bbox"])
            intersection = rect & search_rect
            if intersection.is_empty or intersection.get_area() < 16.0:
                continue
            area_ratio = rect.get_area() / max(1.0, page.rect.get_area())
            if area_ratio >= 0.88:
                continue
            visual.append(intersection)

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    for drawing in drawings:
        raw_rect = drawing.get("rect")
        if raw_rect is None:
            continue
        rect = fitz.Rect(raw_rect)
        if rect.is_empty and rect.width >= 0.8 and rect.height >= 0.8:
            continue
        # Preserve long zero-thickness strokes by rebuilding the rectangle from
        # coordinates. Mutating ``y0`` before ``y1`` can transiently invalidate a
        # PyMuPDF Rect and silently drop horizontal reaction arrows/bonds.
        x0, y0, x1, y1 = tuple(rect)
        if x1 - x0 < 0.8:
            x0 -= 0.6
            x1 += 0.6
        if y1 - y0 < 0.8:
            y0 -= 0.6
            y1 += 0.6
        rect = fitz.Rect(x0, y0, x1, y1)
        intersection = rect & search_rect
        if intersection.is_empty:
            continue
        if rect.get_area() >= 0.70 * max(1.0, page.rect.get_area()):
            continue
        if max(intersection.width, intersection.height) < 3.0:
            continue
        visual.append(intersection)

    if caption is not None and normalize_caption_kind(caption.caption_kind) == "scheme":
        page_number = page.number + 1
        for block in collect_text_blocks(page, page_number):
            text = clean_text(block.text)
            words = text.split()
            compact_scheme_token = len(text) <= 24 and len(words) <= 3
            if (
                not text
                or CAPTION_START_RE.match(text)
                or (looks_like_heading(text) and not compact_scheme_token)
                or _caption_rect_intersects(
                    fitz.Rect(block.bbox_pt),
                    caption,
                    page_number,
                    tolerance=2.0,
                )
            ):
                continue
            if len(text) > 80 or len(words) > 9:
                continue
            if len(words) >= 7 and re.search(r"[.!?。！？]$", text):
                continue
            rect = fitz.Rect(block.bbox_pt)
            intersection = rect & search_rect
            if intersection.is_empty or max(intersection.width, intersection.height) < 2.0:
                continue
            visual.append(intersection)
    return visual

def _cluster_visual_rects(rects: list[fitz.Rect]) -> list[fitz.Rect]:
    if not rects:
        return []
    parents = list(range(len(rects)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parents[right_root] = left_root

    for left in range(len(rects)):
        for right in range(left + 1, len(rects)):
            first = rects[left]
            second = rects[right]
            x_gap = _rect_axis_gap(first, second, horizontal=True)
            y_gap = _rect_axis_gap(first, second, horizontal=False)
            x_overlap = _axis_overlap_ratio(first.x0, first.x1, second.x0, second.x1)
            y_overlap = _axis_overlap_ratio(first.y0, first.y1, second.y0, second.y1)
            if not (first & second).is_empty:
                union(left, right)
            elif y_overlap >= 0.25 and x_gap <= 22.0:
                union(left, right)
            elif x_overlap >= 0.30 and y_gap <= 22.0:
                union(left, right)

    components: dict[int, fitz.Rect] = {}
    counts: Counter[int] = Counter()
    for index, rect in enumerate(rects):
        root = find(index)
        counts[root] += 1
        if root not in components:
            components[root] = fitz.Rect(rect)
        else:
            components[root] |= rect
    clusters = [
        rect
        for root, rect in components.items()
        if (
            rect.width >= 24.0
            and rect.height >= 18.0
            and (rect.get_area() >= 900.0 or counts[root] >= 3)
        )
    ]
    return sorted(clusters, key=lambda rect: (rect.y0, rect.x0))

def _caption_visual_cluster_score(
    page: fitz.Page,
    caption_rect: fitz.Rect,
    cluster: fitz.Rect,
) -> float:
    x_overlap = _axis_overlap_ratio(
        caption_rect.x0,
        caption_rect.x1,
        cluster.x0,
        cluster.x1,
    )
    edge_delta = min(
        abs(caption_rect.x0 - cluster.x0),
        abs(caption_rect.x1 - cluster.x1),
    ) / max(1.0, page.rect.width)
    area_ratio = cluster.get_area() / max(1.0, page.rect.get_area())
    scores: list[float] = []
    above_gap = caption_rect.y0 - cluster.y1
    if above_gap >= -6.0:
        score = (
            9.5
            - max(0.0, above_gap) / 45.0
            + 2.5 * x_overlap
            + 4.0 * min(1.0, area_ratio / 0.18)
            - 3.0 * edge_delta
        )
        if caption_rect.y0 >= page.rect.y0 + 0.40 * page.rect.height:
            score += 1.2
        scores.append(score)
    below_gap = cluster.y0 - caption_rect.y1
    if below_gap >= -6.0:
        score = (
            9.0
            - max(0.0, below_gap) / 45.0
            + 2.5 * x_overlap
            + 4.0 * min(1.0, area_ratio / 0.18)
            - 3.0 * edge_delta
        )
        if caption_rect.y1 <= page.rect.y0 + 0.32 * page.rect.height:
            score += 1.5
        scores.append(score)
    return max(scores, default=-math.inf)

def _grow_visual_cluster_for_caption(
    page: fitz.Page,
    caption: Caption,
    anchor: fitz.Rect,
    clusters: list[fitz.Rect],
) -> fitz.Rect:
    group = fitz.Rect(anchor)
    used: set[int] = {id(anchor)}
    for _ in range(min(12, len(clusters))):
        ranked: list[tuple[tuple[float, float, float], fitz.Rect]] = []
        for cluster in clusters:
            if id(cluster) in used:
                continue
            rank = _figure_part_merge_rank(page, caption, group, cluster)
            if rank is not None:
                ranked.append((rank, cluster))
        if not ranked:
            break
        ranked.sort(key=lambda item: item[0])
        _, chosen = ranked[0]
        group |= chosen
        used.add(id(chosen))
    return group

def _fallback_region_without_visual_cluster(
    page: fitz.Page,
    caption: Caption,
    caption_rect: fitz.Rect,
    column_rect: fitz.Rect,
    min_height_pt: float,
) -> fitz.Rect:
    """Conservative text-barrier fallback when vector bounds are unavailable."""
    page_rect = page.rect
    above_space = caption_rect.y0 - page_rect.y0
    below_space = page_rect.y1 - caption_rect.y1
    use_above = above_space >= min_height_pt + 20.0 and (
        above_space >= below_space or below_space < min_height_pt + 20.0
    )
    blocks = collect_text_blocks(page, page.number + 1)
    if use_above:
        bottom = caption_rect.y0 - 4.0
        top = page_rect.y0 + 36.0
        for block in blocks:
            block_rect = fitz.Rect(block.bbox_pt)
            if block_rect.y1 >= bottom - min_height_pt:
                continue
            if (block_rect & column_rect).is_empty:
                continue
            is_separator = CAPTION_START_RE.match(block.text) or (
                len(block.text) >= 110 and len(block.text.split()) >= 16
            )
            if is_separator and block_rect.y1 > top:
                top = block_rect.y1 + 6.0
        if bottom - top < min_height_pt:
            top = max(page_rect.y0 + 30.0, bottom - min_height_pt)
        return fitz.Rect(column_rect.x0, top, column_rect.x1, bottom)

    top = caption_rect.y1 + 4.0
    bottom = page_rect.y1 - 36.0
    for block in blocks:
        block_rect = fitz.Rect(block.bbox_pt)
        if block_rect.y0 <= top + min_height_pt:
            continue
        if (block_rect & column_rect).is_empty:
            continue
        is_separator = CAPTION_START_RE.match(block.text) or (
            len(block.text) >= 110 and len(block.text.split()) >= 16
        )
        if is_separator and block_rect.y0 < bottom:
            bottom = block_rect.y0 - 6.0
            break
    if bottom - top < min_height_pt:
        bottom = min(page_rect.y1 - 30.0, top + min_height_pt)
    return fitz.Rect(column_rect.x0, top, column_rect.x1, bottom)

def find_caption_figure_rect(
    page: fitz.Page,
    caption: Caption,
    min_height_pt: float,
) -> fitz.Rect:
    """Locate a complete same-page raster/vector figure adjacent to a caption."""
    page_number = page.number + 1
    caption_bbox = caption_bbox_on_page(caption, page_number) or caption.bbox_pt
    caption_rect = fitz.Rect(caption_bbox)
    column_rect = _infer_caption_column_rect(page, caption_rect)
    search_rect = fitz.Rect(
        column_rect.x0,
        page.rect.y0 + 24.0,
        column_rect.x1,
        page.rect.y1 - 24.0,
    )
    visual_rects = _collect_visual_evidence_rects(page, search_rect, caption)
    clusters = [
        cluster
        for cluster in _cluster_visual_rects(visual_rects)
        if (cluster & caption_rect).get_area() <= 0.10 * max(1.0, caption_rect.get_area())
    ]
    ranked = sorted(
        (
            (_caption_visual_cluster_score(page, caption_rect, cluster), cluster)
            for cluster in clusters
        ),
        key=lambda item: -item[0],
    )
    if ranked and ranked[0][0] >= 5.0:
        if len(ranked) >= 2 and ranked[1][0] >= ranked[0][0] - 0.75:
            first_cluster = ranked[0][1]
            second_cluster = ranked[1][1]
            can_join = (
                _figure_part_merge_rank(page, caption, first_cluster, second_cluster)
                is not None
                or _figure_part_merge_rank(page, caption, second_cluster, first_cluster)
                is not None
            )
            if not can_join:
                return fitz.Rect(0.0, 0.0, 0.0, 0.0)
        anchor = ranked[0][1]
        visual_rect = _grow_visual_cluster_for_caption(page, caption, anchor, clusters)
        visual_rect = expand_figure_rect_with_nearby_content(
            page,
            visual_rect,
            caption,
            (),
        )
        return fitz.Rect(
            max(column_rect.x0, visual_rect.x0),
            max(page.rect.y0, visual_rect.y0),
            min(column_rect.x1, visual_rect.x1),
            min(page.rect.y1, visual_rect.y1),
        )

    return _fallback_region_without_visual_cluster(
        page,
        caption,
        caption_rect,
        column_rect,
        min_height_pt,
    )

def choose_caption(
    captions: list[Caption],
    page_number: int,
    figure_rect: fitz.Rect,
) -> Caption | None:
    """Match one rendered figure to the nearest caption on the same page."""

    same_page = [caption for caption in captions if caption.page_number == page_number]
    if not same_page:
        return None

    below = [caption for caption in same_page if caption.bbox_pt[1] >= figure_rect.y1 - 12]
    if below:
        figure_center_x = (figure_rect.x0 + figure_rect.x1) / 2
        return min(
            below,
            key=lambda caption: (
                abs(caption.bbox_pt[1] - figure_rect.y1),
                abs(bbox_center_x(caption.bbox_pt) - figure_center_x),
            ),
        )

    figure_center_y = (figure_rect.y0 + figure_rect.y1) / 2
    return min(
        same_page,
        key=lambda caption: abs(bbox_center_y(caption.bbox_pt) - figure_center_y),
    )

def match_caption_to_image_block_conservative(
    document: fitz.Document,
    caption: Caption,
    candidates: list[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
) -> ImageBlockCandidate | None:
    """Match a caption to an adjacent image block on the same or prior page."""
    caption_width = max(1.0, caption.bbox_pt[2] - caption.bbox_pt[0])

    def eligible(candidate: ImageBlockCandidate, cross_page: bool) -> bool:
        if image_block_is_consumed(candidate, consumed):
            return False
        rect = candidate.bbox_pt
        image_width = rect[2] - rect[0]
        width_ratio = image_width / caption_width
        if width_ratio < 0.68:
            # Centered schematics and graphical abstracts are often narrower than
            # their full-column caption. Keep accepting only an immediate,
            # strongly centered same-page raster so unrelated side art cannot
            # steal the caption.
            compact_centered = (
                not cross_page
                and width_ratio >= 0.45
                and abs(bbox_center_x(rect) - bbox_center_x(caption.bbox_pt))
                <= max(18.0, 0.08 * caption_width)
            )
            if not compact_centered:
                return False
        if horizontal_overlap_ratio(rect, caption.bbox_pt) < 0.65:
            return False
        if cross_page:
            return True
        gap = caption.bbox_pt[1] - rect[3]
        return -12.0 <= gap <= 55.0

    same_page = [
        candidate
        for candidate in candidates
        if candidate.page_number == caption.page_number
        and eligible(candidate, False)
    ]
    if same_page:
        return min(
            same_page,
            key=lambda candidate: (
                abs(caption.bbox_pt[1] - candidate.bbox_pt[3]),
                abs(bbox_center_x(caption.bbox_pt) - bbox_center_x(candidate.bbox_pt)),
            ),
        )

    def lateral_rank(
        candidate: ImageBlockCandidate,
    ) -> tuple[float, float, float] | None:
        if image_block_is_consumed(candidate, consumed):
            return None
        if candidate.page_number != caption.page_number:
            return None

        rect = candidate.bbox_pt
        image_width = rect[2] - rect[0]
        image_height = rect[3] - rect[1]
        caption_height = caption.bbox_pt[3] - caption.bbox_pt[1]
        if image_width < 1.50 * caption_width:
            return None

        gaps = (
            caption.bbox_pt[0] - rect[2],
            rect[0] - caption.bbox_pt[2],
        )
        side_gap = min((gap for gap in gaps if gap >= -3.0), default=None)
        if side_gap is None or side_gap > 24.0:
            return None

        overlap_y = max(
            0.0,
            min(rect[3], caption.bbox_pt[3])
            - max(rect[1], caption.bbox_pt[1]),
        )
        overlap_ratio = overlap_y / max(
            1.0,
            min(image_height, caption_height),
        )
        top_delta = abs(rect[1] - caption.bbox_pt[1])
        if overlap_ratio < 0.72:
            return None
        if top_delta > max(18.0, 0.18 * caption_height):
            return None
        return side_gap, top_delta, -overlap_ratio

    lateral = [
        (rank, candidate)
        for candidate in candidates
        if (rank := lateral_rank(candidate)) is not None
    ]
    if lateral:
        lateral.sort(key=lambda item: item[0])
        if len(lateral) >= 2:
            best_rank = lateral[0][0]
            next_rank = lateral[1][0]
            if (
                next_rank[0] <= best_rank[0] + 6.0
                and next_rank[1] <= best_rank[1] + 12.0
                and next_rank[2] <= best_rank[2] + 0.10
            ):
                return None
        return lateral[0][1]

    inferred_previous = _match_previous_page_full_figure_conservative(
        document,
        caption,
        candidates,
        consumed,
    )
    if inferred_previous is not None:
        return inferred_previous

    page = document[caption.page_number - 1]
    near_page_top = caption.bbox_pt[1] <= page.rect.y0 + 0.25 * page.rect.height
    if not near_page_top or caption.page_number <= 1:
        return None
    previous_page = document[caption.page_number - 2]
    if not _has_next_page_caption_marker_conservative(previous_page):
        return None
    previous = [
        candidate
        for candidate in candidates
        if candidate.page_number == caption.page_number - 1
        and eligible(candidate, True)
    ]
    if not previous:
        return None
    return max(
        previous,
        key=lambda candidate: (
            candidate.bbox_pt[3],
            (candidate.bbox_pt[2] - candidate.bbox_pt[0])
            * (candidate.bbox_pt[3] - candidate.bbox_pt[1]),
        ),
    )

def _match_previous_page_full_figure_conservative(
    document: fitz.Document,
    caption: Caption,
    candidates: list[ImageBlockCandidate],
    consumed: set[tuple[int, int]],
) -> ImageBlockCandidate | None:
    """Match a bottom-page caption to one dominant raster on the prior page.

    Some journals place a full-page figure first and its caption after the body
    text on the following page. The layout evidence here is intentionally
    strict because there is no explicit cross-page marker in that format.
    """
    if caption.page_number <= 1:
        return None

    caption_page = document[caption.page_number - 1]
    caption_rect = fitz.Rect(caption.bbox_pt)
    page_rect = caption_page.rect
    if caption_rect.y0 < page_rect.y0 + 0.70 * page_rect.height:
        return None

    text_region = fitz.Rect(
        page_rect.x0,
        page_rect.y0 + 35.0,
        page_rect.x1,
        caption_rect.y0 - 4.0,
    )
    prose_blocks, prose_words = _prose_evidence_in_rect_conservative(caption_page, text_region)
    if prose_blocks < 2 or prose_words < 120.0:
        return None

    previous_page = document[caption.page_number - 2]
    previous_rect = previous_page.rect
    strong: list[ImageBlockCandidate] = []
    for candidate in candidates:
        if candidate.page_number != caption.page_number - 1:
            continue
        if image_block_is_consumed(candidate, consumed):
            continue
        rect = fitz.Rect(candidate.bbox_pt)
        width_ratio = rect.width / max(1.0, previous_rect.width)
        height_ratio = rect.height / max(1.0, previous_rect.height)
        area_ratio = rect.get_area() / max(1.0, previous_rect.get_area())
        center_offset = abs(
            0.5 * (rect.x0 + rect.x1) - 0.5 * (previous_rect.x0 + previous_rect.x1)
        ) / max(1.0, previous_rect.width)
        if width_ratio < 0.65 or height_ratio < 0.45 or area_ratio < 0.30:
            continue
        if (
            center_offset > 0.12
            or rect.y1 < previous_rect.y0 + 0.65 * previous_rect.height
        ):
            continue
        if horizontal_overlap_ratio(candidate.bbox_pt, caption.bbox_pt) < 0.60:
            continue
        strong.append(candidate)

    if len(strong) != 1:
        return None

    prior_prose_blocks, prior_prose_words = _prose_evidence_in_rect_conservative(
        previous_page,
        previous_rect,
    )
    if prior_prose_blocks >= 2 and prior_prose_words >= 120.0:
        return None
    return strong[0]

def _has_next_page_caption_marker_conservative(page: fitz.Page) -> bool:
    return bool(re.search(r"caption\s+on\s+(?:the\s+)?next\s+page", page.get_text(), re.IGNORECASE))

def find_caption_crop_conservative(
    document: fitz.Document,
    caption: Caption,
    min_height_pt: float,
) -> tuple[int, fitz.Rect | None]:
    """Return a safe caption-relative crop, including explicit cross-page captions."""
    page = document[caption.page_number - 1]
    if caption.bbox_pt[1] - page.rect.y0 >= min_height_pt + 30.0:
        rect = _find_caption_figure_rect_conservative(page, caption, min_height_pt)
        quality_issues = _caption_crop_quality_issues_conservative(page, rect, caption)
        if (
            rect.height < min_height_pt
            or _is_prose_dominated_crop_conservative(page, rect)
            or _caption_crop_has_blocking_issue_conservative(quality_issues)
        ):
            return caption.page_number, None
        return caption.page_number, rect

    if caption.page_number <= 1:
        return caption.page_number, None
    previous_page = document[caption.page_number - 2]
    marker_blocks = [
        block
        for block in collect_text_blocks(previous_page, caption.page_number - 1)
        if re.search(r"caption\s+on\s+(?:the\s+)?next\s+page", block.text, re.IGNORECASE)
    ]
    if not marker_blocks:
        return caption.page_number, None
    marker = marker_blocks[-1]
    pseudo_caption = Caption(
        page_number=caption.page_number - 1,
        figure_number=caption.figure_number,
        text=caption.text,
        bbox_pt=marker.bbox_pt,
    )
    rect = _find_caption_figure_rect_conservative(previous_page, pseudo_caption, min_height_pt)
    quality_issues = _caption_crop_quality_issues_conservative(previous_page, rect, pseudo_caption)
    if (
        rect.height < min_height_pt
        or _is_prose_dominated_crop_conservative(previous_page, rect)
        or _caption_crop_has_blocking_issue_conservative(quality_issues)
    ):
        return pseudo_caption.page_number, None
    return pseudo_caption.page_number, rect

def _prose_evidence_in_rect_conservative(page: fitz.Page, clip: fitz.Rect) -> tuple[int, float]:
    """Return long-paragraph count and overlap-weighted word count in a region."""
    if clip.is_empty:
        return 0, 0.0

    paragraph_count = 0
    weighted_words = 0.0
    min_block_width = max(72.0, 0.18 * page.rect.width)
    for block in collect_text_blocks(page, 1):
        block_rect = fitz.Rect(block.bbox_pt)
        if block_rect.width < min_block_width:
            continue
        word_count = len(block.text.split())
        if len(block.text) < 120 or word_count < 18:
            continue
        if LEGACY_CAPTION_START_RE.match(block.text):
            continue
        intersection = block_rect & clip
        if intersection.is_empty:
            continue
        overlap = intersection.get_area() / max(1.0, block_rect.get_area())
        if overlap < 0.08:
            continue
        paragraph_count += 1
        weighted_words += word_count * overlap
    return paragraph_count, weighted_words

def _is_prose_dominated_crop_conservative(page: fitz.Page, rect: fitz.Rect) -> bool:
    """Reject synthetic figure crops that mainly contain article paragraphs."""
    prose_blocks, prose_words = _prose_evidence_in_rect_conservative(page, rect)
    return prose_blocks >= 2 and prose_words >= 80.0

def _caption_crop_quality_issues_conservative(
    page: fitz.Page,
    rect: fitz.Rect,
    caption: Caption,
) -> list[str]:
    """Detect obvious page-region failures before a synthetic crop is emitted."""
    issues: list[str] = []
    page_rect = page.rect
    width_ratio = rect.width / max(1.0, page_rect.width)
    height_ratio = rect.height / max(1.0, page_rect.height)
    if width_ratio >= 0.72 and height_ratio >= 0.78:
        issues.append("page_like_crop")
        if not _crop_has_visual_evidence_conservative(page, rect):
            issues.append("page_like_without_visual_evidence")

    for block in collect_text_blocks(page, caption.page_number):
        match = LEGACY_CAPTION_START_RE.match(block.text)
        if match is None or match.group("number") == caption.figure_number:
            continue
        block_rect = fitz.Rect(block.bbox_pt)
        intersection = block_rect & rect
        if intersection.is_empty:
            continue
        overlap = intersection.get_area() / max(1.0, block_rect.get_area())
        if overlap >= 0.15:
            issues.append("contains_other_figure_caption")
            break
    return issues

def _caption_crop_has_blocking_issue_conservative(issues: Collection[str]) -> bool:
    """Return True only for failures that make a synthetic crop unsafe."""
    return bool(
        {"contains_other_figure_caption", "page_like_without_visual_evidence"}
        & set(issues)
    )

def _crop_has_visual_evidence_conservative(page: fitz.Page, rect: fitz.Rect) -> bool:
    """Require meaningful raster or vector content for a page-sized fallback."""
    raster_count = 0
    raster_area = 0.0
    try:
        page_dict = page.get_text("dict")
    except Exception:
        page_dict = None
    if isinstance(page_dict, dict):
        for block in page_dict.get("blocks", []):
            if block.get("type") != 1 or "bbox" not in block:
                continue
            block_rect = fitz.Rect(block["bbox"])
            intersection = block_rect & rect
            if intersection.is_empty:
                continue
            coverage = intersection.get_area() / max(1.0, block_rect.get_area())
            if coverage < 0.50:
                continue
            raster_count += 1
            raster_area += intersection.get_area()
    if raster_count >= 2 or raster_area >= 0.04 * max(1.0, rect.get_area()):
        return True

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    drawing_count = 0
    for drawing in drawings:
        drawing_rect = fitz.Rect(drawing.get("rect", (0.0, 0.0, 0.0, 0.0)))
        if (drawing_rect & rect).is_empty:
            continue
        drawing_count += 1
        if drawing_count >= 3:
            return True
    return False

def _find_caption_figure_rect_conservative(
    page: fitz.Page,
    caption: Caption,
    min_height_pt: float,
) -> fitz.Rect:
    """Locate the complete figure immediately above a formal caption."""
    page_rect = page.rect
    caption_width = caption.bbox_pt[2] - caption.bbox_pt[0]
    if caption_width < 0.50 * page_rect.width:
        left = max(page_rect.x0 + 35.0, caption.bbox_pt[0] - 14.0)
        right = min(page_rect.x1 - 24.0, caption.bbox_pt[2] + 14.0)
    else:
        left = max(page_rect.x0 + 45.0, caption.bbox_pt[0] - 18.0)
        right = min(
            page_rect.x1 - 24.0,
            max(caption.bbox_pt[2] + 18.0, page_rect.x1 - 35.0),
        )
    bottom = max(page_rect.y0, caption.bbox_pt[1] - 4.0)
    top = page_rect.y0 + 45.0

    blocks = collect_text_blocks(page, caption.page_number)
    for block in blocks:
        x0, y0, x1, y1 = block.bbox_pt
        if x1 <= left or x0 >= right or y1 >= bottom - min_height_pt:
            continue
        if LEGACY_CAPTION_START_RE.match(block.text):
            # A previous formal caption separates two figures on the same page.
            # Never let a caption-relative fallback crop reach across it.
            if y1 > top:
                top = y1 + 6.0
            continue
        word_count = len(block.text.split())
        if len(block.text) >= 120 and word_count >= 18 and y1 > top:
            top = y1 + 6.0

    panel_labels = [
        block
        for block in blocks
        if re.match(r"^[A-Za-z](?:\s|$)", block.text)
        and block.bbox_pt[1] >= top
        and block.bbox_pt[3] < bottom
    ]
    if panel_labels:
        top = max(top, min(block.bbox_pt[1] for block in panel_labels) - 8.0)

    if bottom - top < min_height_pt:
        top = max(page_rect.y0 + 40.0, bottom - min_height_pt)
    return fitz.Rect(left, top, right, bottom)
