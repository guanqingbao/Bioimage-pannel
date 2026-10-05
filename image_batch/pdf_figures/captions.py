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

"""Figure/Scheme caption detection, continuation merging, and panel-label parsing."""

# Figure and chemistry Scheme captions supported by the production extractor.
CAPTION_LABEL_PATTERN = (
    r"(?:fig(?:ure)?\.?|图|(?:reaction\s+)?(?:schemes?|sch)\.?)"
)
CAPTION_NUMBER_PATTERN = (
    r"(?:[A-Za-z]?\d+[A-Za-z0-9-]*(?:\.\d+[A-Za-z0-9-]*)*)"
)
CAPTION_START_RE = re.compile(
    rf"^(?P<label>{CAPTION_LABEL_PATTERN})\s*"
    rf"(?P<number>{CAPTION_NUMBER_PATTERN})"
    r"(?P<delimiter>\s*[.:：。|])?\s*(?P<body>.*)",
    re.IGNORECASE | re.DOTALL,
)
CAPTION_INLINE_RE = re.compile(
    rf"(?<![A-Za-z0-9])(?P<label>{CAPTION_LABEL_PATTERN})\s*"
    rf"(?P<number>{CAPTION_NUMBER_PATTERN})"
    r"(?P<delimiter>\s*[.:：。|])?\s*",
    re.IGNORECASE,
)

from .models import (
    BBox,
    Caption,
    CaptionSegment,
    DEFAULT_CAPTION_MERGE_GAP_PT,
    MAX_CROSS_PAGE_DISTANCE,
    TextBlock,
    caption_kind_from_label,
    normalize_caption_kind,
    normalize_figure_number,
)
from .page_context import (
    bbox_center_x,
    bbox_coverage_ratio,
    clean_text,
    collect_text_blocks,
    horizontal_overlap_ratio,
    looks_like_heading,
    union_bbox,
)

def _split_caption_text_block(block: TextBlock) -> list[TextBlock]:
    """Split a one-line block containing multiple ``Figure/Scheme N`` captions.

    PDF producers sometimes place two side-by-side visual captions in one text block.
    PyMuPDF then returns one concatenated string. Bounding boxes are apportioned
    by character position for a single-line block; multi-line blocks retain the
    original box because an exact geometry reconstruction would be unsafe.
    """
    matches = list(CAPTION_INLINE_RE.finditer(block.text))
    if not matches or matches[0].start() > 2:
        return []
    if len(matches) == 1:
        return [block]

    parts: list[TextBlock] = []
    text_length = max(1, len(block.text))
    x0, y0, x1, y1 = block.bbox_pt
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(block.text)
        part_text = clean_text(block.text[start:end])
        if not CAPTION_START_RE.match(part_text):
            continue
        if block.line_count <= 1:
            part_x0 = x0 + (x1 - x0) * start / text_length
            part_x1 = x0 + (x1 - x0) * end / text_length
            bbox = (part_x0, y0, part_x1, y1)
        else:
            bbox = block.bbox_pt
        parts.append(
            TextBlock(
                page_number=block.page_number,
                text=part_text,
                bbox_pt=tuple(float(value) for value in bbox),
                font_size_pt=block.font_size_pt,
                font_name=block.font_name,
                bold=block.bold,
                line_count=block.line_count,
            )
        )
    return parts

def _normalized_font_family(font_name: str) -> str:
    name = str(font_name or "").split("+")[-1].lower()
    return re.sub(
        r"(?:bold|semibold|demibold|italic|oblique|regular|roman|medium|light)",
        "",
        name,
    ).replace("-", "").replace("_", "")

def caption_style_compatible(
    font_size_pt: float,
    font_name: str,
    candidate: TextBlock,
) -> bool:
    """Return whether a following block plausibly continues the same caption."""
    if font_size_pt > 0 and candidate.font_size_pt > 0:
        tolerance = max(1.25, 0.16 * font_size_pt)
        if abs(font_size_pt - candidate.font_size_pt) > tolerance:
            return False
    current_family = _normalized_font_family(font_name)
    candidate_family = _normalized_font_family(candidate.font_name)
    if current_family and candidate_family and current_family != candidate_family:
        if (
            font_size_pt > 0
            and candidate.font_size_pt > 0
            and abs(font_size_pt - candidate.font_size_pt) > 0.6
        ):
            return False
    return True

def _caption_continuation_alignment(left: BBox, right: BBox, page_width: float) -> bool:
    overlap = horizontal_overlap_ratio(left, right)
    center_delta = abs(bbox_center_x(left) - bbox_center_x(right))
    width_ratio = min(
        left[2] - left[0],
        right[2] - right[0],
    ) / max(1.0, max(left[2] - left[0], right[2] - right[0]))
    return overlap >= 0.32 or (
        center_delta <= max(24.0, 0.10 * page_width) and width_ratio >= 0.45
    )

def _cross_page_continuation_score(
    caption: Caption,
    candidate: TextBlock,
    current_page: fitz.Page,
    next_page: fitz.Page,
) -> float:
    if looks_like_heading(candidate.text):
        return -math.inf
    if re.match(
        r"^(?:abstract|introduction|methods?|results?|discussion|conclusions?|"
        r"references?|acknowledg(?:e)?ments?)\b",
        candidate.text,
        re.IGNORECASE,
    ):
        return -math.inf
    if not caption_style_compatible(caption.font_size_pt, caption.font_name, candidate):
        return -math.inf
    last_segment = caption.segments_or_primary()[-1]
    if not _caption_continuation_alignment(
        last_segment.bbox_pt,
        candidate.bbox_pt,
        next_page.rect.width,
    ):
        return -math.inf

    score = 0.0
    bottom_distance = current_page.rect.y1 - last_segment.bbox_pt[3]
    top_distance = candidate.bbox_pt[1] - next_page.rect.y0
    score += max(0.0, 2.5 - bottom_distance / 36.0)
    score += max(0.0, 2.5 - top_distance / 36.0)

    current_text = clean_text(caption.text)
    candidate_text = clean_text(candidate.text)
    repeated = CAPTION_START_RE.match(candidate_text)
    if repeated:
        if caption_identity_from_match(repeated) != caption.identity:
            return -math.inf
        score += 4.0
    elif re.match(r"^[a-z,;:)\]\-]", candidate_text):
        score += 2.5
    elif re.match(
        r"^(?:and|or|where|whereas|which|with|without|for|from|of|to|in|on|"
        r"the|a|an)\b",
        candidate_text,
        re.IGNORECASE,
    ):
        score += 1.8

    if not re.search(r"[.!?][\"')\]]?$", current_text):
        score += 2.0
    elif candidate_text[:1].isupper() and len(candidate_text.split()) >= 12:
        score -= 2.2
    if len(candidate_text) <= 400:
        score += 0.5
    if (
        candidate_text.startswith("(")
        and candidate.bbox_pt[1] <= next_page.rect.y0 + max(85.0, 0.10 * next_page.rect.height)
    ):
        # A caption may resume on the next page with an explanatory clause
        # such as "(Middle columns: ...)" after a complete sentence.
        score += 3.0
    return score

def _merge_cross_page_caption_continuations(
    document: fitz.Document,
    captions: list[Caption],
    page_blocks: dict[int, list[TextBlock]],
    caption_merge_gap_pt: float,
) -> list[Caption]:
    """Join a caption ending near one page bottom with text at the next page top."""
    merged: list[Caption] = []
    consumed_caption_indices: set[int] = set()
    captions_by_page: dict[int, list[tuple[int, Caption]]] = {}
    for index, caption in enumerate(captions):
        captions_by_page.setdefault(caption.page_number, []).append((index, caption))

    for index, original in enumerate(captions):
        if index in consumed_caption_indices:
            continue
        caption = original
        current_page_number = caption.end_page_number
        if current_page_number >= len(document):
            merged.append(caption)
            continue

        current_page = document[current_page_number - 1]
        last_segment = caption.segments_or_primary()[-1]
        bottom_distance = current_page.rect.y1 - last_segment.bbox_pt[3]
        if bottom_distance > max(82.0, 0.14 * current_page.rect.height):
            merged.append(caption)
            continue

        next_page_number = current_page_number + 1
        next_page = document[next_page_number - 1]
        top_limit = next_page.rect.y0 + max(180.0, 0.24 * next_page.rect.height)
        top_blocks = [
            block
            for block in page_blocks.get(next_page_number, [])
            if block.bbox_pt[1] <= top_limit
        ]
        if not top_blocks:
            merged.append(caption)
            continue

        ranked = sorted(
            (
                (
                    _cross_page_continuation_score(
                        caption,
                        block,
                        current_page,
                        next_page,
                    ),
                    block,
                )
                for block in top_blocks
            ),
            key=lambda item: (-item[0], item[1].bbox_pt[1], item[1].bbox_pt[0]),
        )
        if not ranked or ranked[0][0] < 3.0:
            merged.append(caption)
            continue

        best_score, first_block = ranked[0]
        if len(ranked) > 1 and ranked[1][0] >= best_score - 0.65:
            # Two equally plausible top-of-page blocks usually indicate columns;
            # avoid guessing which one continues the caption.
            merged.append(caption)
            continue

        continuation_blocks = [first_block]
        current_bbox = first_block.bbox_pt
        for block in top_blocks:
            if block is first_block or block.bbox_pt[1] < first_block.bbox_pt[1] - 1.0:
                continue
            if CAPTION_START_RE.match(block.text):
                break
            if not should_merge_caption_block(
                current_bbox,
                block,
                max(caption_merge_gap_pt, 10.0),
                font_size_pt=caption.font_size_pt,
                font_name=caption.font_name,
            ):
                continue
            continuation_blocks.append(block)
            current_bbox = union_bbox(current_bbox, block.bbox_pt)

        segment_text = " ".join(block.text for block in continuation_blocks)
        repeated_match = CAPTION_START_RE.match(segment_text)
        append_text = segment_text
        if repeated_match and caption_identity_from_match(repeated_match) == caption.identity:
            append_text = clean_text(repeated_match.group("body"))

        segment_bbox = continuation_blocks[0].bbox_pt
        for block in continuation_blocks[1:]:
            segment_bbox = union_bbox(segment_bbox, block.bbox_pt)
        new_segment = CaptionSegment(
            page_number=next_page_number,
            text=segment_text,
            bbox_pt=segment_bbox,
        )
        caption = Caption(
            page_number=caption.page_number,
            figure_number=caption.figure_number,
            text=clean_text(f"{caption.text} {append_text}"),
            bbox_pt=caption.bbox_pt,
            caption_kind=caption.caption_kind,
            caption_label=caption.caption_label,
            segments=caption.segments_or_primary() + (new_segment,),
            font_size_pt=caption.font_size_pt,
            font_name=caption.font_name,
            bold=caption.bold,
        )

        # Suppress a separately collected repeated ``Figure/Scheme N`` block at the top
        # of the continuation page; the merged caption is the canonical record.
        for other_index, other in captions_by_page.get(next_page_number, []):
            if other_index == index:
                continue
            if other.identity != caption.identity:
                continue
            if bbox_coverage_ratio(other.bbox_pt, segment_bbox) >= 0.55:
                consumed_caption_indices.add(other_index)
        merged.append(caption)

    return merged

def collect_captions(
    document: fitz.Document,
    *,
    pages: set[int] | None = None,
    caption_merge_gap_pt: float = DEFAULT_CAPTION_MERGE_GAP_PT,
) -> list[Caption]:
    """Collect same-page and cross-page formal-caption candidates."""

    if pages:
        scan_pages: set[int] = set()
        for page_number in pages:
            for offset in range(-MAX_CROSS_PAGE_DISTANCE, MAX_CROSS_PAGE_DISTANCE + 1):
                adjacent = page_number + offset
                if 1 <= adjacent <= len(document):
                    scan_pages.add(adjacent)
    else:
        scan_pages = set(range(1, len(document) + 1))

    captions: list[Caption] = []
    page_blocks: dict[int, list[TextBlock]] = {}
    for page_index, page in enumerate(document):
        page_number = page_index + 1
        if page_number not in scan_pages:
            continue
        blocks = collect_text_blocks(page, page_number)
        page_blocks[page_number] = blocks
        index = 0
        while index < len(blocks):
            block = blocks[index]
            split_blocks = _split_caption_text_block(block)
            if not split_blocks:
                index += 1
                continue

            if len(split_blocks) > 1:
                for split_block in split_blocks:
                    match = CAPTION_START_RE.match(split_block.text)
                    if match is None:
                        continue
                    captions.append(
                        Caption(
                            page_number=page_number,
                            figure_number=match.group("number"),
                            text=split_block.text,
                            bbox_pt=split_block.bbox_pt,
                            caption_kind=caption_kind_from_label(match.group("label")),
                            caption_label=clean_text(match.group("label")),
                            segments=(
                                CaptionSegment(
                                    page_number=page_number,
                                    text=split_block.text,
                                    bbox_pt=split_block.bbox_pt,
                                ),
                            ),
                            font_size_pt=split_block.font_size_pt,
                            font_name=split_block.font_name,
                            bold=split_block.bold,
                        )
                    )
                index += 1
                continue

            block = split_blocks[0]
            match = CAPTION_START_RE.match(block.text)
            if match is None:
                index += 1
                continue

            caption_text = block.text
            caption_bbox = block.bbox_pt
            next_index = index + 1
            while next_index < len(blocks):
                candidate = blocks[next_index]
                if CAPTION_START_RE.match(candidate.text):
                    break
                if not should_merge_caption_block(
                    caption_bbox,
                    candidate,
                    caption_merge_gap_pt,
                    font_size_pt=block.font_size_pt,
                    font_name=block.font_name,
                    current_text=caption_text,
                ):
                    break
                caption_text = clean_text(f"{caption_text} {candidate.text}")
                caption_bbox = union_bbox(caption_bbox, candidate.bbox_pt)
                next_index += 1

            captions.append(
                Caption(
                    page_number=page_number,
                    figure_number=match.group("number"),
                    text=caption_text,
                    bbox_pt=caption_bbox,
                    caption_kind=caption_kind_from_label(match.group("label")),
                    caption_label=clean_text(match.group("label")),
                    segments=(
                        CaptionSegment(
                            page_number=page_number,
                            text=caption_text,
                            bbox_pt=caption_bbox,
                        ),
                    ),
                    font_size_pt=block.font_size_pt,
                    font_name=block.font_name,
                    bold=block.bold,
                )
            )
            index = next_index

    return _merge_cross_page_caption_continuations(
        document,
        captions,
        page_blocks,
        caption_merge_gap_pt,
    )

def caption_identity_from_match(match: re.Match[str]) -> tuple[str, str]:
    return (
        caption_kind_from_label(match.group("label")),
        normalize_figure_number(match.group("number")),
    )

def effective_caption_min_height(
    caption: Caption,
    figure_min_height_pt: float,
    scheme_min_height_pt: float,
) -> float:
    """Allow wide, shallow reaction schemes without lowering the figure default."""
    if normalize_caption_kind(caption.caption_kind) == "scheme":
        return min(figure_min_height_pt, scheme_min_height_pt)
    return figure_min_height_pt

def _caption_formality_score(caption: Caption) -> float:
    match = CAPTION_START_RE.match(caption.text)
    if match is None:
        return -math.inf
    body = clean_text(match.group("body"))
    delimiter = clean_text(match.group("delimiter"))
    score = 0.0
    if delimiter:
        score += 5.0
    if 12 <= len(caption.text) <= 1600:
        score += 1.2
    if caption.end_page_number > caption.page_number:
        score += 2.0
    if caption.bold:
        score += 0.4
    if re.match(
        r"^(?:shows?|showed|presents?|illustrates?|depicts?|demonstrates?|"
        r"indicates?|compares?|summarizes?|describes?)\b",
        body,
        re.IGNORECASE,
    ):
        # This wording can be a valid caption. Penalize it only when the formal
        # punctuation after the figure number is absent, rather than discarding it.
        score -= 3.0 if not delimiter else 0.3
    if not delimiter and len(body.split()) >= 18:
        score -= 1.0
    return score

def select_formal_captions(captions: list[Caption]) -> list[Caption]:
    """Keep the strongest candidate for each ``(caption kind, number)`` pair."""
    selected: dict[tuple[str, str], tuple[float, Caption]] = {}
    for caption in captions:
        key = caption.identity
        if not key[1]:
            continue
        score = _caption_formality_score(caption)
        current = selected.get(key)
        rank = (
            score,
            caption.end_page_number - caption.page_number,
            len(caption.text),
            -caption.page_number,
        )
        if current is None:
            selected[key] = (score, caption)
            continue
        current_score, current_caption = current
        current_rank = (
            current_score,
            current_caption.end_page_number - current_caption.page_number,
            len(current_caption.text),
            -current_caption.page_number,
        )
        if rank > current_rank:
            selected[key] = (score, caption)
    return sorted(
        (caption for _, caption in selected.values()),
        key=lambda item: (item.page_number, item.bbox_pt[1], item.bbox_pt[0]),
    )

def parse_expected_panel_labels(caption_text: str) -> list[str]:
    """Parse only an explicit, contiguous A/a-prefixed panel sequence."""
    normalized = clean_text(caption_text).translate(
        str.maketrans({
            "–": "-",  # en dash
            "—": "-",  # em dash
            "−": "-",  # mathematical minus
            "‑": "-",  # non-breaking hyphen
            "‒": "-",  # figure dash
        })
    )
    labels: list[str] = []

    def expand_group(group: str) -> list[str]:
        # Reject statistical and prose parentheses such as (n=3) and (P<0.05).
        group = re.sub(r",\s*(?:and|or)\s+", ",", group, flags=re.IGNORECASE)
        group = re.sub(r"\s*&\s*", ",", group)
        if not re.fullmatch(
            r"\s*[A-Za-z](?:\s*-\s*[A-Za-z])?"
            r"(?:\s*(?:,|;|\b(?:and|or)\b)\s*"
            r"[A-Za-z](?:\s*-\s*[A-Za-z])?)*\s*",
            group,
            re.IGNORECASE,
        ):
            return []
        result: list[str] = []
        for start, end in re.findall(
            r"([A-Za-z])(?:\s*-\s*([A-Za-z]))?",
            re.sub(r"\b(?:and|or)\b", ",", group, flags=re.IGNORECASE),
        ):
            if not end:
                result.append(start)
                continue
            if start.isupper() != end.isupper() or ord(start) > ord(end):
                return []
            result.extend(chr(code) for code in range(ord(start), ord(end) + 1))
        return result

    # A panel group is standalone. Word suffixes such as ``author(s)`` are not
    # figure labels even though their parenthesized content is one letter.
    for group in re.findall(
        r"(?<![A-Za-z0-9])\(([^()]{1,24})\)",
        normalized,
    ):
        labels.extend(expand_group(group))

    # Some publishers use ``a) ... b) ...`` rather than ``(a) ... (b) ...``.
    labels.extend(
        re.findall(r"(?<![A-Za-z0-9])([A-Za-z])\s*\)(?=\s|$)", normalized)
    )

    # Explicit prose forms such as ``panel A`` and ``panels A-C`` are common
    # when only one native label lies outside the raster XObject.
    explicit_panel_labels: list[str] = []
    for start, end in re.findall(
        r"\bpanels?\s*\(?([A-Za-z])\)?"
        r"(?:\s*-\s*\(?([A-Za-z])\)?)?",
        normalized,
        re.IGNORECASE,
    ):
        if not end:
            explicit_panel_labels.append(start)
            continue
        if start.isupper() != end.isupper() or ord(start) > ord(end):
            continue
        explicit_panel_labels.extend(
            chr(code) for code in range(ord(start), ord(end) + 1)
        )
    labels.extend(explicit_panel_labels)

    # A number of journals omit parentheses entirely and enumerate panels as
    # sentence-leading markers: ``A TEM images. B DLS results. ...``.  Treat
    # only uppercase markers at sentence boundaries as panel evidence.  The
    # three-label minimum avoids turning ordinary prose such as ``A
    # comparison ... B cells ...`` into a two-panel constraint.
    bare_panel_labels: list[str] = []
    for match in re.finditer(
        r"(?:^|[.!?;:])\s*\(?([A-Z])\)?"
        r"(?:\s*-\s*\(?([A-Z])\)?)?"
        r"(?=\s+[A-Za-z0-9])",
        normalized.replace("\u00ad", ""),
    ):
        start, end = match.groups()
        if not end:
            bare_panel_labels.append(start)
            continue
        if ord(start) <= ord(end):
            bare_panel_labels.extend(
                chr(code) for code in range(ord(start), ord(end) + 1)
            )

    # Compact sentence-leading lists such as ``B, C CLSM images`` denote two
    # panels but do not match the single-marker expression above.
    for first, second in re.findall(
        r"(?:^|[.!?;:])\s*\(?([A-Z])\)?\s*,\s*"
        r"\(?([A-Z])\)?(?=\s+[A-Za-z0-9])",
        normalized.replace("\u00ad", ""),
    ):
        bare_panel_labels.extend((first, second))

    bare_panel_labels = list(dict.fromkeys(bare_panel_labels))
    if len(bare_panel_labels) >= 3 and "A" in bare_panel_labels:
        labels.extend(bare_panel_labels)

    labels = list(dict.fromkeys(labels))
    valid_sequences: list[list[str]] = []
    for alphabet in ("ABCDEFGHIJKLMNOPQRSTUVWXYZ", "abcdefghijklmnopqrstuvwxyz"):
        same_case = {label for label in labels if label in alphabet}
        ordered = [label for label in alphabet if label in same_case]
        if len(ordered) < 2 or ordered[0] != alphabet[0]:
            continue
        expected_prefix = list(alphabet[: alphabet.index(ordered[-1]) + 1])
        if ordered == expected_prefix:
            valid_sequences.append(ordered)

    # Captions occasionally use uppercase panel labels first and then refer to
    # the final panels in lowercase, for example ``(A) ... (J) ... (k) (l)``.
    # Accept the mixed case only when its case-folded union is still one exact
    # A-prefixed sequence and it extends an otherwise valid same-case prefix.
    folded = {label.upper() for label in labels if len(label) == 1}
    folded_ordered = [label for label in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if label in folded]
    if folded_ordered and folded_ordered[0] == "A":
        folded_prefix = list(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[
                : "ABCDEFGHIJKLMNOPQRSTUVWXYZ".index(folded_ordered[-1]) + 1
            ]
        )
        if folded_ordered == folded_prefix:
            longest_same_case = max(valid_sequences, key=len, default=[])
            if len(folded_ordered) > len(longest_same_case):
                use_upper = (
                    not longest_same_case
                    or longest_same_case[0].isupper()
                )
                return (
                    folded_ordered
                    if use_upper
                    else [label.lower() for label in folded_ordered]
                )
    if len(valid_sequences) == 1:
        return valid_sequences[0]
    explicit_panel_labels = list(dict.fromkeys(explicit_panel_labels))
    if len(explicit_panel_labels) == 1:
        return explicit_panel_labels
    return []

def should_merge_caption_block(
    current_bbox: BBox,
    candidate: TextBlock,
    max_gap_pt: float,
    *,
    font_size_pt: float = 0.0,
    font_name: str = "",
    current_text: str = "",
) -> bool:
    if max_gap_pt <= 0:
        return False
    vertical_gap = candidate.bbox_pt[1] - current_bbox[3]
    if vertical_gap < -2 or vertical_gap > max_gap_pt:
        return False
    if looks_like_heading(candidate.text):
        return False
    if not caption_style_compatible(font_size_pt, font_name, candidate):
        return False
    if (
        current_text
        and len(current_text) < 220
        and re.search(r"[.!?][\"')\]]?$", clean_text(current_text))
        and len(candidate.text) >= 105
        and len(candidate.text.split()) >= 16
        and vertical_gap >= 1.5
    ):
        # A completed short caption followed by a full paragraph is much more
        # likely the start of article prose than a caption continuation.
        return False
    overlap = horizontal_overlap_ratio(current_bbox, candidate.bbox_pt)
    if overlap < 0.35:
        return False
    current_width = max(1.0, current_bbox[2] - current_bbox[0])
    left_delta = abs(candidate.bbox_pt[0] - current_bbox[0])
    if overlap < 0.62 and left_delta > max(24.0, 0.16 * current_width):
        return False
    return True
