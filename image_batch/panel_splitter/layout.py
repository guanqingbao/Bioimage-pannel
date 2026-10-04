from __future__ import annotations

import argparse
import json
import statistics
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.optimize import linear_sum_assignment

"""XY-Cut layout construction, panel boundaries, and repair rules."""

from .detection import *  # noqa: F401,F403 - shared detection primitives

def make_content_mask(image: np.ndarray) -> np.ndarray:
    difference_from_white = np.max(
        255 - image.astype(np.int16),
        axis=2,
    )
    mask = (difference_from_white > 10).astype(np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((2, 2), np.uint8),
    )
    mask = cv2.dilate(
        mask,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )
    return mask

def smooth_profile(profile: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    if window % 2 == 0:
        window += 1
    if window <= 1:
        return profile.astype(float)
    kernel = np.ones(window, float) / window
    return np.convolve(profile, kernel, mode="same")

def line_metrics(
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    orientation: str,
    position: int,
    band: int,
) -> tuple[float, float]:
    x1, y1, x2, y2 = rect
    band = max(1, int(band))

    if orientation == "v":
        start = max(x1, position - band)
        end = min(x2, position + band + 1)
        array = mask[y1:y2, start:end]
        orthogonal_length = array.shape[0]
    else:
        start = max(y1, position - band)
        end = min(y2, position + band + 1)
        array = mask[start:end, x1:x2]
        array = array.T
        orthogonal_length = array.shape[0]

    density = float(array.mean()) if array.size else 1.0
    bin_count = max(
        4,
        min(
            16,
            orthogonal_length // 25
            if orthogonal_length >= 100 else 4,
        ),
    )
    values = [
        float(chunk.mean()) if chunk.size else 1.0
        for chunk in np.array_split(array, bin_count, axis=0)
    ]
    blank_fraction = float(
        np.mean(np.array(values) < 0.035)
    )
    return density, blank_fraction

def run_width(
    profile: np.ndarray,
    position: int,
    threshold: float,
    low: int,
    high: int,
) -> int:
    left = position
    right = position
    while left > low and profile[left - 1] <= threshold:
        left -= 1
    while right < high - 1 and profile[right + 1] <= threshold:
        right += 1
    return right - left + 1

def best_cut_for_gap(
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    labels: list[dict],
    orientation: str,
    low: float,
    high: float,
    reference_height: float,
) -> dict | None:
    x1, y1, x2, y2 = rect

    if high - low < max(3, int(0.5 * reference_height)):
        return None

    if orientation == "v":
        profile = mask[y1:y2, x1:x2].mean(axis=0)
        offset = x1
    else:
        profile = mask[y1:y2, x1:x2].mean(axis=1)
        offset = y1

    band = max(2, round(0.18 * reference_height))
    smoothed = smooth_profile(profile, 2 * band + 1)
    low_index = max(0, int(low - offset))
    high_index = min(len(smoothed) - 1, int(high - offset))

    if high_index <= low_index:
        return None

    segment = smoothed[low_index:high_index + 1]
    candidate_indices = np.argsort(segment)[:min(12, len(segment))]
    best: dict | None = None
    region_mean = float(profile.mean()) + 1e-6

    for local_index in candidate_indices:
        profile_index = int(low_index + local_index)
        position = offset + profile_index
        density, blank_fraction = line_metrics(
            mask,
            rect,
            orientation,
            position,
            band,
        )

        threshold = min(
            0.055,
            max(
                0.012,
                float(np.quantile(profile, 0.18)) * 1.35,
            ),
        )
        width = run_width(
            smoothed,
            profile_index,
            threshold,
            low_index,
            high_index + 1,
        )
        width_score = min(
            1.0,
            width / max(1, 1.0 * reference_height),
        )
        relative_density = density / region_mean

        cost = (
            0.50 * density
            + 0.20 * min(1.5, relative_density) * 0.08
            + 0.20 * (1.0 - blank_fraction) * 0.10
            + 0.10 * (1.0 - width_score) * 0.08
        )

        edge_distance = min(
            profile_index - low_index,
            high_index - profile_index,
        ) / max(1, high_index - low_index)
        cost += 0.010 * (
            1.0 - min(1.0, edge_distance * 4.0)
        )

        # 顶层面板标签通常是左上锚点，边界更靠近下一标签之前。
        distance_from_next = (
            high_index - profile_index
        ) / max(1, high_index - low_index)
        cost += 0.018 * distance_from_next

        candidate = {
            "ori": orientation,
            "pos": int(position),
            "cost": float(cost),
            "density": density,
            "blank": blank_fraction,
            "run": int(width),
            "width_score": width_score,
            "lo": float(low),
            "hi": float(high),
            "rel": relative_density,
        }
        if best is None or candidate["cost"] < best["cost"]:
            best = candidate

    # 只有非常清晰的白色沟槽才采用几何极小值。
    # 无白边或内容贴近时，吸附到下一面板标签之前。
    weak_gutter_min_width = max(
        2,
        int(round(0.25 * reference_height)),
    )
    clear_gutter = best is not None and (
        best["density"] < 0.04
        or (
            best["density"] < 0.08
            and best["blank"] >= 0.50
        )
        or (
            best["density"] < 0.10
            and best["blank"] >= 0.50
            and best["run"] >= weak_gutter_min_width
        )
    )

    if best is not None and not clear_gutter:
        profile_index = high_index
        position = offset + profile_index
        density, blank_fraction = line_metrics(
            mask,
            rect,
            orientation,
            position,
            band,
        )
        threshold = min(
            0.055,
            max(
                0.012,
                float(np.quantile(profile, 0.18)) * 1.35,
            ),
        )
        width = run_width(
            smoothed,
            profile_index,
            threshold,
            low_index,
            high_index + 1,
        )
        width_score = min(
            1.0,
            width / max(1, 1.0 * reference_height),
        )
        relative_density = density / region_mean
        best = {
            "ori": orientation,
            "pos": int(position),
            "cost": float(
                0.50 * density
                + 0.20 * min(1.5, relative_density) * 0.08
                + 0.20 * (1.0 - blank_fraction) * 0.10
                + 0.10 * (1.0 - width_score) * 0.08
            ),
            "density": density,
            "blank": blank_fraction,
            "run": int(width),
            "width_score": width_score,
            "lo": float(low),
            "hi": float(high),
            "rel": relative_density,
            "anchor_fallback": True,
        }

    return best

def sequence_partition_penalty(
    first_group: list[dict],
    second_group: list[dict],
) -> tuple[int, float]:
    """Penalize cuts that interleave the detected top-level label sequence."""
    ordered: list[tuple[int, int]] = []
    for group_index, group in enumerate((first_group, second_group)):
        for label in group:
            letter = str(label.get("letter", ""))
            if len(letter) != 1 or not letter.isalpha() or not letter.isascii():
                return 0, 0.0
            ordered.append((ord(letter.lower()) - ord("a"), group_index))

    ordered.sort(key=lambda item: item[0])
    if len({index for index, _ in ordered}) != len(ordered):
        return 0, 0.0

    switches = sum(
        ordered[index - 1][1] != ordered[index][1]
        for index in range(1, len(ordered))
    )
    # One switch is the normal contiguous split. Extra switches mean that the
    # proposed cut crosses the reading order, such as {b,e} | {c,d,f}.
    penalty = min(0.24, 0.06 * max(0, switches - 1))
    return switches, float(penalty)

def candidate_cuts(
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    labels: list[dict],
    reference_height: float,
) -> list[dict]:
    x1, y1, x2, y2 = rect
    candidates: list[dict] = []

    for orientation in ("h", "v"):
        if orientation == "h":
            ordered = sorted(
                (
                    (label["y"] + 0.5 * label["h"], label)
                    for label in labels
                ),
                key=lambda item: item[0],
            )
            start_key = lambda label: label["y"]
        else:
            ordered = sorted(
                (
                    (label["x"] + 0.5 * label["w"], label)
                    for label in labels
                ),
                key=lambda item: item[0],
            )
            start_key = lambda label: label["x"]

        for index in range(len(ordered) - 1):
            first_position = ordered[index][0]
            second_position = ordered[index + 1][0]

            if second_position - first_position < 1.65 * reference_height:
                continue

            midpoint = (first_position + second_position) / 2
            first_group = [
                label
                for label in labels
                if (
                    label["y"] + 0.5 * label["h"]
                    if orientation == "h"
                    else label["x"] + 0.5 * label["w"]
                ) < midpoint
            ]
            second_group = [
                label for label in labels
                if label not in first_group
            ]

            if not first_group or not second_group:
                continue

            next_start = min(
                start_key(label) for label in second_group
            )
            high = next_start - max(
                1,
                int(0.12 * reference_height),
            )
            low = max(
                first_position + 0.45 * reference_height,
                next_start - 3.4 * reference_height,
            )

            if high <= low:
                low = first_position + 0.30 * (
                    second_position - first_position
                )
                high = first_position + 0.92 * (
                    second_position - first_position
                )

            outer_low = (
                y1 if orientation == "h" else x1
            ) + int(
                0.02
                * (
                    y2 - y1
                    if orientation == "h"
                    else x2 - x1
                )
            )
            outer_high = (
                y2 if orientation == "h" else x2
            ) - int(
                0.02
                * (
                    y2 - y1
                    if orientation == "h"
                    else x2 - x1
                )
            )
            low = max(low, outer_low)
            high = min(high, outer_high)

            if (
                orientation == "h"
                and y1 == 0
                and len(first_group) == 1
                and first_group[0].get("letter") == "A"
            ):
                low = max(low, high - 1.1 * reference_height)

            candidate = best_cut_for_gap(
                mask,
                rect,
                labels,
                orientation,
                low,
                high,
                reference_height,
            )
            if candidate is None:
                continue

            candidate["n1"] = len(first_group)
            candidate["n2"] = len(second_group)
            candidate["gap"] = float(
                second_position - first_position
            )

            balance = min(
                len(first_group),
                len(second_group),
            ) / len(labels)

            if orientation == "v":
                child1 = (x1, y1, candidate["pos"], y2)
                child2 = (candidate["pos"], y1, x2, y2)
            else:
                child1 = (x1, y1, x2, candidate["pos"])
                child2 = (x1, candidate["pos"], x2, y2)

            def corner_distance(
                group: list[dict],
                child: tuple[int, int, int, int],
            ) -> float:
                child_x, child_y, _, _ = child
                return min(
                    max(
                        max(
                            0.0,
                            (label["x"] - child_x)
                            / reference_height,
                        ),
                        max(
                            0.0,
                            (label["y"] - child_y)
                            / reference_height,
                        ),
                    )
                    for label in group
                )

            distance1 = corner_distance(first_group, child1)
            distance2 = corner_distance(second_group, child2)
            anchor_penalty = 0.040 * (
                max(0.0, distance1 - 2.2) ** 1.15
                + max(0.0, distance2 - 2.2) ** 1.15
            )
            anchor_penalty = min(0.85, anchor_penalty)
            layout_bonus = 0.0
            stagger_penalty = 0.0
            clear_candidate_gutter = (
                not candidate.get("anchor_fallback", False)
                and (
                    candidate["density"] < 0.045
                    or candidate["blank"] >= 0.85
                )
            )

            if (
                orientation == "h"
                and len(first_group) >= 2
                and len(second_group) >= 1
            ):
                lower_center = min(
                    label["y"] + 0.5 * label["h"]
                    for label in second_group
                )
                upper_center = max(
                    label["y"] + 0.5 * label["h"]
                    for label in first_group
                )
                lower_is_left_anchor = any(
                    label["x"] - x1 <= 0.18 * (x2 - x1)
                    for label in second_group
                )
                upper_x_centers = [
                    label["x"] + 0.5 * label["w"]
                    for label in first_group
                ]
                upper_spans_columns = (
                    max(upper_x_centers) - min(upper_x_centers)
                    >= 4.0 * reference_height
                )
                if (
                    lower_is_left_anchor
                    and upper_spans_columns
                    and clear_candidate_gutter
                    and lower_center - upper_center
                    >= 2.4 * reference_height
                ):
                    layout_bonus = 0.30

                lower_y_centers = [
                    label["y"] + 0.5 * label["h"]
                    for label in second_group
                ]
                lower_y_span = max(lower_y_centers) - min(lower_y_centers)
                if (
                    len(second_group) >= 2
                    and lower_y_span > 3.0 * reference_height
                    and not clear_candidate_gutter
                ):
                    stagger_penalty = min(
                        0.45,
                        0.045
                        * (lower_y_span / reference_height - 3.0),
                    )

            if orientation == "v" and not clear_candidate_gutter:
                y_centers1 = [
                    label["y"] + 0.5 * label["h"]
                    for label in first_group
                ]
                y_centers2 = [
                    label["y"] + 0.5 * label["h"]
                    for label in second_group
                ]
                y_span = max(
                    max(y_centers1) - min(y_centers1),
                    max(y_centers2) - min(y_centers2),
                )
                if y_span > 6.0 * reference_height:
                    stagger_penalty = max(
                        stagger_penalty,
                        min(
                            0.45,
                            0.025
                            * (y_span / reference_height - 6.0),
                        ),
                    )

            candidate["corner_dist"] = [
                float(distance1),
                float(distance2),
            ]
            candidate["anchor_penalty"] = float(anchor_penalty)
            candidate["layout_bonus"] = float(layout_bonus)
            candidate["stagger_penalty"] = float(stagger_penalty)
            sequence_switches, sequence_penalty = sequence_partition_penalty(
                first_group,
                second_group,
            )
            candidate["sequence_switches"] = int(sequence_switches)
            candidate["sequence_penalty"] = float(sequence_penalty)
            candidate["rank_cost"] = (
                candidate["cost"]
                + anchor_penalty
                + stagger_penalty
                + sequence_penalty
                - 0.008
                * min(2.0, candidate["gap"] / reference_height)
                - 0.004 * balance
                - layout_bonus
            )
            candidates.append(candidate)

    return candidates

def recursive_xycut(
    mask: np.ndarray,
    rect: tuple[int, int, int, int],
    labels: list[dict],
    reference_height: float,
    path: tuple[int, ...] = (),
    depth: int = 0,
) -> tuple[list[dict], list[dict]]:
    if len(labels) <= 1:
        return [
            {
                "rect": tuple(map(int, rect)),
                "labels": labels,
                "path": path,
                "cuts": [],
            }
        ], []

    candidates = candidate_cuts(
        mask,
        rect,
        labels,
        reference_height,
    )
    if not candidates:
        return [
            {
                "rect": tuple(map(int, rect)),
                "labels": labels,
                "path": path,
                "cuts": [],
                "unresolved": True,
            }
        ], []

    best = min(candidates, key=lambda item: item["rank_cost"])
    x1, y1, x2, y2 = rect

    if best["ori"] == "v":
        position = int(best["pos"])
        rect1 = (x1, y1, position, y2)
        rect2 = (position, y1, x2, y2)
        labels1 = [
            label
            for label in labels
            if label["x"] + 0.5 * label["w"] < position
        ]
    else:
        position = int(best["pos"])
        rect1 = (x1, y1, x2, position)
        rect2 = (x1, position, x2, y2)
        labels1 = [
            label
            for label in labels
            if label["y"] + 0.5 * label["h"] < position
        ]

    labels2 = [
        label for label in labels
        if label not in labels1
    ]

    if not labels1 or not labels2:
        return [
            {
                "rect": tuple(map(int, rect)),
                "labels": labels,
                "path": path,
                "cuts": [],
                "unresolved": True,
            }
        ], []

    leaves1, cuts1 = recursive_xycut(
        mask,
        rect1,
        labels1,
        reference_height,
        path + (0,),
        depth + 1,
    )
    leaves2, cuts2 = recursive_xycut(
        mask,
        rect2,
        labels2,
        reference_height,
        path + (1,),
        depth + 1,
    )

    cut_record = dict(best)
    cut_record.update(
        rect=tuple(map(int, rect)),
        labels="".join(
            sorted(label["letter"] for label in labels)
        ),
        depth=depth,
    )
    return leaves1 + leaves2, [cut_record] + cuts1 + cuts2

def crop_confidence(
    leaf: dict,
    cuts: list[dict],
    label: dict,
    image_shape: tuple[int, ...],
    mask: np.ndarray,
    reference_height: float,
) -> tuple[float, float]:
    x1, y1, x2, y2 = leaf["rect"]
    label_quality = float(label.get("quality", 0.8))
    confidence = min(1.0, label_quality / 1.15)

    related_cut_scores: list[float] = []
    for cut in cuts:
        rx1, ry1, rx2, ry2 = cut["rect"]
        if (
            rx1 <= x1 and ry1 <= y1
            and rx2 >= x2 and ry2 >= y2
        ):
            if cut.get("anchor_fallback"):
                whitespace_score = 0.68
            else:
                whitespace_score = (
                    0.55
                    * max(0.0, 1.0 - cut["density"] / 0.08)
                    + 0.30 * cut["blank"]
                    + 0.15 * cut["width_score"]
                )
            related_cut_scores.append(
                max(0.0, min(1.0, whitespace_score))
            )

    if related_cut_scores:
        confidence *= (
            0.65 + 0.35 * min(related_cut_scores)
        )

    if (
        x2 - x1 < 3 * reference_height
        or y2 - y1 < 3 * reference_height
    ):
        confidence *= 0.55

    margin = min(
        label["x"] - x1,
        label["y"] - y1,
        x2 - (label["x"] + label["w"]),
        y2 - (label["y"] + label["h"]),
    )
    if margin < 0:
        confidence *= 0.40

    occupancy = (
        float(mask[y1:y2, x1:x2].mean())
        if x2 > x1 and y2 > y1 else 0.0
    )
    if occupancy < 0.003:
        confidence *= 0.50

    return float(max(0.0, min(1.0, confidence))), occupancy

def assess_composite_figure(
    panels: dict[str, dict],
    panel_labels: dict[str, dict],
    alphabet: str,
    cuts: list[dict],
    image_shape: tuple[int, ...],
    expected_labels: list[str] | None = None,
) -> dict[str, object]:
    """Decide whether proposed crops have enough evidence for a real composite."""
    panel_count = len(panels)
    decision: dict[str, object] = {
        "is_composite": False,
        "classification": "single_or_unconfirmed",
        "reasons": [],
        "panel_count": panel_count,
        "anchor_fraction": 0.0,
        "sequence_continuity": 0.0,
        "low_confidence_fraction": 0.0,
        "strong_cut_fraction": 0.0,
        "expected_labels": expected_labels or [],
    }
    if expected_labels:
        expected_set = set(expected_labels)
        actual_set = set(panels)
        if not expected_set.issubset(actual_set):
            decision["reasons"] = ["caption_label_mismatch"]
            return decision
    if panel_count < 2:
        decision["reasons"] = ["fewer_than_two_panels"]
        return decision

    anchored = 0
    indices: list[int] = []
    for letter, panel in panels.items():
        label = panel_labels.get(letter)
        if label is None:
            continue
        x1, y1, x2, y2 = panel["rect"]
        width = max(1, x2 - x1)
        height = max(1, y2 - y1)
        relative_x = (float(label["x"]) - x1) / width
        relative_y = (float(label["y"]) - y1) / height
        if relative_x <= 0.20 and relative_y <= 0.25:
            anchored += 1
        if letter in alphabet:
            indices.append(alphabet.index(letter))

    anchor_fraction = anchored / panel_count
    low_confidence_fraction = sum(
        float(panel.get("confidence", 0.0)) < 0.70
        for panel in panels.values()
    ) / panel_count
    if indices:
        expected = max(indices) - min(indices) + 1
        continuity = len(set(indices)) / max(1, expected)
    else:
        continuity = 0.0

    strong_cuts = sum(
        float(cut.get("density", 1.0)) <= 0.06
        and float(cut.get("blank", 0.0)) >= 0.75
        and not cut.get("anchor_fallback", False)
        for cut in cuts
    )
    strong_cut_fraction = strong_cuts / max(1, len(cuts))
    vertical_fraction = sum(cut.get("ori") == "v" for cut in cuts) / max(1, len(cuts))
    label_y_values = sorted(float(label["y"]) for label in panel_labels.values())
    image_height = max(1, int(image_shape[0]))
    row_band = 0.15 * image_height
    largest_row_cluster = 0
    for start, y_value in enumerate(label_y_values):
        end = start
        while (
            end < len(label_y_values)
            and label_y_values[end] - y_value <= row_band
        ):
            end += 1
        largest_row_cluster = max(largest_row_cluster, end - start)
    labels_in_single_row = (
        len(label_y_values) >= 3
        and largest_row_cluster / len(label_y_values) >= 0.70
        and vertical_fraction >= 0.70
    )

    minimum_anchor_fraction = 2.0 / 3.0
    reasons: list[str] = list(decision["reasons"])
    if anchor_fraction < minimum_anchor_fraction:
        reasons.append("labels_not_panel_anchors")
    if continuity < 0.80:
        reasons.append("label_sequence_is_sparse")
    if low_confidence_fraction >= 0.40:
        reasons.append("too_many_low_confidence_panels")
    if strong_cut_fraction < 0.50:
        reasons.append("weak_separation_evidence")
    if labels_in_single_row:
        reasons.append("labels_likely_belong_to_one_axis")

    two_panel_balance = 0.0
    two_panel_label_quality = 0.0
    two_panel_leading_edge_aligned = False
    two_panel_cut_blank = 0.0
    two_panel_cut_density = 1.0
    if panel_count == 2 and cuts:
        first_cut = cuts[0]
        two_panel_cut_blank = float(first_cut.get("blank", 0.0))
        two_panel_cut_density = float(first_cut.get("density", 1.0))
        if first_cut.get("ori") == "h":
            spans = [
                max(0, int(panel["rect"][3]) - int(panel["rect"][1]))
                / max(1, int(image_shape[0]))
                for panel in panels.values()
            ]
        else:
            spans = [
                max(0, int(panel["rect"][2]) - int(panel["rect"][0]))
                / max(1, int(image_shape[1]))
                for panel in panels.values()
            ]
        two_panel_balance = min(spans, default=0.0)
        qualities = [
            float(panel_labels.get(letter, {}).get("quality", 0.0))
            for letter in panels
        ]
        two_panel_label_quality = min(qualities, default=0.0)
        labels = [panel_labels.get(letter, {}) for letter in panels]
        if len(labels) == 2 and all("x" in label and "y" in label for label in labels):
            if first_cut.get("ori") == "h":
                leading = [float(label["x"]) for label in labels]
                leading_span = max(leading) - min(leading)
                two_panel_leading_edge_aligned = (
                    max(leading) <= 0.12 * max(1, int(image_shape[1]))
                    and leading_span <= 0.06 * max(1, int(image_shape[1]))
                )
            else:
                leading = [float(label["y"]) for label in labels]
                leading_span = max(leading) - min(leading)
                two_panel_leading_edge_aligned = (
                    max(leading) <= 0.12 * max(1, int(image_shape[0]))
                    and leading_span <= 0.06 * max(1, int(image_shape[0]))
                )

    moderate_quality_with_structure = (
        two_panel_label_quality >= 0.88
        and two_panel_leading_edge_aligned
        and two_panel_balance >= 0.35
        and two_panel_cut_blank >= 0.30
        and two_panel_cut_density <= 0.50
    )

    two_panel_fallback_valid = (
        panel_count == 2
        and anchor_fraction >= 1.0
        and continuity >= 1.0
        and two_panel_balance >= 0.24
        and (
            two_panel_label_quality >= 0.90
            or moderate_quality_with_structure
        )
    )
    unconstrained_two_panel_weak = (
        not expected_labels
        and panel_count == 2
        and strong_cut_fraction < 0.50
        and not two_panel_fallback_valid
    )
    if unconstrained_two_panel_weak:
        reasons.append("two_panel_layout_lacks_strong_separation")

    weak_anchors = "labels_not_panel_anchors" in reasons
    corroborating_reasons = {
        "label_sequence_is_sparse",
        "too_many_low_confidence_panels",
        "weak_separation_evidence",
        "labels_likely_belong_to_one_axis",
    }
    reject = (
        bool(expected_labels) and weak_anchors
    ) or unconstrained_two_panel_weak or (
        weak_anchors
        and any(reason in corroborating_reasons for reason in reasons)
    )
    decision.update(
        is_composite=not reject,
        classification="composite" if not reject else "single_or_unconfirmed",
        reasons=reasons,
        anchor_fraction=float(anchor_fraction),
        sequence_continuity=float(continuity),
        low_confidence_fraction=float(low_confidence_fraction),
        strong_cut_fraction=float(strong_cut_fraction),
        two_panel_balance=float(two_panel_balance),
        two_panel_label_quality=float(two_panel_label_quality),
        two_panel_leading_edge_aligned=two_panel_leading_edge_aligned,
        two_panel_cut_blank=float(two_panel_cut_blank),
        two_panel_cut_density=float(two_panel_cut_density),
    )
    return decision

def assess_detection_layout_evidence(
    image: np.ndarray,
    mask: np.ndarray,
    detection: tuple,
    alphabet: str,
) -> dict[str, object]:
    """Test whether one alphabet hypothesis forms a reliable panel layout."""
    labels = detection[0]
    reference_height = float(detection[3])
    if len(labels) < 3 or reference_height <= 0:
        return {
            "strong": False,
            "origin_anchor": False,
            "resolved_fraction": 0.0,
            "anchor_fraction": 0.0,
            "sequence_continuity": 0.0,
            "strong_cut_fraction": 0.0,
        }

    first_label = next(
        (label for label in labels if label["letter"] == alphabet[0]),
        None,
    )
    origin_anchor = bool(
        first_label is not None
        and float(first_label.get("quality", 0.0)) >= 0.90
        and float(first_label["x"]) < 0.08 * image.shape[1]
        and float(first_label["y"]) < 0.08 * image.shape[0]
    )
    leaves, cuts = recursive_xycut(
        mask,
        (0, 0, image.shape[1], image.shape[0]),
        labels,
        reference_height,
    )
    panels: dict[str, dict] = {}
    panel_labels: dict[str, dict] = {}
    for leaf in leaves:
        if len(leaf["labels"]) != 1:
            continue
        label = leaf["labels"][0]
        panels[label["letter"]] = {
            "rect": leaf["rect"],
            "confidence": 1.0,
        }
        panel_labels[label["letter"]] = label

    decision = assess_composite_figure(
        panels,
        panel_labels,
        alphabet,
        cuts,
        image.shape,
    )
    resolved_fraction = len(panels) / max(1, len(labels))
    anchor_fraction = float(decision["anchor_fraction"])
    continuity = float(decision["sequence_continuity"])
    strong_cut_fraction = float(decision["strong_cut_fraction"])
    strong = bool(
        origin_anchor
        and decision["is_composite"]
        and resolved_fraction >= 0.85
        and anchor_fraction >= 0.80
        and continuity >= 0.90
        and strong_cut_fraction >= 0.75
    )
    return {
        "strong": strong,
        "origin_anchor": origin_anchor,
        "resolved_fraction": float(resolved_fraction),
        "anchor_fraction": anchor_fraction,
        "sequence_continuity": continuity,
        "strong_cut_fraction": strong_cut_fraction,
    }

def load_figure_constraints(path: Path) -> dict[str, object]:
    """Load caption/native-label constraints written by the PDF extractor."""
    metadata_path = path.parent.parent / "figures_metadata.json"
    if not metadata_path.is_file():
        return {"expected_labels": [], "native_labels": []}
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"expected_labels": [], "native_labels": []}

    resolved = str(path.resolve())
    for figure in payload.get("figures", []):
        image_path = Path(str(figure.get("image_path", "")))
        if str(image_path.resolve()) == resolved or image_path.name == path.name:
            return {
                "expected_labels": list(figure.get("expected_panel_labels", [])),
                "native_labels": list(figure.get("native_panel_labels", [])),
            }
    return {"expected_labels": [], "native_labels": []}

def align_caption_labels_to_detection_case(
    expected_labels: list[str],
    detection: tuple,
) -> list[str]:
    """Match caption label case to the alphabet selected from the figure image."""
    if not expected_labels:
        return []
    if not all(
        len(label) == 1 and label.upper() in UPPER_LETTERS
        for label in expected_labels
    ):
        return list(expected_labels)

    alphabet = str(detection[7].get("alphabet", ""))
    if alphabet == UPPER_LETTERS:
        return [label.upper() for label in expected_labels]
    if alphabet == LOWER_LETTERS:
        return [label.lower() for label in expected_labels]
    return list(expected_labels)

def apply_native_label_evidence(
    labels: list[dict],
    native_labels: list[dict],
    expected_labels: list[str],
    reference_height: float,
) -> int:
    """Annotate CV glyphs corroborated by native PDF text without creating labels."""
    native_by_label = {
        str(item.get("label", "")).upper(): item
        for item in native_labels
        if len(item.get("anchor_px", [])) == 2
    }
    if not expected_labels or not native_by_label:
        return 0

    match_count = 0
    max_distance = max(24.0, 1.0 * reference_height)
    for label in labels:
        native = native_by_label.get(str(label.get("letter", "")).upper())
        if native is None:
            continue
        anchor_x, anchor_y = map(float, native["anchor_px"])
        center_x = float(label["x"]) + 0.5 * float(label["w"])
        center_y = float(label["y"]) + 0.5 * float(label["h"])
        distance = float(np.hypot(center_x - anchor_x, center_y - anchor_y))
        if distance > max_distance:
            continue
        label["source"] = "pdf_text+cv"
        label["native_anchor_distance"] = distance
        match_count += 1
    return match_count

def apply_caption_sequence_retry(
    detection: tuple,
    expected_labels: list[str],
    image_shape: tuple[int, ...],
) -> tuple[tuple, bool]:
    """Retry geometry with a fully observed caption sequence after CV pruning."""
    if not expected_labels:
        return detection, False

    current_labels = detection[0]
    expected_set = set(expected_labels)
    current_set = {str(label.get("letter", "")) for label in current_labels}
    if expected_set.issubset(current_set):
        return detection, False

    assignment_by_letter = {
        str(row.get("letter", "")): row
        for row in detection[1]
        if not row.get("missing", True)
    }
    if any(label not in assignment_by_letter for label in expected_labels):
        return detection, False

    retry_labels: list[dict] = []
    for expected in expected_labels:
        candidate = dict(assignment_by_letter[expected])
        if float(candidate.get("quality", 0.0)) < 0.95:
            return detection, False
        candidate["letter"] = expected
        candidate["source"] = "caption_sequence_retry"
        retry_labels.append(candidate)

    image_height, image_width = image_shape[:2]
    centers_x = [float(label["x"] + 0.5 * label["w"]) for label in retry_labels]
    centers_y = [float(label["y"] + 0.5 * label["h"]) for label in retry_labels]
    span_x = max(centers_x) - min(centers_x)
    span_y = max(centers_y) - min(centers_y)
    if min(centers_y) > 0.18 * image_height:
        return detection, False

    compact_column = span_x < 0.16 * image_width
    if compact_column and (
        statistics.median(centers_x) > 0.30 * image_width
        or span_y < 0.20 * image_height
    ):
        return detection, False

    label_info = detection[7]
    label_info["caption_sequence_retry"] = True
    return (
        (
            retry_labels,
            detection[1],
            detection[2],
            detection[3],
            len(expected_labels),
            detection[5],
            detection[6],
            label_info,
        ),
        True,
    )

def apply_caption_sequence_upper_bound(
    detection: tuple,
    expected_labels: list[str],
) -> tuple[tuple, list[str]]:
    """Prune CV labels beyond a caption sequence after visual promotion.

    ``promote_strong_visual_caption_extension`` must run first.  Therefore a
    caption remains the upper bound only when the longer visual sequence did
    not form a strongly resolved panel layout.  This still removes chart and
    legend glyphs without truncating a genuine figure whose caption parser
    observed only its first few panel references.
    """
    if not expected_labels:
        return detection, []

    alphabet = (
        UPPER_LETTERS if expected_labels[0].isupper() else LOWER_LETTERS
    )
    if expected_labels != list(alphabet[: len(expected_labels)]):
        return detection, []

    labels = list(detection[0])
    current_letters = {
        str(label.get("letter", ""))
        for label in labels
    }
    expected_set = set(expected_labels)
    if not expected_set.issubset(current_letters):
        return detection, []

    last_expected_index = alphabet.index(expected_labels[-1])
    trailing = [
        label
        for label in labels
        if (
            str(label.get("letter", "")) in alphabet
            and alphabet.index(str(label.get("letter", "")))
            > last_expected_index
        )
    ]
    if not trailing:
        return detection, []

    non_trailing_extras = current_letters - expected_set - {
        str(label.get("letter", ""))
        for label in trailing
    }
    if non_trailing_extras:
        return detection, []

    retained = [
        label
        for label in labels
        if str(label.get("letter", "")) in expected_set
    ]
    pruned_letters = sorted(
        str(label.get("letter", ""))
        for label in trailing
    )
    label_info = dict(detection[7])
    label_info["caption_pruned_trailing_labels"] = pruned_letters
    return (
        (
            retained,
            detection[1],
            detection[2],
            detection[3],
            len(expected_labels),
            detection[5],
            detection[6],
            label_info,
        ),
        pruned_letters,
    )


def promote_strong_visual_caption_extension(
    detection: tuple,
    expected_labels: list[str],
    layout_evidence: dict[str, object],
) -> tuple[tuple, list[str], bool]:
    """Let a strong contiguous visual layout extend an incomplete caption.

    Caption parsing is intentionally conservative and may stop at ``(A-B)``
    even when the rendered figure visibly continues through ``L``.  Promote
    only an exact A/a-prefixed sequence whose complete XY-Cut layout already
    passed the strict visual-evidence gates.
    """
    if not expected_labels or not bool(layout_evidence.get("strong", False)):
        return detection, expected_labels, False

    alphabet = (
        UPPER_LETTERS if expected_labels[0].isupper() else LOWER_LETTERS
    )
    if expected_labels != list(alphabet[: len(expected_labels)]):
        return detection, expected_labels, False

    current_letters = {
        str(label.get("letter", ""))
        for label in detection[0]
        if str(label.get("letter", "")) in alphabet
    }
    ordered = [letter for letter in alphabet if letter in current_letters]
    # A single trailing glyph is the common false-positive case this caption
    # guard was introduced for.  Caption parsing failures that justify an
    # override usually omit a continuation block and therefore lose several
    # consecutive panels at once.
    if len(ordered) < len(expected_labels) + 2:
        return detection, expected_labels, False
    if ordered != list(alphabet[: len(ordered)]):
        return detection, expected_labels, False

    label_info = dict(detection[7])
    promoted = ordered[len(expected_labels) :]
    label_info.update(
        caption_visual_extension_promoted=promoted,
        caption_visual_extension_evidence=dict(layout_evidence),
    )
    return (
        (
            detection[0],
            detection[1],
            detection[2],
            detection[3],
            len(ordered),
            detection[5],
            detection[6],
            label_info,
        ),
        ordered,
        True,
    )

def prune_nested_caption_terminal_grid_label(
    detection: tuple,
    expected_labels: list[str],
) -> tuple[tuple, list[str], list[str]]:
    """Drop a caption terminal label embedded inside a complete two-column grid."""
    labels = list(detection[0])
    if (
        len(expected_labels) < 5
        or len(expected_labels) % 2 == 0
        or len(labels) != len(expected_labels)
        or [str(label.get("letter", "")) for label in labels]
        != expected_labels
    ):
        return detection, expected_labels, []

    prefix = labels[:-1]
    row_pairs = [prefix[index:index + 2] for index in range(0, len(prefix), 2)]
    reference_height = max(float(detection[3]), 1.0)
    if any(
        abs(
            (float(pair[0]["y"]) + 0.5 * float(pair[0].get("h", 0.0)))
            - (float(pair[1]["y"]) + 0.5 * float(pair[1].get("h", 0.0)))
        )
        > 2.5 * reference_height
        for pair in row_pairs
    ):
        return detection, expected_labels, []

    left_x = [
        float(pair[0]["x"]) + 0.5 * float(pair[0].get("w", 0.0))
        for pair in row_pairs
    ]
    right_x = [
        float(pair[1]["x"]) + 0.5 * float(pair[1].get("w", 0.0))
        for pair in row_pairs
    ]
    left_center = float(statistics.median(left_x))
    right_center = float(statistics.median(right_x))
    column_gap = right_center - left_center
    if (
        column_gap < 8.0 * reference_height
        or max(left_x) - min(left_x) > 2.5 * reference_height
        or max(right_x) - min(right_x) > 2.5 * reference_height
    ):
        return detection, expected_labels, []

    row_y = [
        statistics.mean(
            float(label["y"]) + 0.5 * float(label.get("h", 0.0))
            for label in pair
        )
        for pair in row_pairs
    ]
    row_gaps = [later - earlier for earlier, later in zip(row_y, row_y[1:])]
    if not row_gaps or min(row_gaps) < 5.0 * reference_height:
        return detection, expected_labels, []

    terminal = labels[-1]
    terminal_x = float(terminal["x"]) + 0.5 * float(terminal.get("w", 0.0))
    terminal_y = float(terminal["y"]) + 0.5 * float(terminal.get("h", 0.0))
    typical_row_gap = float(statistics.median(row_gaps))
    terminal_advance = terminal_y - row_y[-1]
    nested_in_right_column = (
        abs(terminal_x - right_center) <= 0.25 * column_gap
        and abs(terminal_x - right_center) < abs(terminal_x - left_center)
        and terminal_advance > 3.0 * reference_height
        and terminal_advance < 0.78 * typical_row_gap
    )
    if not nested_in_right_column:
        return detection, expected_labels, []

    pruned_letter = str(terminal.get("letter", ""))
    label_info = dict(detection[7])
    label_info["caption_pruned_nested_terminal_labels"] = [pruned_letter]
    constrained_expected = expected_labels[:-1]
    return (
        (
            prefix,
            detection[1],
            detection[2],
            detection[3],
            len(prefix),
            detection[5],
            detection[6],
            label_info,
        ),
        constrained_expected,
        [pruned_letter],
    )

def repair_regular_grid_label_outlier(
    labels: list[dict],
    expected_labels: list[str],
    reference_height: float,
    image_shape: tuple[int, ...] | None = None,
) -> tuple[list[dict], list[dict[str, object]]]:
    """Replace one geometrically impossible anchor in a strong regular grid."""
    if len(labels) < 6 or len(labels) != len(expected_labels):
        return labels, []
    if [str(label.get("letter", "")) for label in labels] != expected_labels:
        return labels, []

    count = len(labels)
    ref = max(float(reference_height), 1.0)
    centers = np.asarray(
        [
            (
                float(label["x"]) + 0.5 * float(label["w"]),
                float(label["y"]) + 0.5 * float(label["h"]),
            )
            for label in labels
        ],
        dtype=np.float64,
    )
    candidates: list[dict[str, object]] = []

    for columns in range(2, min(8, count // 2) + 1):
        if count % columns:
            continue
        rows = count // columns
        if rows < 2:
            continue

        for outlier_index in range(count):
            row_centers: list[float] = []
            column_centers: list[float] = []
            valid = True
            for row in range(rows):
                values = [
                    centers[index, 1]
                    for index in range(count)
                    if index != outlier_index and index // columns == row
                ]
                if len(values) < 2:
                    valid = False
                    break
                row_centers.append(float(np.median(values)))
            if not valid:
                continue
            for column in range(columns):
                values = [
                    centers[index, 0]
                    for index in range(count)
                    if index != outlier_index and index % columns == column
                ]
                if not values:
                    valid = False
                    break
                column_centers.append(float(np.median(values)))
            if not valid:
                continue

            column_gaps = np.diff(column_centers)
            row_gaps = np.diff(row_centers)
            if (
                np.any(column_gaps <= 4.0 * ref)
                or np.any(row_gaps <= 4.0 * ref)
            ):
                continue
            median_column_gap = float(np.median(column_gaps))
            median_row_gap = float(np.median(row_gaps))
            if (
                float(np.max(np.abs(column_gaps - median_column_gap)))
                > 0.28 * median_column_gap
                or float(np.max(np.abs(row_gaps - median_row_gap)))
                > 0.35 * median_row_gap
            ):
                continue

            residuals: list[float] = []
            for index, (center_x, center_y) in enumerate(centers):
                target_x = column_centers[index % columns]
                target_y = row_centers[index // columns]
                residuals.append(
                    float(np.hypot(center_x - target_x, center_y - target_y))
                    / ref
                )
            other_residuals = [
                value
                for index, value in enumerate(residuals)
                if index != outlier_index
            ]
            outlier_residual = residuals[outlier_index]
            if (
                max(other_residuals) > 2.0
                or statistics.mean(other_residuals) > 0.85
                or outlier_residual < 4.0
                or outlier_residual < 4.0 * max(
                    statistics.median(other_residuals),
                    0.75,
                )
            ):
                continue

            target_x = column_centers[outlier_index % columns]
            target_y = row_centers[outlier_index // columns]
            if image_shape is not None:
                image_height, image_width = image_shape[:2]
                if not (
                    0.0 <= target_x < float(image_width)
                    and 0.0 <= target_y < float(image_height)
                ):
                    continue
            if any(
                index != outlier_index
                and np.hypot(
                    centers[index, 0] - target_x,
                    centers[index, 1] - target_y,
                )
                < 2.0 * ref
                for index in range(count)
            ):
                continue

            candidates.append(
                {
                    "columns": columns,
                    "rows": rows,
                    "outlier_index": outlier_index,
                    "target_x": target_x,
                    "target_y": target_y,
                    "outlier_residual": outlier_residual,
                    "mean_other_residual": statistics.mean(other_residuals),
                }
            )

    if not candidates:
        return labels, []
    candidates.sort(
        key=lambda candidate: (
            float(candidate["mean_other_residual"]),
            -float(candidate["outlier_residual"]),
        )
    )
    best = candidates[0]
    comparable = [
        candidate
        for candidate in candidates[1:]
        if int(candidate["outlier_index"]) != int(best["outlier_index"])
        and float(candidate["mean_other_residual"])
        <= float(best["mean_other_residual"]) + 0.15
    ]
    if comparable:
        return labels, []

    repaired = [dict(label) for label in labels]
    index = int(best["outlier_index"])
    item = repaired[index]
    old_xy = [int(item["x"]), int(item["y"])]
    item["x"] = int(round(float(best["target_x"]) - 0.5 * float(item["w"])))
    item["y"] = int(round(float(best["target_y"]) - 0.5 * float(item["h"])))
    item["source"] = "caption_grid_interpolation"
    item["grid_original_xy"] = old_xy
    item["grid_columns"] = int(best["columns"])
    item["grid_rows"] = int(best["rows"])
    item["quality"] = min(float(item.get("quality", 0.0)), 0.82)
    repair = {
        "letter": expected_labels[index],
        "old_xy": old_xy,
        "new_xy": [int(item["x"]), int(item["y"])],
        "columns": int(best["columns"]),
        "rows": int(best["rows"]),
        "normalized_residual": float(best["outlier_residual"]),
    }
    return repaired, [repair]

def repair_caption_regular_grid_from_candidates(
    labels: list[dict],
    candidates: list[dict],
    expected_labels: list[str],
    reference_height: float,
    image_shape: tuple[int, ...],
) -> tuple[list[dict], list[dict[str, object]]]:
    """Recover several bad anchors when strong candidates form a regular grid."""
    if (
        len(labels) < 6
        or len(labels) != len(expected_labels)
        or [str(label.get("letter", "")) for label in labels]
        != expected_labels
    ):
        return labels, []

    count = len(labels)
    ref = max(float(reference_height), 1.0)
    image_height, _ = image_shape[:2]
    alphabet = (
        UPPER_LETTERS if expected_labels[0].isupper() else LOWER_LETTERS
    )

    def center(item: dict) -> tuple[float, float]:
        return (
            float(item["x"]) + 0.5 * float(item["w"]),
            float(item["y"]) + 0.5 * float(item["h"]),
        )

    def options(letter: str, target_x: float) -> list[dict]:
        letter_index = alphabet.index(letter)
        result: list[dict] = []
        for raw in candidates:
            if str(raw.get("top", "")) != letter:
                continue
            ratio = float(raw["h"]) / ref
            if not 0.70 <= ratio <= 1.50:
                continue
            shape_score = float(raw["sims"][letter_index])
            if shape_score < 0.82:
                continue
            candidate_x, _ = center(raw)
            x_residual = abs(candidate_x - target_x) / ref
            if x_residual > 2.5:
                continue
            if float(raw.get("iso", 0.0)) < 0.55:
                continue
            if float(raw.get("gutter", 0.0)) < 0.78:
                continue
            item = dict(raw)
            item["letter"] = letter
            item["shape_for_letter"] = shape_score
            item["grid_option_score"] = (
                shape_score
                + 0.08 * float(raw.get("iso", 0.0))
                + 0.04 * float(raw.get("gutter", 0.0))
                - 0.04 * x_residual
            )
            result.append(item)
        return sorted(
            result,
            key=lambda item: float(item["grid_option_score"]),
            reverse=True,
        )[:8]

    hypotheses: list[dict[str, object]] = []
    for columns in range(2, min(6, count // 2) + 1):
        if count % columns:
            continue
        rows = count // columns
        first_row = labels[:columns]
        first_centers = [center(label) for label in first_row]
        if max(y for _, y in first_centers) - min(
            y for _, y in first_centers
        ) > 2.0 * ref:
            continue
        column_centers = [x for x, _ in first_centers]
        if any(
            right - left <= 4.0 * ref
            for left, right in zip(column_centers, column_centers[1:])
        ):
            continue

        selected = [dict(label) for label in first_row]
        row_centers = [statistics.median(y for _, y in first_centers)]
        valid = True
        for row in range(1, rows):
            row_start = row * columns
            first_options = options(
                expected_labels[row_start],
                column_centers[0],
            )
            best_row: tuple[float, list[dict], float] | None = None
            for first_option in first_options:
                _, anchor_y = center(first_option)
                if anchor_y <= row_centers[-1] + 4.0 * ref:
                    continue
                row_items = [first_option]
                score = float(first_option["grid_option_score"])
                for column in range(1, columns):
                    letter = expected_labels[row_start + column]
                    matches = []
                    for option in options(letter, column_centers[column]):
                        _, option_y = center(option)
                        y_residual = abs(option_y - anchor_y) / ref
                        if y_residual <= 1.5:
                            matches.append(
                                (
                                    float(option["grid_option_score"])
                                    - 0.08 * y_residual,
                                    option,
                                )
                            )
                    if not matches:
                        break
                    matches.sort(key=lambda pair: pair[0], reverse=True)
                    option_score, option = matches[0]
                    row_items.append(option)
                    score += option_score
                if len(row_items) != columns:
                    continue
                candidate_row_y = statistics.median(
                    center(item)[1] for item in row_items
                )
                if best_row is None or score > best_row[0]:
                    best_row = (score, row_items, candidate_row_y)
            if best_row is None:
                valid = False
                break
            selected.extend(best_row[1])
            row_centers.append(float(best_row[2]))

        if not valid or len(selected) != count:
            continue
        row_gaps = np.diff(row_centers)
        if (
            np.any(row_gaps <= 4.0 * ref)
            or float(np.max(row_gaps)) > 2.0 * float(np.min(row_gaps))
        ):
            continue
        if row_centers[-1] >= image_height + 0.5 * ref:
            continue

        selected_centers = [center(item) for item in selected]
        current_centers = [center(item) for item in labels]
        bad_indices = [
            index
            for index, (current_xy, selected_xy) in enumerate(
                zip(current_centers, selected_centers)
            )
            if np.hypot(
                current_xy[0] - selected_xy[0],
                current_xy[1] - selected_xy[1],
            )
            > 3.0 * ref
        ]
        if len(bad_indices) < 2:
            continue
        mean_score = statistics.mean(
            float(item.get("grid_option_score", 1.0))
            for item in selected[columns:]
        )
        hypotheses.append(
            {
                "columns": columns,
                "rows": rows,
                "selected": selected,
                "bad_indices": bad_indices,
                "score": mean_score - 0.02 * len(bad_indices),
            }
        )

    if not hypotheses:
        return labels, []
    hypotheses.sort(key=lambda item: float(item["score"]), reverse=True)
    best = hypotheses[0]
    if (
        len(hypotheses) > 1
        and int(hypotheses[1]["columns"]) != int(best["columns"])
        and float(hypotheses[1]["score"]) >= float(best["score"]) - 0.03
    ):
        return labels, []

    repaired = [dict(label) for label in labels]
    repairs: list[dict[str, object]] = []
    selected = list(best["selected"])
    for index in list(best["bad_indices"]):
        old = repaired[index]
        replacement = dict(selected[index])
        replacement["letter"] = expected_labels[index]
        replacement["quality"] = min(
            1.12,
            max(
                0.82,
                float(replacement.get("shape_for_letter", 0.82))
                + 0.10,
            ),
        )
        replacement["source"] = "caption_candidate_grid"
        replacement["grid_original_xy"] = [int(old["x"]), int(old["y"])]
        replacement["grid_columns"] = int(best["columns"])
        replacement["grid_rows"] = int(best["rows"])
        repaired[index] = replacement
        repairs.append(
            {
                "letter": expected_labels[index],
                "old_xy": [int(old["x"]), int(old["y"])],
                "new_xy": [
                    int(replacement["x"]),
                    int(replacement["y"]),
                ],
                "columns": int(best["columns"]),
                "rows": int(best["rows"]),
                "reason": "strong_regular_grid_candidates",
            }
        )
    return repaired, repairs


def repair_caption_regular_grid_from_geometry(
    labels: list[dict],
    expected_labels: list[str],
    reference_height: float,
    image_shape: tuple[int, ...],
) -> tuple[list[dict], list[dict[str, object]]]:
    """Repair several false anchors from a strongly supported row-major grid.

    Glyph matching is deliberately permissive because panel letters may use
    many fonts.  A side effect is that an isolated ``h`` in a legend or a thin
    image feature resembling ``i`` can win the global glyph assignment.  When
    a complete sequence is available and at least two rows already agree on
    one regular column lattice, geometry is stronger evidence than the
    isolated glyph score.  The sequence normally comes from the caption, but
    an exact visually detected A.../a... prefix is sufficient when no caption
    is available.  This fallback moves only the spatial outliers; it does not
    invent a grid for irregular layouts.
    """
    if len(labels) < 6:
        return labels, []

    current_letters = [str(label.get("letter", "")) for label in labels]
    effective_expected = list(expected_labels)
    if not effective_expected:
        if current_letters[0].isupper():
            alphabet = UPPER_LETTERS
        elif current_letters[0].islower():
            alphabet = LOWER_LETTERS
        else:
            return labels, []
        inferred = list(alphabet[: len(current_letters)])
        if current_letters != inferred:
            return labels, []
        effective_expected = inferred
    if (
        len(labels) != len(effective_expected)
        or current_letters != effective_expected
    ):
        return labels, []
    expected_labels = effective_expected

    count = len(labels)
    ref = max(float(reference_height), 1.0)
    image_height, image_width = image_shape[:2]

    def center(item: dict) -> tuple[float, float]:
        return (
            float(item["x"]) + 0.5 * float(item["w"]),
            float(item["y"]) + 0.5 * float(item["h"]),
        )

    centers = [center(label) for label in labels]
    hypotheses: list[dict[str, object]] = []
    for columns in range(2, min(6, count // 2) + 1):
        if count % columns:
            continue
        rows = count // columns
        if rows < 2:
            continue

        # A trustworthy row has co-linear anchors in reading order.  Requiring
        # two agreeing rows prevents an irregular montage from being coerced
        # into a grid merely because its caption contains a contiguous range.
        row_hypotheses: list[tuple[int, list[float], float]] = []
        for row in range(rows):
            row_centers = centers[row * columns : (row + 1) * columns]
            xs = [point[0] for point in row_centers]
            ys = [point[1] for point in row_centers]
            if max(ys) - min(ys) > 2.0 * ref:
                continue
            gaps = np.diff(xs)
            if np.any(gaps <= 4.0 * ref):
                continue
            median_gap = float(np.median(gaps))
            if (
                len(gaps) > 1
                and float(np.max(np.abs(gaps - median_gap)))
                > 0.28 * median_gap
            ):
                continue
            row_hypotheses.append((row, xs, float(np.median(ys))))

        if len(row_hypotheses) < 2:
            continue

        column_centers = [
            float(np.median([row[1][column] for row in row_hypotheses]))
            for column in range(columns)
        ]
        column_gaps = np.diff(column_centers)
        if np.any(column_gaps <= 4.0 * ref):
            continue
        median_column_gap = float(np.median(column_gaps))
        if (
            len(column_gaps) > 1
            and float(np.max(np.abs(column_gaps - median_column_gap)))
            > 0.28 * median_column_gap
        ):
            continue

        # The agreeing rows must describe the same columns, not merely two
        # unrelated horizontal groups with internally regular spacing.
        alignment_residuals = [
            abs(x - column_centers[column]) / ref
            for _, xs, _ in row_hypotheses
            for column, x in enumerate(xs)
        ]
        if max(alignment_residuals) > 2.5:
            continue

        row_centers: list[float | None] = [None] * rows
        for row, _, row_y in row_hypotheses:
            row_centers[row] = row_y

        # A damaged row may still contain one or more correct anchors (``g``
        # in an otherwise bad ``g/h/i`` row).  Use only anchors close to the
        # established column lattice to estimate that row's vertical level.
        for row in range(rows):
            if row_centers[row] is not None:
                continue
            supported_y = []
            for column in range(columns):
                point_x, point_y = centers[row * columns + column]
                if abs(point_x - column_centers[column]) <= 2.5 * ref:
                    supported_y.append(point_y)
            if supported_y and max(supported_y) - min(supported_y) <= 2.0 * ref:
                row_centers[row] = float(np.median(supported_y))

        known_rows = [
            (row, float(value))
            for row, value in enumerate(row_centers)
            if value is not None
        ]
        if len(known_rows) < 2:
            continue
        known_steps = [
            (right_y - left_y) / (right_row - left_row)
            for (left_row, left_y), (right_row, right_y)
            in zip(known_rows, known_rows[1:])
            if right_row > left_row
        ]
        if not known_steps or any(step <= 4.0 * ref for step in known_steps):
            continue
        median_row_gap = float(np.median(known_steps))
        if (
            len(known_steps) > 1
            and float(np.max(np.abs(np.asarray(known_steps) - median_row_gap)))
            > 0.38 * median_row_gap
        ):
            continue

        # Interpolate a wholly missed row only after a stable row pitch exists.
        anchor_row, anchor_y = known_rows[0]
        for row, value in enumerate(row_centers):
            if value is None:
                row_centers[row] = anchor_y + (row - anchor_row) * median_row_gap

        resolved_rows = [float(value) for value in row_centers]
        if any(
            value < 0.0 or value >= float(image_height)
            for value in resolved_rows
        ):
            continue
        row_gaps = np.diff(resolved_rows)
        if np.any(row_gaps <= 4.0 * ref):
            continue
        median_resolved_gap = float(np.median(row_gaps))
        if (
            len(row_gaps) > 1
            and float(np.max(np.abs(row_gaps - median_resolved_gap)))
            > 0.38 * median_resolved_gap
        ):
            continue

        targets = [
            (column_centers[index % columns], resolved_rows[index // columns])
            for index in range(count)
        ]
        residuals = [
            float(np.hypot(point[0] - target[0], point[1] - target[1])) / ref
            for point, target in zip(centers, targets)
        ]
        bad_indices = [
            index for index, residual in enumerate(residuals) if residual > 3.0
        ]
        inlier_indices = [
            index for index, residual in enumerate(residuals) if residual <= 2.0
        ]
        if not (2 <= len(bad_indices) <= max(2, int(np.ceil(0.34 * count)))):
            continue
        if len(inlier_indices) < max(5, int(np.ceil(0.66 * count))):
            continue

        # Every row and column must retain real visual support.  This prevents
        # extrapolating an entire missing side of an otherwise irregular image.
        inlier_rows = {index // columns for index in inlier_indices}
        inlier_columns = {index % columns for index in inlier_indices}
        if len(inlier_rows) != rows or len(inlier_columns) != columns:
            continue
        if not all(0.0 <= x < float(image_width) for x in column_centers):
            continue

        hypotheses.append(
            {
                "columns": columns,
                "rows": rows,
                "targets": targets,
                "bad_indices": bad_indices,
                "score": (
                    statistics.mean(residuals[index] for index in inlier_indices)
                    + 0.10 * len(bad_indices)
                ),
            }
        )

    if not hypotheses:
        return labels, []
    hypotheses.sort(key=lambda item: float(item["score"]))
    best = hypotheses[0]
    if (
        len(hypotheses) > 1
        and int(hypotheses[1]["columns"]) != int(best["columns"])
        and float(hypotheses[1]["score"]) <= float(best["score"]) + 0.15
    ):
        return labels, []

    repaired = [dict(label) for label in labels]
    repairs: list[dict[str, object]] = []
    targets = list(best["targets"])
    for index in list(best["bad_indices"]):
        item = repaired[index]
        old_xy = [int(item["x"]), int(item["y"])]
        target_x, target_y = targets[index]
        item["x"] = int(round(float(target_x) - 0.5 * float(item["w"])))
        item["y"] = int(round(float(target_y) - 0.5 * float(item["h"])))
        item["source"] = "caption_grid_geometry_interpolation"
        item["grid_original_xy"] = old_xy
        item["grid_columns"] = int(best["columns"])
        item["grid_rows"] = int(best["rows"])
        item["quality"] = min(float(item.get("quality", 0.0)), 0.82)
        repairs.append(
            {
                "letter": expected_labels[index],
                "old_xy": old_xy,
                "new_xy": [int(item["x"]), int(item["y"])],
                "columns": int(best["columns"]),
                "rows": int(best["rows"]),
                "reason": "strong_row_major_grid_geometry",
            }
        )
    return repaired, repairs

def repair_caption_row_start_outlier(
    labels: list[dict],
    candidates: list[dict],
    expected_labels: list[str],
    reference_height: float,
    image_shape: tuple[int, ...],
) -> tuple[list[dict], list[dict[str, object]]]:
    """Move a false singleton row label back to a repeated row-start anchor."""
    if (
        len(labels) < 5
        or len(labels) != len(expected_labels)
        or [str(label.get("letter", "")) for label in labels]
        != expected_labels
    ):
        return labels, []

    ref = max(float(reference_height), 1.0)
    _, image_width = image_shape[:2]
    alphabet = (
        UPPER_LETTERS if expected_labels[0].isupper() else LOWER_LETTERS
    )

    def center(item: dict) -> tuple[float, float]:
        return (
            float(item["x"]) + 0.5 * float(item["w"]),
            float(item["y"]) + 0.5 * float(item["h"]),
        )

    rows: list[list[int]] = []
    for index, label in enumerate(labels):
        if not rows:
            rows.append([index])
            continue
        previous = labels[rows[-1][-1]]
        previous_x, previous_y = center(previous)
        current_x, current_y = center(label)
        if (
            current_y - previous_y > 2.5 * ref
            or current_x <= previous_x + 1.0 * ref
        ):
            rows.append([index])
        else:
            rows[-1].append(index)
    if len(rows) < 3:
        return labels, []

    row_start_x = [center(labels[row[0]])[0] for row in rows]
    repaired = [dict(label) for label in labels]
    repairs: list[dict[str, object]] = []
    for row_index, row in enumerate(rows):
        if len(row) != 1:
            continue
        peers = [
            value
            for index, value in enumerate(row_start_x)
            if index != row_index
        ]
        if len(peers) < 2:
            continue
        target_x = float(statistics.median(peers))
        peer_residuals = [abs(value - target_x) for value in peers]
        if statistics.median(peer_residuals) > 2.0 * ref:
            continue

        label_index = row[0]
        current = repaired[label_index]
        current_x, _ = center(current)
        if abs(current_x - target_x) < max(4.0 * ref, 0.12 * image_width):
            continue

        previous_y = (
            statistics.median(
                center(labels[index])[1] for index in rows[row_index - 1]
            )
            if row_index > 0
            else -float("inf")
        )
        next_y = (
            statistics.median(
                center(labels[index])[1] for index in rows[row_index + 1]
            )
            if row_index + 1 < len(rows)
            else float("inf")
        )
        letter = expected_labels[label_index]
        letter_index = alphabet.index(letter)
        options: list[tuple[float, dict]] = []
        for raw in candidates:
            if str(raw.get("top", "")) != letter:
                continue
            ratio = float(raw["h"]) / ref
            if not 0.70 <= ratio <= 1.50:
                continue
            shape_score = float(raw["sims"][letter_index])
            if shape_score < 0.84:
                continue
            option_x, option_y = center(raw)
            x_residual = abs(option_x - target_x) / ref
            if x_residual > 2.5:
                continue
            if not (
                option_y > previous_y + 2.5 * ref
                and option_y < next_y - 2.5 * ref
            ):
                continue
            if float(raw.get("iso", 0.0)) < 0.50:
                continue
            score = (
                shape_score
                + 0.08 * float(raw.get("iso", 0.0))
                + 0.04 * float(raw.get("gutter", 0.0))
                - 0.05 * x_residual
            )
            options.append((score, raw))
        if not options:
            continue
        options.sort(key=lambda pair: pair[0], reverse=True)
        _, raw = options[0]
        replacement = dict(raw)
        replacement["letter"] = letter
        replacement["quality"] = min(
            1.12,
            max(0.82, float(replacement["sims"][letter_index]) + 0.10),
        )
        replacement["source"] = "caption_row_start_candidate"
        replacement["row_start_original_xy"] = [
            int(current["x"]),
            int(current["y"]),
        ]
        repaired[label_index] = replacement
        repairs.append(
            {
                "letter": letter,
                "old_xy": [int(current["x"]), int(current["y"])],
                "new_xy": [int(replacement["x"]), int(replacement["y"])],
                "reason": "repeated_row_start_anchor",
            }
        )
    return repaired, repairs

def repair_caption_stacked_side_leader_outlier(
    labels: list[dict],
    expected_labels: list[str],
    reference_height: float,
    image_shape: tuple[int, ...],
) -> tuple[list[dict], list[dict[str, object]]]:
    """Repair a false second leader in two repeated side-stack blocks."""
    count = len(labels)
    if (
        count < 8
        or count % 2
        or count != len(expected_labels)
        or [str(label.get("letter", "")) for label in labels]
        != expected_labels
    ):
        return labels, []

    ref = max(float(reference_height), 1.0)
    image_height, image_width = image_shape[:2]
    half = count // 2
    follower_count = half - 1
    if follower_count < 3:
        return labels, []

    def center(item: dict) -> tuple[float, float]:
        return (
            float(item["x"]) + 0.5 * float(item["w"]),
            float(item["y"]) + 0.5 * float(item["h"]),
        )

    first_leader = labels[0]
    second_leader = labels[half]
    first_followers = labels[1:half]
    second_followers = labels[half + 1 :]
    first_centers = [center(item) for item in first_followers]
    second_centers = [center(item) for item in second_followers]
    first_x = [x for x, _ in first_centers]
    second_x = [x for x, _ in second_centers]
    first_y = [y for _, y in first_centers]
    second_y = [y for _, y in second_centers]

    if (
        max(first_x) - min(first_x) > 1.75 * ref
        or max(second_x) - min(second_x) > 1.75 * ref
        or abs(statistics.median(first_x) - statistics.median(second_x))
        > 1.75 * ref
    ):
        return labels, []
    if any(right - left <= 4.0 * ref for left, right in zip(first_y, first_y[1:])):
        return labels, []
    if any(right - left <= 4.0 * ref for left, right in zip(second_y, second_y[1:])):
        return labels, []

    first_gaps = [right - left for left, right in zip(first_y, first_y[1:])]
    second_gaps = [right - left for left, right in zip(second_y, second_y[1:])]
    typical_gap = statistics.median(first_gaps + second_gaps)
    if typical_gap <= 0:
        return labels, []
    if any(not 0.65 * typical_gap <= gap <= 1.45 * typical_gap for gap in first_gaps + second_gaps):
        return labels, []
    block_gap = second_y[0] - first_y[-1]
    if not 0.65 * typical_gap <= block_gap <= 1.55 * typical_gap:
        return labels, []

    first_leader_x, first_leader_y = center(first_leader)
    right_column_x = statistics.median(first_x + second_x)
    if right_column_x - first_leader_x < max(8.0 * ref, 0.18 * image_width):
        return labels, []
    if abs(first_leader_y - first_y[0]) > 2.5 * ref:
        return labels, []

    current_x, current_y = center(second_leader)
    target_x = first_leader_x
    target_y = second_y[0]
    if abs(current_x - target_x) <= 2.5 * ref and abs(current_y - target_y) <= 3.0 * ref:
        return labels, []
    if not (
        0 <= target_x < image_width
        and first_y[-1] + 4.0 * ref < target_y < image_height
        and current_y < target_y - 4.0 * ref
    ):
        return labels, []

    repaired = [dict(label) for label in labels]
    replacement = dict(second_leader)
    old_xy = [int(replacement["x"]), int(replacement["y"])]
    replacement["x"] = int(round(target_x - 0.5 * float(replacement["w"])))
    replacement["y"] = int(round(target_y - 0.5 * float(replacement["h"])))
    replacement["quality"] = min(float(replacement.get("quality", 0.0)), 0.82)
    replacement["source"] = "caption_stacked_side_inference"
    replacement["stack_original_xy"] = old_xy
    repaired[half] = replacement
    return repaired, [
        {
            "letter": expected_labels[half],
            "old_xy": old_xy,
            "new_xy": [int(replacement["x"]), int(replacement["y"])],
            "followers_per_block": follower_count,
            "reason": "repeated_side_stack_block",
        }
    ]

def apply_caption_layout_retry(
    image: np.ndarray,
    ocr_words: list[dict] | None,
    detection: tuple,
    expected_labels: list[str],
    content_mask: np.ndarray | None = None,
) -> tuple[tuple, bool]:
    """Recover a caption-bounded sequence only when XY-cut confirms its layout."""
    if len(expected_labels) < 2:
        return detection, False

    current_set = {str(label.get("letter", "")) for label in detection[0]}
    expected_set = set(expected_labels)
    current_complete = expected_set.issubset(current_set)

    alphabet = (
        UPPER_LETTERS if expected_labels[0].isupper() else LOWER_LETTERS
    )
    if expected_labels != list(alphabet[: len(expected_labels)]):
        return detection, False

    mask = content_mask
    complete_suspect_letters: set[str] = set()
    if current_complete:
        if mask is None:
            mask = make_content_mask(image)
        current_layout = assess_detection_layout_evidence(
            image,
            mask,
            detection,
            alphabet,
        )
        current_by_letter = {
            str(row.get("letter", "")): row
            for row in detection[0]
        }
        ordered_current = [current_by_letter[label] for label in expected_labels]
        reverse_x = max(2.0 * float(detection[3]), 0.08 * image.shape[1])
        reverse_y = max(1.5 * float(detection[3]), 0.025 * image.shape[0])
        reverse_diagonal = False
        for previous, current in zip(
            ordered_current,
            ordered_current[1:],
        ):
            if not (
                float(current["x"]) < float(previous["x"]) - reverse_x
                and float(current["y"]) < float(previous["y"]) - reverse_y
            ):
                continue
            reverse_diagonal = True
            complete_suspect_letters.update(
                {
                    str(previous["letter"]),
                    str(current["letter"]),
                }
            )
        # A complete sequence can still contain correctly shaped letters from
        # titles such as "24 h" or thin chart strokes assigned as ``i``.  Skip
        # the expensive forced retry only when nearly every label already sits
        # at a valid panel anchor and the sequence has no impossible backwards
        # move on both page axes.  A legitimate row or column wrap reverses at
        # most one axis, never both at once.
        if (
            bool(current_layout.get("strong", False))
            and float(current_layout.get("anchor_fraction", 0.0)) >= 0.95
            and float(current_layout.get("sequence_continuity", 0.0)) >= 1.0
            and not reverse_diagonal
        ):
            return detection, False

    # A complete-but-geometrically-suspicious sequence already carries the
    # full raw candidate pool, so reuse it instead of running the expensive
    # glyph detector a second time.  Forced redetection is reserved for an
    # actually missing caption label.
    forced = detection if current_complete else detect_panel_labels_for_alphabet(
        image,
        ocr_words,
        alphabet,
        minimum_sequence_length=len(expected_labels),
    )
    assignment_by_letter = {
        str(row.get("letter", "")): row
        for row in forced[1]
        if not row.get("missing", True)
    }
    if any(label not in assignment_by_letter for label in expected_labels):
        return detection, False

    current_by_letter = {
        str(row.get("letter", "")): row
        for row in detection[0]
    }
    recovered: list[dict] = []
    for expected in expected_labels:
        source = (
            current_by_letter[expected]
            if current_complete
            else assignment_by_letter[expected]
        )
        candidate = dict(source)
        candidate["letter"] = expected
        candidate["source"] = "caption_layout_retry"
        recovered.append(candidate)

    qualities = [float(label.get("quality", 0.0)) for label in recovered]
    mean_quality = statistics.mean(qualities)
    if min(qualities) < 0.70 or mean_quality < 0.82:
        return detection, False

    image_height, image_width = image.shape[:2]
    first = recovered[0]
    if (
        float(first.get("quality", 0.0)) < 0.90
        or float(first["x"]) >= 0.15 * image_width
        or float(first["y"]) >= 0.12 * image_height
    ):
        return detection, False

    reference_height = float(forced[3])
    centers = [
        (
            float(label["x"]) + 0.5 * float(label["w"]),
            float(label["y"]) + 0.5 * float(label["h"]),
        )
        for label in recovered
    ]
    for index, center in enumerate(centers):
        for other in centers[index + 1 :]:
            if np.hypot(center[0] - other[0], center[1] - other[1]) < max(
                8.0,
                0.75 * reference_height,
            ):
                return detection, False

    if mask is None:
        mask = make_content_mask(image)

    def option_quality(candidate: dict, letter_index: int) -> float:
        ratio = float(candidate["h"]) / max(reference_height, 1e-6)
        size_score = float(
            np.exp(
                -0.5
                * (np.log(max(ratio, 1e-4)) / 0.27) ** 2
            )
        )
        shape_score = float(candidate["sims"][letter_index])
        quality = (
            0.66 * shape_score
            + 0.16 * size_score
            + 0.08 * (1.0 - float(candidate["darkring"]))
            + 0.05 * float(candidate["iso"])
            + 0.05 * float(candidate["gutter"])
        )
        quality -= candidate_word_penalty(candidate)
        quality -= float(candidate.get("neighbor_word_penalty", 0.0))
        quality -= float(candidate.get("ocr_penalty", 0.0))
        quality -= candidate_edge_penalty(
            candidate,
            image_width,
            image_height,
        )
        letter = alphabet[letter_index]
        if candidate["top"] == letter:
            quality += 0.08
        initial = forced[6]
        if (
            letter in initial
            and candidate["idx"] == initial[letter]["idx"]
        ):
            quality += 0.16
        if shape_score < 0.52:
            quality -= 0.16
        return float(quality)

    def evaluate_layout(
        labels: list[dict],
    ) -> tuple[float, dict, dict[str, dict], dict[str, dict], list[dict]]:
        leaves, cuts = recursive_xycut(
            mask,
            (0, 0, image_width, image_height),
            labels,
            reference_height,
        )
        panels: dict[str, dict] = {}
        panel_labels: dict[str, dict] = {}
        for leaf in leaves:
            if len(leaf["labels"]) != 1:
                continue
            label = leaf["labels"][0]
            panels[label["letter"]] = {
                "rect": leaf["rect"],
                "confidence": 1.0,
            }
            panel_labels[label["letter"]] = label

        layout = assess_composite_figure(
            panels,
            panel_labels,
            alphabet,
            cuts,
            image.shape,
        )
        resolved_fraction = len(panels) / len(expected_labels)
        candidate_qualities = [
            float(label.get("quality", 0.0)) for label in labels
        ]
        score = (
            5.0 * resolved_fraction
            + 4.0 * float(layout["anchor_fraction"])
            + 2.0 * float(layout["strong_cut_fraction"])
            + 0.35 * statistics.mean(candidate_qualities)
            + 0.25 * min(candidate_qualities)
        )
        if not layout["is_composite"]:
            score -= 0.80
        if "labels_likely_belong_to_one_axis" in layout["reasons"]:
            score -= 2.0
        return score, layout, panels, panel_labels, cuts

    options_by_letter: dict[str, list[dict]] = {}
    for letter_index, expected in enumerate(expected_labels):
        by_index: dict[int, dict] = {
            int(recovered[letter_index]["idx"]): recovered[letter_index]
        }
        if (
            current_complete
            and complete_suspect_letters
            and expected not in complete_suspect_letters
        ):
            options_by_letter[expected] = list(by_index.values())
            continue
        for raw_candidate in forced[2]:
            ratio = float(raw_candidate["h"]) / max(reference_height, 1e-6)
            if not 0.70 <= ratio <= 1.55:
                continue
            if float(raw_candidate["topsim"]) <= 0.48:
                continue
            quality = option_quality(raw_candidate, letter_index)
            if quality < 0.70:
                continue
            candidate = dict(raw_candidate)
            candidate["letter"] = expected
            candidate["quality"] = quality
            candidate["shape_for_letter"] = float(
                candidate["sims"][letter_index]
            )
            candidate["source"] = "caption_layout_retry"
            by_index[int(candidate["idx"])] = candidate
        # Repeated unit text such as ``0 h / 6 h / 12 h / 24 h`` can yield
        # several slightly higher-scoring copies of the same glyph.  Preserve
        # the nearest same-row follower of the preceding panel label even when
        # it falls outside the four best template matches.  Keeping the list
        # capped at four avoids multiplying XY-Cut retry time on large figures.
        ranked_options = sorted(
            by_index.values(),
            key=lambda candidate: float(candidate.get("quality", 0.0)),
            reverse=True,
        )
        shortlist = ranked_options[:4]
        if letter_index > 0 and len(ranked_options) > 4:
            previous = recovered[letter_index - 1]
            previous_x = float(previous["x"]) + 0.5 * float(previous["w"])
            previous_y = float(previous["y"]) + 0.5 * float(previous["h"])
            same_row_followers = [
                candidate
                for candidate in ranked_options
                if (
                    float(candidate["x"]) + 0.5 * float(candidate["w"])
                    > previous_x + reference_height
                    and abs(
                        float(candidate["y"])
                        + 0.5 * float(candidate["h"])
                        - previous_y
                    )
                    <= 2.0 * reference_height
                )
            ]
            if same_row_followers:
                nearest_follower = min(
                    same_row_followers,
                    key=lambda candidate: (
                        float(candidate["x"])
                        + 0.5 * float(candidate["w"])
                        - previous_x,
                        abs(
                            float(candidate["y"])
                            + 0.5 * float(candidate["h"])
                            - previous_y
                        ),
                    ),
                )
                selected_indices = {
                    int(candidate["idx"])
                    for candidate in shortlist
                }
                if int(nearest_follower["idx"]) not in selected_indices:
                    shortlist = ranked_options[:3] + [nearest_follower]
        options_by_letter[expected] = shortlist

    replacements: list[dict[str, object]] = []
    current_score, _, _, _, _ = evaluate_layout(recovered)
    # A caption sequence can contain several individually strong false glyphs
    # from legends or prose (for example b in "biofilm" and h in "healing").
    # Let the full XY-Cut score replace a small minority of assignments rather
    # than stopping after two corrections.  The +0.08 improvement requirement
    # and the final layout gates still apply to every accepted replacement.
    max_replacements = min(4, max(2, int(np.ceil(0.34 * len(expected_labels)))))
    if current_complete and complete_suspect_letters:
        max_replacements = min(
            max_replacements,
            len(complete_suspect_letters),
        )
    for _ in range(max_replacements):
        best_score = current_score + 0.08
        best_trial: list[dict] | None = None
        best_replacement: dict[str, object] | None = None
        used_indices = {int(label["idx"]) for label in recovered}
        for letter_index, expected in enumerate(expected_labels):
            current = recovered[letter_index]
            for option in options_by_letter[expected]:
                option_index = int(option["idx"])
                if option_index == int(current["idx"]):
                    continue
                if option_index in used_indices:
                    continue
                trial = [dict(label) for label in recovered]
                trial[letter_index] = dict(option)
                trial_score, _, _, _, _ = evaluate_layout(trial)
                if trial_score <= best_score:
                    continue
                best_score = trial_score
                best_trial = trial
                best_replacement = {
                    "letter": expected,
                    "old_xy": [int(current["x"]), int(current["y"])],
                    "new_xy": [int(option["x"]), int(option["y"])],
                }
        if best_trial is None or best_replacement is None:
            break
        recovered = best_trial
        current_score = best_score
        replacements.append(best_replacement)

    recovered, grid_repairs = repair_regular_grid_label_outlier(
        recovered,
        expected_labels,
        reference_height,
        image.shape,
    )
    if current_complete and not replacements and not grid_repairs:
        return detection, False

    qualities = [float(label.get("quality", 0.0)) for label in recovered]
    mean_quality = statistics.mean(qualities)
    centers = [
        (
            float(label["x"]) + 0.5 * float(label["w"]),
            float(label["y"]) + 0.5 * float(label["h"]),
        )
        for label in recovered
    ]
    for index, center in enumerate(centers):
        for other in centers[index + 1 :]:
            if np.hypot(center[0] - other[0], center[1] - other[1]) < max(
                8.0,
                0.75 * reference_height,
            ):
                return detection, False

    _, _, preliminary_panels, preliminary_labels, cuts = evaluate_layout(
        recovered
    )
    if len(preliminary_panels) != len(expected_labels):
        return detection, False

    layout = assess_composite_figure(
        preliminary_panels,
        preliminary_labels,
        alphabet,
        cuts,
        image.shape,
        expected_labels,
    )
    if (
        not layout["is_composite"]
        or float(layout["anchor_fraction"]) < 0.75
        or float(layout["sequence_continuity"]) < 1.0
        or float(layout["strong_cut_fraction"]) < 0.50
    ):
        return detection, False

    label_info = dict(detection[7])
    label_info.update(
        selected="upper" if alphabet == UPPER_LETTERS else "lower",
        alphabet=alphabet,
        caption_layout_retry=True,
        caption_layout_replacements=replacements,
        caption_grid_repairs=grid_repairs,
        caption_layout_min_quality=min(qualities),
        caption_layout_mean_quality=mean_quality,
        caption_layout_anchor_fraction=float(layout["anchor_fraction"]),
        caption_layout_strong_cut_fraction=float(
            layout["strong_cut_fraction"]
        ),
    )
    return (
        (
            recovered,
            forced[1],
            forced[2],
            reference_height,
            len(expected_labels),
            forced[5],
            forced[6],
            label_info,
        ),
        True,
    )

def trim_left_edge_to_label_band(
    rect: tuple[int, int, int, int],
    label: dict,
    mask: np.ndarray,
    reference_height: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in rect]
    label_x = int(label["x"])
    label_y = int(label["y"])
    label_h = int(label["h"])
    lead_width = label_x - x1

    if not (
        1.8 * reference_height <= lead_width <= 4.0 * reference_height
    ):
        return rect

    margin = max(2, int(round(0.30 * reference_height)))
    proposed_x1 = max(x1, label_x - margin)
    if (
        proposed_x1 <= x1
        or x2 - proposed_x1 < max(4.0 * reference_height, 30.0)
    ):
        return rect

    band_y1 = max(y1, label_y)
    band_y2 = min(y2, label_y + label_h)
    if band_y2 <= band_y1:
        return rect

    leading_band = mask[band_y1:band_y2, x1:proposed_x1]
    if leading_band.size == 0:
        return rect

    if float(leading_band.mean()) > 0.06:
        return rect

    return proposed_x1, y1, x2, y2

def trim_top_edge_to_label_gap(
    rect: tuple[int, int, int, int],
    label: dict,
    mask: np.ndarray,
    reference_height: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in rect]
    label_y = int(label["y"])
    lead_height = label_y - y1

    if not (
        4.0 * reference_height <= lead_height <= 10.0 * reference_height
    ):
        return rect

    margin = max(2, int(round(0.30 * reference_height)))
    proposed_y1 = max(y1, label_y - margin)
    if (
        proposed_y1 <= y1
        or y2 - proposed_y1 < max(4.0 * reference_height, 30.0)
    ):
        return rect

    gap_y1 = max(
        y1,
        int(round(proposed_y1 - 0.75 * reference_height)),
    )
    gap_y2 = min(
        y2,
        int(round(proposed_y1 - 0.15 * reference_height)),
    )
    if gap_y2 <= gap_y1:
        return rect

    gap_band = mask[gap_y1:gap_y2, x1:x2]
    if gap_band.size == 0:
        return rect

    if float(gap_band.mean()) > 0.04:
        return rect

    return x1, proposed_y1, x2, y2

def trim_top_edge_to_label_band(
    rect: tuple[int, int, int, int],
    label: dict,
    mask: np.ndarray,
    reference_height: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(value) for value in rect]
    label_x = int(label["x"])
    label_y = int(label["y"])
    label_w = int(label["w"])
    lead_height = label_y - y1
    width = x2 - x1

    if not (
        1.25 * reference_height <= lead_height <= 2.2 * reference_height
    ):
        return rect
    if width > 12.0 * reference_height:
        return rect
    if label_x - x1 > 0.90 * reference_height:
        return rect

    margin = max(2, int(round(0.30 * reference_height)))
    proposed_y1 = max(y1, label_y - margin)
    if (
        proposed_y1 <= y1
        or y2 - proposed_y1 < max(4.0 * reference_height, 30.0)
    ):
        return rect

    label_band_x1 = max(x1, int(round(label_x - 0.25 * reference_height)))
    label_band_x2 = min(
        x2,
        int(round(label_x + label_w + 0.60 * reference_height)),
    )
    if label_band_x2 <= label_band_x1:
        return rect

    label_band = mask[y1:proposed_y1, label_band_x1:label_band_x2]
    if label_band.size == 0:
        return rect

    if float(label_band.mean()) > 0.32:
        return rect

    return x1, proposed_y1, x2, y2

def repair_right_panel_continuation(
    panels: dict[str, dict],
    panel_labels: dict[str, dict],
    cuts: list[dict],
    image_shape: tuple[int, ...],
    mask: np.ndarray,
    reference_height: float,
    alphabet: str = LETTERS,
) -> None:
    for upper_letter in list(panels):
        upper_index = alphabet.index(upper_letter)
        if upper_index < 2:
            continue

        left_top_letter = alphabet[upper_index - 2]
        lower_letter = alphabet[upper_index - 1]
        if (
            left_top_letter not in panels
            or lower_letter not in panels
        ):
            continue

        upper_panel = panels[upper_letter]
        left_top_panel = panels[left_top_letter]
        lower_panel = panels[lower_letter]
        ux1, uy1, ux2, uy2 = upper_panel["rect"]
        tx1, ty1, tx2, ty2 = left_top_panel["rect"]
        lx1, ly1, lx2, ly2 = lower_panel["rect"]

        if abs(ly1 - uy2) > 0.75 * reference_height:
            continue
        if abs(ty2 - ly1) > 0.75 * reference_height:
            continue
        if abs(tx2 - ux1) > 0.75 * reference_height:
            continue
        if abs(lx2 - ux2) > 0.75 * reference_height:
            continue
        if ux1 - lx1 < 5.0 * reference_height:
            continue
        if ux2 - ux1 < 8.0 * reference_height:
            continue
        if ly2 - ly1 < 6.0 * reference_height:
            continue
        if ly2 - ly1 > 14.0 * reference_height:
            continue

        upper_label = panel_labels[upper_letter]
        lower_label = panel_labels[lower_letter]
        upper_label_x = float(upper_label["x"] + 0.5 * upper_label["w"])
        upper_label_y = float(upper_label["y"] + 0.5 * upper_label["h"])
        lower_label_x = float(lower_label["x"] + 0.5 * lower_label["w"])

        if lower_label_x > ux1 - 2.0 * reference_height:
            continue
        if upper_label_x > ux1 + 3.0 * reference_height:
            continue
        if upper_label_y > uy1 + 2.5 * reference_height:
            continue

        right_lower_region = mask[ly1:ly2, ux1:ux2]
        left_lower_region = mask[ly1:ly2, lx1:ux1]
        if (
            right_lower_region.size == 0
            or left_lower_region.size == 0
        ):
            continue
        if float(right_lower_region.mean()) < 0.08:
            continue
        if float(left_lower_region.mean()) < 0.08:
            continue

        new_upper_rect = (ux1, uy1, ux2, ly2)
        new_lower_rect = (lx1, ly1, ux1, ly2)
        if new_lower_rect[2] - new_lower_rect[0] < 4.0 * reference_height:
            continue

        for letter, rect in (
            (upper_letter, new_upper_rect),
            (lower_letter, new_lower_rect),
        ):
            confidence, occupancy = crop_confidence(
                {"rect": rect},
                cuts,
                panel_labels[letter],
                image_shape,
                mask,
                reference_height,
            )
            panels[letter]["rect"] = rect
            panels[letter]["confidence"] = confidence
            panels[letter]["occupancy"] = occupancy

def repair_upper_right_panel_continuation(
    panels: dict[str, dict],
    panel_labels: dict[str, dict],
    cuts: list[dict],
    image_shape: tuple[int, ...],
    mask: np.ndarray,
    reference_height: float,
) -> None:
    image_height, image_width = image_shape[:2]
    del image_height

    for upper_letter in list(panels):
        upper_panel = panels[upper_letter]
        ux1, uy1, ux2, uy2 = upper_panel["rect"]
        upper_label = panel_labels[upper_letter]
        upper_label_x = float(
            upper_label["x"] + 0.5 * upper_label["w"]
        )
        upper_label_y = float(
            upper_label["y"] + 0.5 * upper_label["h"]
        )

        if ux1 < 0.45 * image_width:
            continue
        if ux2 - ux1 < 7.0 * reference_height:
            continue
        if upper_label_x < ux1 - 0.5 * reference_height:
            continue
        if upper_label_y > uy1 + 3.5 * reference_height:
            continue

        for lower_letter in list(panels):
            if lower_letter == upper_letter:
                continue

            lower_panel = panels[lower_letter]
            lx1, ly1, lx2, ly2 = lower_panel["rect"]
            lower_label = panel_labels[lower_letter]
            lower_label_x = float(
                lower_label["x"] + 0.5 * lower_label["w"]
            )

            if abs(ly1 - uy2) > 0.75 * reference_height:
                continue
            if lx1 > 0.15 * image_width:
                has_left_row_neighbor = any(
                    other_letter not in {upper_letter, lower_letter}
                    and other_panel["rect"][0] <= 0.15 * image_width
                    and abs(other_panel["rect"][2] - lx1)
                    <= 0.75 * reference_height
                    and abs(other_panel["rect"][1] - ly1)
                    <= 0.75 * reference_height
                    and abs(other_panel["rect"][3] - ly2)
                    <= 0.75 * reference_height
                    for other_letter, other_panel in panels.items()
                )
                if not has_left_row_neighbor:
                    continue
            if abs(lx2 - ux2) > 0.75 * reference_height:
                continue
            if ux1 - lx1 < 12.0 * reference_height:
                continue
            if ly2 - ly1 > 24.0 * reference_height:
                continue
            if lower_label_x > ux1 - 8.0 * reference_height:
                continue

            left_lower_region = mask[ly1:ly2, lx1:ux1]
            right_lower_region = mask[ly1:ly2, ux1:ux2]
            if (
                left_lower_region.size == 0
                or right_lower_region.size == 0
            ):
                continue
            if float(left_lower_region.mean()) < 0.08:
                continue
            if float(right_lower_region.mean()) < 0.08:
                continue

            gap_x1 = max(
                lx1,
                int(round(ux1 - 2.0 * reference_height)),
            )
            gap_x2 = min(
                ux2,
                int(round(ux1 + 0.25 * reference_height)),
            )
            if gap_x2 <= gap_x1:
                continue

            column_scores: list[float] = []
            for x in range(gap_x1, gap_x2):
                column = mask[ly1:ly2, x:x + 1]
                if column.size:
                    column_scores.append(float(column.mean()))
            if not column_scores or min(column_scores) > 0.04:
                continue

            blank_runs: list[tuple[int, int]] = []
            run_start: int | None = None
            for offset, score in enumerate(column_scores):
                if score <= 0.04 and run_start is None:
                    run_start = offset
                if run_start is not None and (
                    score > 0.04 or offset == len(column_scores) - 1
                ):
                    run_end = offset if score > 0.04 else offset + 1
                    blank_runs.append(
                        (gap_x1 + run_start, gap_x1 + run_end)
                    )
                    run_start = None
            minimum_run = max(3, int(round(0.30 * reference_height)))
            blank_runs = [
                run
                for run in blank_runs
                if run[1] - run[0] >= minimum_run
            ]
            if not blank_runs:
                continue
            blank_runs.sort(
                key=lambda run: (
                    -(run[1] - run[0]),
                    abs(0.5 * (run[0] + run[1]) - ux1),
                )
            )
            boundary_x = int(round(0.5 * sum(blank_runs[0])))

            new_upper_y1 = uy1
            stack_above_rect: tuple[str, tuple[int, int, int, int]] | None = None
            short_lead_height = int(upper_label["y"]) - uy1
            if (
                0.75 * reference_height
                <= short_lead_height
                <= 1.40 * reference_height
                and int(upper_label["x"]) - ux1 <= 1.20 * reference_height
            ):
                margin = max(2, int(round(0.30 * reference_height)))
                proposed_y1 = max(uy1, int(upper_label["y"]) - margin)
                if (
                    proposed_y1 > uy1 + 0.35 * reference_height
                    and ly2 - proposed_y1 >= max(4.0 * reference_height, 30.0)
                ):
                    for above_letter, above_panel in panels.items():
                        if above_letter in {upper_letter, lower_letter}:
                            continue
                        ax1, ay1, ax2, ay2 = above_panel["rect"]
                        if abs(int(ay2) - uy1) > 1:
                            continue
                        if (
                            abs(ax1 - ux1) > 0.75 * reference_height
                            or abs(ax2 - ux2) > 0.75 * reference_height
                        ):
                            continue
                        overlap = max(0, min(ax2, ux2) - max(ax1, ux1))
                        overlap_ratio = overlap / min(
                            max(1, ax2 - ax1),
                            max(1, ux2 - ux1),
                        )
                        if overlap_ratio < 0.80:
                            continue
                        new_upper_y1 = proposed_y1
                        stack_above_rect = (
                            above_letter,
                            (ax1, ay1, ax2, proposed_y1),
                        )
                        break

            new_upper_rect = (boundary_x, new_upper_y1, ux2, ly2)
            new_lower_rect = (lx1, ly1, boundary_x, ly2)
            if new_lower_rect[2] - new_lower_rect[0] < (
                8.0 * reference_height
            ):
                continue

            updated_rects = [
                (upper_letter, new_upper_rect),
                (lower_letter, new_lower_rect),
            ]
            for left_letter, left_panel in panels.items():
                if left_letter in {upper_letter, lower_letter}:
                    continue
                ax1, ay1, ax2, ay2 = left_panel["rect"]
                if (
                    abs(ax2 - ux1) <= 0.75 * reference_height
                    and abs(ay1 - uy1) <= 0.75 * reference_height
                    and abs(ay2 - uy2) <= 0.75 * reference_height
                ):
                    updated_rects.insert(
                        0,
                        (left_letter, (ax1, ay1, boundary_x, ay2)),
                    )
                    break
            if stack_above_rect is not None:
                updated_rects.insert(0, stack_above_rect)

            for letter, rect in updated_rects:
                confidence, occupancy = crop_confidence(
                    {"rect": rect},
                    cuts,
                    panel_labels[letter],
                    image_shape,
                    mask,
                    reference_height,
                )
                panels[letter]["rect"] = rect
                panels[letter]["confidence"] = confidence
                panels[letter]["occupancy"] = occupancy

def repair_right_grid_row_continuation(
    panels: dict[str, dict],
    panel_labels: dict[str, dict],
    cuts: list[dict],
    image_shape: tuple[int, ...],
    mask: np.ndarray,
    reference_height: float,
) -> None:
    image_height, image_width = image_shape[:2]
    del image_height

    for upper_letter in list(panels):
        upper_panel = panels[upper_letter]
        ux1, uy1, ux2, uy2 = upper_panel["rect"]
        upper_label = panel_labels[upper_letter]

        if ux1 < 0.40 * image_width:
            continue
        if ux2 - ux1 < max(30.0 * reference_height, 0.45 * image_width):
            continue
        if uy2 - uy1 < 6.0 * reference_height:
            continue
        if int(upper_label["x"]) - ux1 > 1.50 * reference_height:
            continue
        if int(upper_label["y"]) - uy1 > 1.60 * reference_height:
            continue

        for lower_letter in list(panels):
            if lower_letter == upper_letter:
                continue

            lower_panel = panels[lower_letter]
            lx1, ly1, lx2, ly2 = lower_panel["rect"]
            lower_label = panel_labels[lower_letter]
            lower_label_x = float(
                lower_label["x"] + 0.5 * lower_label["w"]
            )

            if abs(ly1 - uy2) > 1:
                continue
            if abs(lx2 - ux2) > 0.75 * reference_height:
                continue
            if ux1 - lx1 < max(8.0 * reference_height, 0.18 * image_width):
                continue
            if not (
                5.0 * reference_height
                <= ly2 - ly1
                <= 14.0 * reference_height
            ):
                continue
            if lower_label_x > ux1 - 2.0 * reference_height:
                continue
            if int(lower_label["y"]) - ly1 > 1.60 * reference_height:
                continue

            new_lower_rect = (lx1, ly1, ux1, ly2)
            if new_lower_rect[2] - new_lower_rect[0] < (
                5.0 * reference_height
            ):
                continue

            new_upper_y1 = uy1
            stack_above_rect: tuple[str, tuple[int, int, int, int]] | None = None
            upper_lead = int(upper_label["y"]) - uy1
            if (
                0.75 * reference_height
                <= upper_lead
                <= 1.40 * reference_height
                and int(upper_label["x"]) - ux1 <= 1.50 * reference_height
            ):
                margin = max(2, int(round(0.30 * reference_height)))
                proposed_y1 = max(uy1, int(upper_label["y"]) - margin)
                if (
                    proposed_y1 > uy1 + 0.35 * reference_height
                    and ly2 - proposed_y1 >= max(8.0 * reference_height, 30.0)
                ):
                    for above_letter, above_panel in panels.items():
                        if above_letter in {upper_letter, lower_letter}:
                            continue
                        ax1, ay1, ax2, ay2 = above_panel["rect"]
                        if abs(int(ay2) - uy1) > 1:
                            continue
                        if abs(ax2 - ux2) > 0.75 * reference_height:
                            continue
                        if abs(ax1 - ux1) > 1.50 * reference_height:
                            continue

                        new_upper_y1 = proposed_y1
                        stack_above_rect = (
                            above_letter,
                            (ax1, ay1, ax2, proposed_y1),
                        )
                        break

            new_upper_y2 = ly2
            below_rect: tuple[str, tuple[int, int, int, int]] | None = None
            for below_letter, below_panel in panels.items():
                if below_letter in {upper_letter, lower_letter}:
                    continue
                bx1, by1, bx2, by2 = below_panel["rect"]
                if abs(by1 - ly2) > 1:
                    continue
                if abs(bx2 - ux2) > 0.75 * reference_height:
                    continue
                if abs(bx1 - ux1) > 1.50 * reference_height:
                    continue

                below_label = panel_labels[below_letter]
                below_lead = int(below_label["y"]) - by1
                if not (
                    1.50 * reference_height
                    <= below_lead
                    <= 3.00 * reference_height
                ):
                    continue

                margin = max(2, int(round(0.30 * reference_height)))
                proposed_y1 = max(by1, int(below_label["y"]) - margin)
                if proposed_y1 <= by1 + 0.40 * reference_height:
                    continue
                if by2 - proposed_y1 < max(4.0 * reference_height, 30.0):
                    continue

                new_upper_y2 = proposed_y1
                below_rect = (
                    below_letter,
                    (bx1, proposed_y1, bx2, by2),
                )
                break

            new_upper_rect = (ux1, new_upper_y1, ux2, new_upper_y2)
            updated_rects = [
                (upper_letter, new_upper_rect),
                (lower_letter, new_lower_rect),
            ]
            if stack_above_rect is not None:
                updated_rects.insert(0, stack_above_rect)
            if below_rect is not None:
                updated_rects.append(below_rect)

            for letter, rect in updated_rects:
                confidence, occupancy = crop_confidence(
                    {"rect": rect},
                    cuts,
                    panel_labels[letter],
                    image_shape,
                    mask,
                    reference_height,
                )
                panels[letter]["rect"] = rect
                panels[letter]["confidence"] = confidence
                panels[letter]["occupancy"] = occupancy
            break

def repair_right_next_panel_continuation(
    panels: dict[str, dict],
    panel_labels: dict[str, dict],
    cuts: list[dict],
    image_shape: tuple[int, ...],
    mask: np.ndarray,
    reference_height: float,
    alphabet: str = LETTERS,
) -> None:
    image_height, image_width = image_shape[:2]
    del image_height

    for upper_letter in list(panels):
        upper_index = alphabet.index(upper_letter)
        if upper_index + 1 >= len(alphabet):
            continue
        lower_letter = alphabet[upper_index + 1]
        if lower_letter not in panels:
            continue

        upper_panel = panels[upper_letter]
        lower_panel = panels[lower_letter]
        ux1, uy1, ux2, uy2 = upper_panel["rect"]
        lx1, ly1, lx2, ly2 = lower_panel["rect"]
        upper_label = panel_labels[upper_letter]
        lower_label = panel_labels[lower_letter]
        upper_label_x = float(
            upper_label["x"] + 0.5 * upper_label["w"]
        )
        upper_lead = int(upper_label["y"]) - uy1
        lower_label_x = float(
            lower_label["x"] + 0.5 * lower_label["w"]
        )

        if ux1 < 0.45 * image_width:
            continue
        if ux2 - ux1 < max(12.0 * reference_height, 0.30 * image_width):
            continue
        if abs(ly1 - uy2) > 1:
            continue
        if lx1 > 0.15 * image_width:
            continue
        if abs(lx2 - ux2) > 0.75 * reference_height:
            continue
        if ux1 - lx1 < max(8.0 * reference_height, 0.25 * image_width):
            continue
        if ly2 - ly1 < 12.0 * reference_height:
            continue
        if upper_label_x > ux1 + 2.0 * reference_height:
            continue
        if lower_label_x > ux1 - 8.0 * reference_height:
            continue
        if int(lower_label["y"]) - ly1 > 1.60 * reference_height:
            continue
        if not (
            2.0 * reference_height
            <= upper_lead
            <= 4.50 * reference_height
        ):
            continue

        left_lower_region = mask[ly1:ly2, lx1:ux1]
        right_lower_region = mask[ly1:ly2, ux1:ux2]
        if (
            left_lower_region.size == 0
            or right_lower_region.size == 0
        ):
            continue
        if float(left_lower_region.mean()) < 0.08:
            continue
        if float(right_lower_region.mean()) < 0.08:
            continue

        junction_y1 = max(uy1, ly1 - int(round(reference_height)))
        junction_y2 = min(ly2, ly1 + int(round(reference_height)) + 1)
        right_junction = mask[junction_y1:junction_y2, ux1:ux2]
        if right_junction.size:
            row_scores = right_junction.mean(axis=1)
            longest_blank_run = 0
            current_blank_run = 0
            for score in row_scores:
                if float(score) <= 0.04:
                    current_blank_run += 1
                    longest_blank_run = max(
                        longest_blank_run,
                        current_blank_run,
                    )
                else:
                    current_blank_run = 0
            minimum_blank_run = max(
                2,
                int(round(0.15 * reference_height)),
            )
            if longest_blank_run >= minimum_blank_run:
                continue

        gap_x1 = max(
            lx1,
            int(round(ux1 - 0.25 * reference_height)),
        )
        gap_x2 = min(
            ux2,
            int(round(ux1 + 1.25 * reference_height)),
        )
        if gap_x2 <= gap_x1:
            continue

        column_scores: list[float] = []
        for x in range(gap_x1, gap_x2):
            column = mask[ly1:ly2, x:x + 1]
            if column.size:
                column_scores.append(float(column.mean()))
        if not column_scores or min(column_scores) > 0.04:
            continue

        new_upper_y1 = uy1
        stack_above_rect: tuple[str, tuple[int, int, int, int]] | None = None
        if 2.0 * reference_height <= upper_lead <= 4.50 * reference_height:
            margin = max(2, int(round(0.30 * reference_height)))
            proposed_y1 = max(uy1, int(upper_label["y"]) - margin)
            if (
                proposed_y1 > uy1 + 0.75 * reference_height
                and ly2 - proposed_y1 >= max(8.0 * reference_height, 30.0)
            ):
                for above_letter, above_panel in panels.items():
                    if above_letter in {upper_letter, lower_letter}:
                        continue
                    ax1, ay1, ax2, ay2 = above_panel["rect"]
                    if abs(int(ay2) - uy1) > 1:
                        continue
                    if abs(ax2 - ux2) > 0.75 * reference_height:
                        continue
                    if abs(ax1 - ux1) > 1.50 * reference_height:
                        continue

                    extension_region = mask[
                        uy1:proposed_y1,
                        max(ax1, ux1):min(ax2, ux2),
                    ]
                    if (
                        extension_region.size == 0
                        or float(extension_region.mean()) < 0.08
                    ):
                        continue

                    new_upper_y1 = proposed_y1
                    stack_above_rect = (
                        above_letter,
                        (ax1, ay1, ax2, proposed_y1),
                    )
                    break

        new_upper_rect = (ux1, new_upper_y1, ux2, ly2)
        new_lower_rect = (lx1, ly1, ux1, ly2)
        if new_lower_rect[2] - new_lower_rect[0] < (
            8.0 * reference_height
        ):
            continue

        updated_rects = [
            (upper_letter, new_upper_rect),
            (lower_letter, new_lower_rect),
        ]
        if stack_above_rect is not None:
            updated_rects.insert(0, stack_above_rect)

        for letter, rect in updated_rects:
            confidence, occupancy = crop_confidence(
                {"rect": rect},
                cuts,
                panel_labels[letter],
                image_shape,
                mask,
                reference_height,
            )
            panels[letter]["rect"] = rect
            panels[letter]["confidence"] = confidence
            panels[letter]["occupancy"] = occupancy

def repair_lower_row_span_under_right_panel(
    panels: dict[str, dict],
    panel_labels: dict[str, dict],
    cuts: list[dict],
    image_shape: tuple[int, ...],
    mask: np.ndarray,
    reference_height: float,
    alphabet: str = LETTERS,
) -> None:
    for upper_right_letter in list(panels):
        try:
            upper_right_index = alphabet.index(upper_right_letter)
        except ValueError:
            continue
        if upper_right_index < 1 or upper_right_index + 1 >= len(alphabet):
            continue

        upper_left_letter = alphabet[upper_right_index - 1]
        lower_left_letter = alphabet[upper_right_index + 1]
        if (
            upper_left_letter not in panels
            or lower_left_letter not in panels
        ):
            continue

        upper_left = panels[upper_left_letter]
        upper_right = panels[upper_right_letter]
        lower_left = panels[lower_left_letter]
        lx1, ly1, lx2, ly2 = upper_left["rect"]
        rx1, ry1, rx2, ry2 = upper_right["rect"]
        cx1, cy1, cx2, cy2 = lower_left["rect"]

        if abs(ly1 - ry1) > 0.75 * reference_height:
            continue
        if abs(lx2 - rx1) > 0.75 * reference_height:
            continue
        if abs(cx1 - lx1) > 0.75 * reference_height:
            continue
        if abs(cx2 - rx1) > 0.75 * reference_height:
            continue
        if abs(cy1 - ly2) > 0.75 * reference_height:
            continue
        if abs(cy2 - ry2) > 0.75 * reference_height:
            continue
        if ry2 - ry1 < (cy1 - ry1) + 5.0 * reference_height:
            continue
        if rx2 - rx1 < 5.0 * reference_height:
            continue
        if cy2 - cy1 < 5.0 * reference_height:
            continue

        upper_right_label = panel_labels[upper_right_letter]
        lower_left_label = panel_labels[lower_left_letter]
        if int(upper_right_label["y"]) > ry1 + 2.0 * reference_height:
            continue
        if int(lower_left_label["y"]) > cy1 + 2.0 * reference_height:
            continue

        band = max(2, int(round(0.25 * reference_height)))
        gap_y1 = max(ry1, cy1)
        gap_y2 = min(
            ry2,
            cy1 + max(2 * band + 1, int(round(1.25 * reference_height))),
        )
        gap_region = mask[gap_y1:gap_y2, rx1:rx2]
        if gap_region.size == 0:
            continue
        row_scores = gap_region.mean(axis=1)
        if (
            float(row_scores.min()) > 0.06
            and float(np.mean(row_scores < 0.04)) < 0.20
        ):
            continue

        lower_right_region = mask[
            min(cy2, gap_y1 + band):cy2,
            rx1:rx2,
        ]
        if (
            lower_right_region.size == 0
            or float(lower_right_region.mean()) < 0.04
        ):
            continue

        new_upper_right_rect = (rx1, ry1, rx2, cy1)
        new_lower_left_rect = (cx1, cy1, rx2, cy2)
        if new_upper_right_rect[3] - new_upper_right_rect[1] < (
            5.0 * reference_height
        ):
            continue

        for letter, rect in (
            (upper_right_letter, new_upper_right_rect),
            (lower_left_letter, new_lower_left_rect),
        ):
            confidence, occupancy = crop_confidence(
                {"rect": rect},
                cuts,
                panel_labels[letter],
                image_shape,
                mask,
                reference_height,
            )
            panels[letter]["rect"] = rect
            panels[letter]["confidence"] = confidence
            panels[letter]["occupancy"] = occupancy

def rollback_conflicting_panel_repairs(
    panels: dict[str, dict],
    baseline_panels: dict[str, dict],
    panel_labels: dict[str, dict],
    reference_height: float,
) -> list[str]:
    """Undo repair-stage rectangles that cross another top-level panel."""
    changed = {
        letter
        for letter, panel in panels.items()
        if letter in baseline_panels
        and tuple(panel["rect"]) != tuple(baseline_panels[letter]["rect"])
    }
    invalid: set[str] = set()

    for letter in changed:
        x1, y1, x2, y2 = panels[letter]["rect"]
        margin = max(1.0, 0.10 * reference_height)
        for other_letter, other_label in panel_labels.items():
            if other_letter == letter:
                continue
            center_x = float(other_label["x"]) + 0.5 * float(other_label["w"])
            center_y = float(other_label["y"]) + 0.5 * float(other_label["h"])
            if (
                x1 + margin < center_x < x2 - margin
                and y1 + margin < center_y < y2 - margin
            ):
                invalid.add(letter)
                break

    letters = sorted(panels)
    for index, letter in enumerate(letters):
        ax1, ay1, ax2, ay2 = panels[letter]["rect"]
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        for other_letter in letters[index + 1 :]:
            bx1, by1, bx2, by2 = panels[other_letter]["rect"]
            overlap_width = max(0, min(ax2, bx2) - max(ax1, bx1))
            overlap_height = max(0, min(ay2, by2) - max(ay1, by1))
            overlap_area = overlap_width * overlap_height
            if overlap_area <= 0:
                continue
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            if (
                overlap_area < reference_height * reference_height
                and overlap_area / min(area_a, area_b) < 0.01
            ):
                continue
            if letter in changed:
                invalid.add(letter)
            if other_letter in changed:
                invalid.add(other_letter)

    for letter in invalid:
        panels[letter] = dict(baseline_panels[letter])
    return sorted(invalid)
