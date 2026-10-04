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

"""Panel crop, preview, layout.json, input collection, and overview output."""

from .detection import *  # noqa: F401,F403
from .layout import *  # noqa: F401,F403

# This flag belonged to the original monolithic module. It is output-layer
# state, so define it here explicitly after the module split.
_OCR_UNAVAILABLE_REPORTED = False

def process_image(
    path: Path,
    output_dir: Path,
    folder_name: str | None = None,
    ocr_filter: bool = False,
    ocr_engine: str = "auto",
    label_mode: str = "auto",
) -> tuple[Path, dict]:
    global _OCR_UNAVAILABLE_REPORTED

    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(
            f"OpenCV 无法读取图像：{path}"
        )

    image_height, image_width = image.shape[:2]
    constraints = load_figure_constraints(path)
    expected_labels = list(constraints["expected_labels"])
    native_labels = list(constraints["native_labels"])
    if ocr_filter:
        ocr_words, ocr_info = run_ocr_words(image, ocr_engine)
        if (
            ocr_info.get("status") != "ok"
            and not _OCR_UNAVAILABLE_REPORTED
        ):
            print(
                "[OCR] 未找到可用 OCR 引擎，已退回 OpenCV 规则："
                + "; ".join(ocr_info.get("errors", []))
            )
            _OCR_UNAVAILABLE_REPORTED = True
    else:
        ocr_words = []
        ocr_info = {
            "enabled": False,
            "engine": None,
            "word_count": 0,
            "status": "disabled",
            "errors": [],
        }

    requested_label_mode = label_mode
    detection = detect_panel_labels(image, ocr_words, requested_label_mode)
    expected_labels_for_detection = align_caption_labels_to_detection_case(
        expected_labels,
        detection,
    )
    mask = make_content_mask(image)
    if (
        expected_labels_for_detection
        and len(detection[0]) > len(expected_labels_for_detection)
    ):
        alphabet = str(detection[7].get("alphabet", ""))
        visual_extension_evidence = assess_detection_layout_evidence(
            image,
            mask,
            detection,
            alphabet,
        )
        (
            detection,
            expected_labels_for_detection,
            _,
        ) = promote_strong_visual_caption_extension(
            detection,
            expected_labels_for_detection,
            visual_extension_evidence,
        )
    # Let the layout-aware retry inspect the unpruned visual result first.
    # A simple sequence retry can otherwise fill missing letters with glyphs
    # from internal prose and then make the layout retry return early because
    # every expected character is technically present.
    detection, layout_retry = apply_caption_layout_retry(
        image,
        ocr_words,
        detection,
        expected_labels_for_detection,
        content_mask=mask,
    )
    detection, sequence_retry = apply_caption_sequence_retry(
        detection,
        expected_labels_for_detection,
        image.shape,
    )
    # Remove visual letters beyond the effective sequence before geometry
    # repairs.  A longer sequence with strong whole-layout evidence was already
    # promoted above, so this now targets chart/legend false positives only.
    detection, early_caption_pruned_labels = apply_caption_sequence_upper_bound(
        detection,
        expected_labels_for_detection,
    )
    candidate_repairs: list[dict[str, object]] = []
    candidate_labels, grid_candidate_repairs = (
        repair_caption_regular_grid_from_candidates(
            detection[0],
            detection[2],
            expected_labels_for_detection,
            float(detection[3]),
            image.shape,
        )
    )
    candidate_repairs.extend(grid_candidate_repairs)
    candidate_labels, row_start_repairs = repair_caption_row_start_outlier(
        candidate_labels,
        detection[2],
        expected_labels_for_detection,
        float(detection[3]),
        image.shape,
    )
    candidate_repairs.extend(row_start_repairs)
    candidate_labels, side_stack_repairs = (
        repair_caption_stacked_side_leader_outlier(
            candidate_labels,
            expected_labels_for_detection,
            float(detection[3]),
            image.shape,
        )
    )
    candidate_repairs.extend(side_stack_repairs)
    candidate_labels, geometry_grid_repairs = (
        repair_caption_regular_grid_from_geometry(
            candidate_labels,
            expected_labels_for_detection,
            float(detection[3]),
            image.shape,
        )
    )
    candidate_repairs.extend(geometry_grid_repairs)
    if candidate_repairs:
        candidate_label_info = dict(detection[7])
        candidate_label_info["caption_candidate_repairs"] = candidate_repairs
        detection = (
            candidate_labels,
            detection[1],
            detection[2],
            detection[3],
            detection[4],
            detection[5],
            detection[6],
            candidate_label_info,
        )
    grid_labels, grid_repairs = repair_regular_grid_label_outlier(
        detection[0],
        expected_labels_for_detection,
        float(detection[3]),
        image.shape,
    )
    if grid_repairs:
        grid_label_info = dict(detection[7])
        grid_label_info["caption_grid_repairs"] = grid_repairs
        detection = (
            grid_labels,
            detection[1],
            detection[2],
            detection[3],
            detection[4],
            detection[5],
            detection[6],
            grid_label_info,
        )
    detection, expected_labels_for_detection, nested_terminal_pruned = (
        prune_nested_caption_terminal_grid_label(
            detection,
            expected_labels_for_detection,
        )
    )
    detection, caption_pruned_labels = apply_caption_sequence_upper_bound(
        detection,
        expected_labels_for_detection,
    )
    if expected_labels:
        native_match_count = apply_native_label_evidence(
            detection[0],
            native_labels,
            expected_labels_for_detection,
            float(detection[3]),
        )
        retry_strategy = (
            "caption_layout_retry"
            if layout_retry
            else "caption_sequence_retry"
            if sequence_retry
            else "caption_validated_cv"
        )
        detection[7].update(
            strategy=(
                f"{retry_strategy}+pdf_text"
                if native_match_count
                else retry_strategy
            ),
            expected_labels=expected_labels_for_detection,
            caption_expected_labels=expected_labels,
            native_match_count=native_match_count,
            caption_pruned_labels=(
                early_caption_pruned_labels
                + nested_terminal_pruned
                + caption_pruned_labels
            ),
        )

    (
        labels,
        assignment_rows,
        candidates,
        reference_height,
        sequence_length,
        presence,
        initial,
        label_info,
    ) = detection
    alphabet = str(label_info["alphabet"])

    leaves, cuts = recursive_xycut(
        mask,
        (0, 0, image_width, image_height),
        labels,
        reference_height,
    )
    panels: dict[str, dict] = {}
    panel_labels: dict[str, dict] = {}
    left_edge_adjustments: list[dict] = []
    top_edge_adjustments: list[dict] = []
    diagnostics: list[dict] = []

    for leaf in leaves:
        if len(leaf["labels"]) == 1:
            label = leaf["labels"][0]
            rect = trim_left_edge_to_label_band(
                leaf["rect"],
                label,
                mask,
                reference_height,
            )
            if rect[0] != leaf["rect"][0]:
                left_edge_adjustments.append(
                    {
                        "letter": label["letter"],
                        "old_x": int(leaf["rect"][0]),
                        "new_x": int(rect[0]),
                        "y1": int(rect[1]),
                        "y2": int(rect[3]),
                    }
                )
            before_top_trim = rect
            rect = trim_top_edge_to_label_gap(
                rect,
                label,
                mask,
                reference_height,
            )
            rect = trim_top_edge_to_label_band(
                rect,
                label,
                mask,
                reference_height,
            )
            if rect[1] != before_top_trim[1]:
                top_edge_adjustments.append(
                    {
                        "letter": label["letter"],
                        "old_y": int(before_top_trim[1]),
                        "new_y": int(rect[1]),
                        "x1": int(rect[0]),
                        "x2": int(rect[2]),
                    }
                )
            trimmed_leaf = dict(leaf)
            trimmed_leaf["rect"] = rect
            confidence, occupancy = crop_confidence(
                trimmed_leaf,
                cuts,
                label,
                image.shape,
                mask,
                reference_height,
            )
            panels[label["letter"]] = {
                "rect": rect,
                "confidence": confidence,
                "occupancy": occupancy,
                "label_quality": float(label["quality"]),
            }
            panel_labels[label["letter"]] = label
        else:
            diagnostics.append(
                {
                    "rect": leaf["rect"],
                    "labels": [
                        label["letter"]
                        for label in leaf["labels"]
                    ],
                    "unresolved": True,
                }
            )

    for adjustment in left_edge_adjustments:
        old_x = int(adjustment["old_x"])
        new_x = int(adjustment["new_x"])
        adjusted_y1 = int(adjustment["y1"])
        adjusted_y2 = int(adjustment["y2"])
        adjusted_height = max(1, adjusted_y2 - adjusted_y1)
        neighbor_letters: list[str] = []

        for letter, panel in panels.items():
            if letter == adjustment["letter"]:
                continue

            x1, y1, x2, y2 = panel["rect"]
            if abs(int(x2) - old_x) > 1:
                continue
            if not (x1 < old_x < new_x < x2 + 6.0 * reference_height):
                continue

            overlap = max(
                0,
                min(y2, adjusted_y2) - max(y1, adjusted_y1),
            )
            overlap_ratio = overlap / min(
                max(1, y2 - y1),
                adjusted_height,
            )
            if overlap_ratio < 0.70:
                continue
            neighbor_letters.append(letter)

        if not neighbor_letters:
            continue

        for neighbor_letter in neighbor_letters:
            neighbor_panel = panels[neighbor_letter]
            x1, y1, _, y2 = neighbor_panel["rect"]
            neighbor_rect = (x1, y1, new_x, y2)
            neighbor_leaf = {"rect": neighbor_rect}
            confidence, occupancy = crop_confidence(
                neighbor_leaf,
                cuts,
                panel_labels[neighbor_letter],
                image.shape,
                mask,
                reference_height,
            )
            neighbor_panel["rect"] = neighbor_rect
            neighbor_panel["confidence"] = confidence
            neighbor_panel["occupancy"] = occupancy

    for adjustment in top_edge_adjustments:
        old_y = int(adjustment["old_y"])
        new_y = int(adjustment["new_y"])
        adjusted_x1 = int(adjustment["x1"])
        adjusted_x2 = int(adjustment["x2"])
        adjusted_width = max(1, adjusted_x2 - adjusted_x1)
        best_neighbor: tuple[float, str] | None = None

        for letter, panel in panels.items():
            if letter == adjustment["letter"]:
                continue

            x1, y1, x2, y2 = panel["rect"]
            if abs(int(y2) - old_y) > 1:
                continue
            if not (y1 < old_y < new_y < y2 + 10.0 * reference_height):
                continue

            overlap = max(
                0,
                min(x2, adjusted_x2) - max(x1, adjusted_x1),
            )
            overlap_ratio = overlap / min(
                max(1, x2 - x1),
                adjusted_width,
            )
            if overlap_ratio < 0.70:
                continue
            shared_width_ratio = overlap / max(
                max(1, x2 - x1),
                adjusted_width,
            )
            if shared_width_ratio < 0.55:
                continue
            score = overlap_ratio * (x2 - x1)
            if best_neighbor is None or score > best_neighbor[0]:
                best_neighbor = (score, letter)

        if best_neighbor is None:
            continue

        neighbor_letter = best_neighbor[1]
        neighbor_panel = panels[neighbor_letter]
        x1, y1, x2, _ = neighbor_panel["rect"]
        neighbor_rect = (x1, y1, x2, new_y)
        neighbor_leaf = {"rect": neighbor_rect}
        confidence, occupancy = crop_confidence(
            neighbor_leaf,
            cuts,
            panel_labels[neighbor_letter],
            image.shape,
            mask,
            reference_height,
        )
        neighbor_panel["rect"] = neighbor_rect
        neighbor_panel["confidence"] = confidence
        neighbor_panel["occupancy"] = occupancy

    panels_before_repairs = {
        letter: dict(panel)
        for letter, panel in panels.items()
    }
    repair_right_panel_continuation(
        panels,
        panel_labels,
        cuts,
        image.shape,
        mask,
        reference_height,
        alphabet,
    )
    repair_upper_right_panel_continuation(
        panels,
        panel_labels,
        cuts,
        image.shape,
        mask,
        reference_height,
    )
    repair_right_grid_row_continuation(
        panels,
        panel_labels,
        cuts,
        image.shape,
        mask,
        reference_height,
    )
    repair_right_next_panel_continuation(
        panels,
        panel_labels,
        cuts,
        image.shape,
        mask,
        reference_height,
        alphabet,
    )
    repair_lower_row_span_under_right_panel(
        panels,
        panel_labels,
        cuts,
        image.shape,
        mask,
        reference_height,
        alphabet,
    )
    repair_rollbacks = rollback_conflicting_panel_repairs(
        panels,
        panels_before_repairs,
        panel_labels,
        reference_height,
    )

    proposed_panels = {
        letter: dict(panel)
        for letter, panel in panels.items()
    }
    split_decision = assess_composite_figure(
        panels,
        panel_labels,
        alphabet,
        cuts,
        image.shape,
        expected_labels_for_detection,
    )
    proposed_cuts = [dict(cut) for cut in cuts]
    if not split_decision["is_composite"]:
        panels = {}
        cuts = []

    output_dir.mkdir(parents=True, exist_ok=True)
    subfolder = output_dir / (
        folder_name if folder_name is not None else path.stem
    )
    if subfolder.exists():
        shutil.rmtree(subfolder)
    subfolder.mkdir(parents=True)

    preview_image = image.copy()
    colors = [
        (0, 0, 255),
        (255, 0, 0),
        (0, 140, 255),
        (180, 0, 180),
        (0, 150, 0),
    ]
    line_width = max(
        1,
        round(min(image_height, image_width) / 500),
    )

    for index, (letter, panel) in enumerate(
        sorted(panels.items())
    ):
        x1, y1, x2, y2 = panel["rect"]
        color = colors[index % len(colors)]
        cv2.rectangle(
            preview_image,
            (x1, y1),
            (x2 - 1, y2 - 1),
            color,
            line_width,
        )
        caption = f"{letter} {panel['confidence']:.2f}"
        cv2.putText(
            preview_image,
            caption,
            (x1 + 4, min(y2 - 4, y1 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(
            str(subfolder / f"panel_{letter}.png"),
            crop,
        )

    for cut in cuts:
        x1, y1, x2, y2 = cut["rect"]
        color = (0, 200, 200)
        if cut["ori"] == "v":
            cv2.line(
                preview_image,
                (cut["pos"], y1),
                (cut["pos"], y2),
                color,
                1,
            )
        else:
            cv2.line(
                preview_image,
                (x1, cut["pos"]),
                (x2, cut["pos"]),
                color,
                1,
            )

    preview_path = subfolder / "preview.png"
    cv2.imwrite(str(preview_path), preview_image)

    metadata = {
        "source": str(path),
        "image_size": [image_width, image_height],
        "reference_label_height": reference_height,
        "sequence_length": sequence_length,
        "label_mode": label_info,
        "detected_labels": {
            label["letter"]: {
                "bbox": [
                    label["x"],
                    label["y"],
                    label["w"],
                    label["h"],
                ],
                "quality": float(label["quality"]),
                "ocr_penalty": float(label.get("ocr_penalty", 0.0)),
                "polarity": label.get("polarity", "dark"),
                "source": label.get("source", "cv"),
                "native_anchor_distance": label.get("native_anchor_distance"),
            }
            for label in labels
        },
        "panels": panels,
        "proposed_panels": proposed_panels,
        "split_decision": split_decision,
        "figure_constraints": {
            "expected_labels": expected_labels,
            "effective_expected_labels": expected_labels_for_detection,
            "native_label_count": len(native_labels),
            "detection_strategy": label_info.get("strategy", "baseline"),
        },
        "cuts": cuts,
        "proposed_cuts": proposed_cuts,
        "diagnostics": diagnostics,
        "repair_rollbacks": repair_rollbacks,
        "ocr": ocr_info,
        "notes": {
            "confidence": (
                "内部启发式复核指标，不是校准概率；"
                "低于 0.70 建议人工查看 preview。"
            ),
            "scope": (
                "输出顶层 A-Z 面板，不继续拆分面板内部"
                "未标字母的子图。"
            ),
        },
    }
    (subfolder / "layout.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return preview_path, metadata

def collect_inputs(
    items: list[str],
    recursive: bool,
) -> list[Path]:
    result: list[Path] = []
    seen: set[str] = set()

    for raw in items:
        path = Path(raw)
        candidates: list[Path] = []

        if (
            path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        ):
            candidates = [path]
        elif path.is_dir():
            iterator = (
                path.rglob("*")
                if recursive else path.glob("*")
            )
            candidates = sorted(
                [
                    item
                    for item in iterator
                    if item.is_file()
                    and item.suffix.lower() in IMAGE_EXTENSIONS
                ],
                key=lambda item: str(item).lower(),
            )
        else:
            print(
                f"[跳过] 不存在或不是支持的图像：{path}"
            )

        for candidate in candidates:
            key = str(candidate.resolve())
            if key not in seen:
                seen.add(key)
                result.append(candidate)

    return result

def build_overview(
    records: list[dict],
    output_path: Path,
) -> None:
    if not records:
        return

    columns = 3
    tile_width, tile_height = 600, 620
    rows = (len(records) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * tile_height),
        "white",
    )

    font = None
    for font_path in FONT_PATHS:
        try:
            font = ImageFont.truetype(font_path, 24)
            break
        except OSError:
            continue
    if font is None:
        font = ImageFont.load_default()

    for index, record in enumerate(records):
        preview = Image.open(
            record["preview"]
        ).convert("RGB")
        preview.thumbnail(
            (tile_width - 14, tile_height - 62)
        )

        tile = Image.new(
            "RGB",
            (tile_width, tile_height),
            "white",
        )
        x = (tile_width - preview.width) // 2
        y = 53 + (
            tile_height - 53 - preview.height
        ) // 2
        tile.paste(preview, (x, y))

        draw = ImageDraw.Draw(tile)
        title = (
            f"{index + 1:02d}  {record['labels']}  "
            f"mean={record['mean_confidence']:.2f}"
        )
        draw.text((9, 9), title, fill="black", font=font)

        canvas.paste(
            tile,
            (
                (index % columns) * tile_width,
                (index // columns) * tile_height,
            ),
        )

    canvas.save(output_path)
