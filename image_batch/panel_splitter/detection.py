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

"""A-Z/a-z panel-label detection, OCR filtering, and sequence arbitration."""

UPPER_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWER_LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTERS = UPPER_LETTERS
NORMALIZED_SIZE = 48
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"
}

def discover_fonts() -> list[str]:
    fixed = [
        # Windows
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibri.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
    ]

    result: list[str] = []
    for item in fixed:
        if Path(item).is_file() and item not in result:
            result.append(item)

    if result:
        return result[:14]

    roots = [
        Path("/usr/share/fonts"),
        Path.home() / ".fonts",
        Path("C:/Windows/Fonts"),
        Path("/System/Library/Fonts"),
        Path("/Library/Fonts"),
    ]
    keywords = (
        "sans", "arial", "helvetica", "arimo",
        "liberation", "dejavu", "noto", "lato"
    )

    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for path in root.rglob(pattern):
                if any(k in path.name.lower() for k in keywords):
                    value = str(path)
                    if value not in result:
                        result.append(value)
                if len(result) >= 14:
                    break
            if len(result) >= 14:
                break
        if len(result) >= 14:
            break

    if not result:
        raise RuntimeError(
            "未找到无衬线字体。请在 discover_fonts() 中加入 "
            "Arial、Helvetica、Noto Sans 等本机字体路径。"
        )
    return result


FONT_PATHS = discover_fonts()
_TEMPLATE_ARRAY: np.ndarray | None = None
_TEMPLATE_SUMS: np.ndarray | None = None
_TEMPLATE_NAMES: list[str] | None = None
_TEMPLATE_INDEX: dict[str, np.ndarray] | None = None
_OCR_UNAVAILABLE_REPORTED = False

def normalize_glyph(
    mask: np.ndarray,
    size: int = NORMALIZED_SIZE,
) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return np.zeros((size, size), np.uint8)

    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    height, width = crop.shape
    scale = min(
        (size - 6) / max(width, 1),
        (size - 6) / max(height, 1),
    )
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(
        crop,
        (new_width, new_height),
        interpolation=cv2.INTER_NEAREST,
    )

    output = np.zeros((size, size), np.uint8)
    x = (size - new_width) // 2
    y = (size - new_height) // 2
    output[y:y + new_height, x:x + new_width] = resized
    return (output > 0).astype(np.uint8)

def get_template_bank() -> tuple[
    np.ndarray,
    np.ndarray,
    list[str],
    dict[str, np.ndarray],
]:
    global _TEMPLATE_ARRAY
    global _TEMPLATE_SUMS
    global _TEMPLATE_NAMES
    global _TEMPLATE_INDEX

    if _TEMPLATE_ARRAY is not None:
        return (
            _TEMPLATE_ARRAY,
            _TEMPLATE_SUMS,
            _TEMPLATE_NAMES,
            _TEMPLATE_INDEX,
        )

    templates: list[np.ndarray] = []
    names: list[str] = []

    for font_path in FONT_PATHS:
        font = ImageFont.truetype(font_path, 64)
        for letter in UPPER_LETTERS + LOWER_LETTERS:
            image = Image.new("L", (100, 100), 255)
            draw = ImageDraw.Draw(image)
            bbox = draw.textbbox((0, 0), letter, font=font)
            draw.text(
                (10 - bbox[0], 10 - bbox[1]),
                letter,
                font=font,
                fill=0,
            )
            glyph = normalize_glyph(
                (np.array(image) < 128).astype(np.uint8) * 255
            )
            templates.append(glyph.ravel().astype(np.float32))
            names.append(letter)

    array = np.stack(templates)
    sums = array.sum(axis=1)
    index = {
        letter: np.array(
            [i for i, name in enumerate(names) if name == letter]
        )
        for letter in UPPER_LETTERS + LOWER_LETTERS
    }

    _TEMPLATE_ARRAY = array
    _TEMPLATE_SUMS = sums
    _TEMPLATE_NAMES = names
    _TEMPLATE_INDEX = index
    return array, sums, names, index

def classify_all_letters(
    mask: np.ndarray,
    alphabet: str = LETTERS,
) -> np.ndarray:
    templates, sums, _, index = get_template_bank()
    glyph = normalize_glyph(mask).ravel().astype(np.float32)
    glyph_sum = glyph.sum()
    dice = 2 * (templates @ glyph) / (sums + glyph_sum + 1e-6)
    return np.array(
        [dice[index[letter]].max() for letter in alphabet],
        np.float32,
    )

def gap_to_dark(
    dark: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    right: bool,
    max_distance: int,
) -> int:
    image_height, image_width = dark.shape
    y1 = max(0, y - int(0.12 * height))
    y2 = min(image_height, y + height + int(0.12 * height))

    if right:
        start = x + width
        end = min(image_width, start + max_distance)
        columns = (
            dark[y1:y2, start:end].any(axis=0)
            if end > start else np.array([], bool)
        )
    else:
        end = x
        start = max(0, end - max_distance)
        columns = (
            dark[y1:y2, start:end].any(axis=0)[::-1]
            if end > start else np.array([], bool)
        )

    indices = np.flatnonzero(columns)
    return int(indices[0]) if len(indices) else max_distance

def extract_label_candidates(
    image: np.ndarray,
    alphabet: str = LETTERS,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    image_height, image_width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    dark = (gray < 135).astype(np.uint8)
    bright = (gray > 205).astype(np.uint8)

    candidates: list[dict] = []

    def scan_foreground(
        foreground: np.ndarray,
        polarity: str,
    ) -> None:
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            foreground,
            connectivity=8,
        )

        for component_id in range(1, count):
            x, y, width, height, area = map(
                int,
                stats[component_id],
            )

            if not (
                0.0085 * image_height <= height <= 0.055 * image_height
                and 1 <= width <= min(0.055 * image_width, 1.75 * height)
            ):
                continue
            if area < max(7, 0.035 * width * height):
                continue
            if width > 1.35 * height and area / (width * height) < 0.25:
                continue

            padding = max(4, int(0.55 * height))
            x1 = max(0, x - padding)
            x2 = min(image_width, x + width + padding)
            y1 = max(0, y - padding)
            y2 = min(image_height, y + height + padding)

            roi = gray[y1:y2, x1:x2]
            roi_foreground = foreground[y1:y2, x1:x2]
            ring = np.ones(roi.shape, bool)
            ring[
                y - y1:y - y1 + height,
                x - x1:x - x1 + width,
            ] = False
            foreground_ring = (
                float(roi_foreground[ring].mean()) if ring.any() else 1.0
            )
            if foreground_ring > 0.36:
                continue
            if polarity == "light" and ring.any():
                ring_pixels = roi[ring]
                dark_background = float((ring_pixels < 170).mean())
                if dark_background < 0.20:
                    continue

            component = (
                labels[y:y + height, x:x + width] == component_id
            ).astype(np.uint8) * 255
            similarities = classify_all_letters(component, alphabet)
            top_index = int(similarities.argmax())
            top_similarity = float(similarities[top_index])
            second_similarity = float(np.partition(similarities, -2)[-2])

            if top_similarity < 0.48:
                continue

            max_distance = max(8, int(2.2 * height))
            right_gap = gap_to_dark(
                foreground, x, y, width, height, True, max_distance
            )
            left_gap = gap_to_dark(
                foreground, x, y, width, height, False, max_distance
            )
            isolation = min(
                1.0,
                max(right_gap, left_gap) / (0.9 * height + 1e-6),
            )

            left_start = max(0, x - int(1.2 * height))
            top_start = max(0, y - int(1.0 * height))

            left_density = (
                float(
                    foreground[
                        max(0, y - int(0.25 * height)):
                        min(image_height, y + int(1.25 * height)),
                        left_start:x,
                    ].mean()
                )
                if x > left_start else 0.0
            )
            top_density = (
                float(
                    foreground[
                        top_start:y,
                        max(0, x - int(0.25 * height)):
                        min(image_width, x + width + int(1.25 * height)),
                    ].mean()
                )
                if y > top_start else 0.0
            )
            gutter = 1.0 - min(left_density, top_density)

            base_score = (
                0.67 * top_similarity
                + 0.10 * min(
                    1.0,
                    max(
                        0.0,
                        (top_similarity - second_similarity + 0.05) / 0.20,
                    ),
                )
                + 0.10 * (1.0 - foreground_ring)
                + 0.07 * isolation
                + 0.06 * gutter
            )

            candidates.append(
                {
                    "x": x,
                    "y": y,
                    "w": width,
                    "h": height,
                    "area": area,
                    "sims": similarities,
                    "top": alphabet[top_index],
                    "topsim": top_similarity,
                    "second": second_similarity,
                    "base": base_score,
                    "darkring": foreground_ring,
                    "iso": isolation,
                    "gutter": gutter,
                    "rg": right_gap,
                    "lg": left_gap,
                    "polarity": polarity,
                }
            )

    scan_foreground(dark, "dark")
    scan_foreground(bright, "light")

    return candidates, gray, dark

def quantile(values: list[float], q: float) -> float:
    return (
        float(np.quantile(np.asarray(values, float), q))
        if values else 0.0
    )

def normalize_ocr_text(text: str) -> str:
    return "".join(
        char for char in text.strip()
        if char.isalnum()
    )

def parse_tesseract_tsv(
    content: str,
    engine: str,
) -> list[dict]:
    lines = [
        line for line in content.splitlines()
        if line.strip()
    ]
    if not lines:
        return []

    header = lines[0].split("\t")
    columns = {name: index for index, name in enumerate(header)}
    required = {"left", "top", "width", "height", "conf", "text"}
    if not required.issubset(columns):
        return []

    words: list[dict] = []
    for line in lines[1:]:
        fields = line.split("\t")
        if len(fields) < len(header):
            continue

        text = fields[columns["text"]].strip()
        normalized = normalize_ocr_text(text)
        if not normalized:
            continue

        try:
            confidence = float(fields[columns["conf"]])
            x = int(float(fields[columns["left"]]))
            y = int(float(fields[columns["top"]]))
            width = int(float(fields[columns["width"]]))
            height = int(float(fields[columns["height"]]))
        except ValueError:
            continue

        if confidence >= 0 and confidence < 20:
            continue
        if width <= 0 or height <= 0:
            continue

        words.append(
            {
                "text": text,
                "normalized": normalized,
                "conf": confidence,
                "bbox": [x, y, width, height],
                "engine": engine,
            }
        )

    return words

def ocr_words_with_pytesseract(image: np.ndarray) -> list[dict]:
    try:
        import pytesseract  # type: ignore
    except ImportError as exc:
        raise RuntimeError("未安装 pytesseract") from exc

    executable = find_tesseract_executable()
    if executable:
        pytesseract.pytesseract.tesseract_cmd = executable

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    data = pytesseract.image_to_data(
        rgb,
        config="--psm 11",
        output_type=pytesseract.Output.DICT,
    )

    words: list[dict] = []
    for index, text in enumerate(data.get("text", [])):
        normalized = normalize_ocr_text(str(text))
        if not normalized:
            continue
        try:
            confidence = float(data["conf"][index])
            x = int(data["left"][index])
            y = int(data["top"][index])
            width = int(data["width"][index])
            height = int(data["height"][index])
        except (KeyError, ValueError, TypeError):
            continue
        if confidence >= 0 and confidence < 20:
            continue
        if width <= 0 or height <= 0:
            continue
        words.append(
            {
                "text": str(text),
                "normalized": normalized,
                "conf": confidence,
                "bbox": [x, y, width, height],
                "engine": "pytesseract",
            }
        )

    return words

def find_tesseract_executable() -> str | None:
    found = shutil.which("tesseract")
    if found:
        return found

    for path in (
        Path("C:/Program Files/Tesseract-OCR/tesseract.exe"),
        Path("C:/Program Files (x86)/Tesseract-OCR/tesseract.exe"),
    ):
        if path.is_file():
            return str(path)

    return None

def ocr_words_with_tesseract_cli(image: np.ndarray) -> list[dict]:
    executable = find_tesseract_executable()
    if not executable:
        raise RuntimeError("未找到 tesseract 可执行文件")

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
        cv2.imwrite(str(temp_path), image)

        result = subprocess.run(
            [
                executable,
                str(temp_path),
                "stdout",
                "--psm",
                "11",
                "tsv",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            message = result.stderr.strip() or result.stdout.strip()
            raise RuntimeError(message[:200])
        return parse_tesseract_tsv(result.stdout, "tesseract")
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

def run_ocr_words(
    image: np.ndarray,
    engine: str,
) -> tuple[list[dict], dict]:
    engines = (
        ["pytesseract", "tesseract"]
        if engine == "auto" else [engine]
    )
    errors: list[str] = []

    for item in engines:
        try:
            if item == "pytesseract":
                words = ocr_words_with_pytesseract(image)
            elif item == "tesseract":
                words = ocr_words_with_tesseract_cli(image)
            else:
                raise RuntimeError(f"不支持的 OCR 引擎：{item}")
        except Exception as exc:
            errors.append(f"{item}: {exc}")
            continue

        return (
            words,
            {
                "enabled": True,
                "engine": item,
                "word_count": len(words),
                "status": "ok",
                "errors": errors,
            },
        )

    return (
        [],
        {
            "enabled": True,
            "engine": None,
            "word_count": 0,
            "status": "unavailable",
            "errors": errors,
        },
    )

def prune_close_stacked_labels(
    labels: list[dict],
    reference_height: float,
    alphabet: str = LETTERS,
) -> list[dict]:
    kept: list[dict] = []
    max_x_gap = 1.35 * reference_height
    max_y_gap = 4.25 * reference_height

    for label in labels:
        if kept:
            previous = kept[-1]
            previous_index = alphabet.index(previous["letter"])
            current_index = alphabet.index(label["letter"])
            previous_x = previous["x"] + 0.5 * previous["w"]
            current_x = label["x"] + 0.5 * label["w"]
            previous_y = previous["y"] + 0.5 * previous["h"]
            current_y = label["y"] + 0.5 * label["h"]
            if (
                current_index == previous_index + 1
                and abs(current_x - previous_x) <= max_x_gap
                and 0 < current_y - previous_y <= max_y_gap
            ):
                continue
        kept.append(label)

    return kept

def prune_trailing_spatial_outliers(
    labels: list[dict],
    reference_height: float,
) -> list[dict]:
    result = list(labels)

    while len(result) >= 4:
        last = result[-1]
        previous = result[-4:-1]
        previous_y = [
            item["y"] + 0.5 * item["h"]
            for item in previous
        ]
        last_y = last["y"] + 0.5 * last["h"]
        if last_y < statistics.median(previous_y) - 8.0 * reference_height:
            result.pop()
            continue
        break

    return result

def prune_bottom_edge_labels(
    labels: list[dict],
    reference_height: float,
    image_height: int,
) -> list[dict]:
    result = list(labels)

    while result:
        last = result[-1]
        label_bottom = float(last["y"] + last["h"])
        bottom_space = image_height - label_bottom
        if (
            bottom_space < 0.75 * reference_height
            and label_bottom > image_height - 1.75 * reference_height
        ):
            result.pop()
            continue
        break

    return result

def repair_same_row_letter_swaps(
    labels: list[dict],
    reference_height: float,
    image_width: int,
    alphabet: str = LETTERS,
) -> list[dict]:
    result = [dict(label) for label in labels]

    changed = True
    while changed:
        changed = False
        by_letter = {
            label["letter"]: label
            for label in result
        }
        for index in range(len(alphabet) - 1):
            if alphabet[index:index + 2] in {"IJ", "ij"}:
                continue
            first = by_letter.get(alphabet[index])
            second = by_letter.get(alphabet[index + 1])
            if first is None or second is None:
                continue

            first_x = first["x"] + 0.5 * first["w"]
            second_x = second["x"] + 0.5 * second["w"]
            first_y = first["y"] + 0.5 * first["h"]
            second_y = second["y"] + 0.5 * second["h"]
            if (
                abs(first_y - second_y) <= 1.25 * reference_height
                and first_x > second_x + 2.0 * reference_height
                and second_x > 0.22 * image_width
            ):
                first["letter"], second["letter"] = (
                    second["letter"],
                    first["letter"],
                )
                changed = True
                break

    return sorted(result, key=lambda item: alphabet.index(item["letter"]))

def repair_co_spatial_swap(
    labels: list[dict],
    reference_height: float,
) -> list[dict]:
    result = [dict(label) for label in labels]
    by_letter = {
        label["letter"]: label
        for label in result
    }
    c_label = by_letter.get("C")
    o_label = by_letter.get("O")
    if c_label is None or o_label is None:
        return result

    def center(label: dict) -> tuple[float, float]:
        return (
            float(label["x"] + 0.5 * label["w"]),
            float(label["y"] + 0.5 * label["h"]),
        )

    def reading_before(first: dict, second: dict) -> bool:
        first_x, first_y = center(first)
        second_x, second_y = center(second)
        if first_y < second_y - 2.5 * reference_height:
            return True
        if abs(first_y - second_y) <= 2.5 * reference_height:
            return first_x < second_x
        return False

    c_sims = c_label.get("sims")
    o_sims = o_label.get("sims")
    if c_sims is None or o_sims is None:
        return result

    c_ambiguous = abs(float(c_sims[2]) - float(c_sims[14])) <= 0.08
    o_ambiguous = abs(float(o_sims[2]) - float(o_sims[14])) <= 0.08
    if (
        c_ambiguous
        and o_ambiguous
        and reading_before(o_label, c_label)
    ):
        c_label["letter"], o_label["letter"] = "O", "C"

    return sorted(result, key=lambda item: ord(item["letter"]))

def repair_right_chart_false_i(
    labels: list[dict],
    candidates: list[dict],
    reference_height: float,
    image_width: int,
    image_height: int,
) -> list[dict]:
    result = [dict(label) for label in labels]
    by_letter = {
        label["letter"]: label
        for label in result
    }
    h_label = by_letter.get("H")
    i_label = by_letter.get("I")
    j_label = by_letter.get("J")
    if h_label is None or i_label is None or j_label is None:
        return result

    def center(item: dict) -> tuple[float, float]:
        return (
            float(item["x"] + 0.5 * item["w"]),
            float(item["y"] + 0.5 * item["h"]),
        )

    h_x, h_y = center(h_label)
    i_x, i_y = center(i_label)
    j_x, j_y = center(j_label)
    i_width = float(i_label["w"])
    i_height = max(float(i_label["h"]), 1.0)

    false_i_pattern = (
        h_x < 0.18 * image_width
        and h_y > 0.38 * image_height
        and i_x > 0.65 * image_width
        and i_y < h_y - 3.0 * reference_height
        and abs(i_y - j_y) <= 2.0 * reference_height
        and i_x > j_x + 2.5 * reference_height
        and i_width <= 0.25 * i_height
    )
    if not false_i_pattern:
        return result

    used_indices = {
        int(label["idx"])
        for label in result
        if "idx" in label
    }
    if "idx" in i_label:
        used_indices.discard(int(i_label["idx"]))

    best: tuple[float, dict] | None = None
    for candidate in candidates:
        if int(candidate.get("idx", -1)) in used_indices:
            continue
        if float(candidate.get("ocr_penalty", 0.0)) >= 0.40:
            continue

        height = max(float(candidate["h"]), 1.0)
        width = float(candidate["w"])
        ratio = height / max(reference_height, 1e-6)
        if not 0.75 <= ratio <= 1.45:
            continue
        if width > 0.45 * height:
            continue

        candidate_x, candidate_y = center(candidate)
        if candidate_y <= h_y + 4.0 * reference_height:
            continue
        if candidate_y > image_height - 1.0 * reference_height:
            continue
        if not (
            candidate_x < 0.18 * image_width
            or abs(candidate_x - h_x) <= 4.0 * reference_height
        ):
            continue

        i_shape = float(candidate["sims"][8])
        if i_shape < 0.82:
            continue
        if float(candidate.get("topq", 0.0)) < 0.82:
            continue
        if candidate_edge_penalty(
            candidate,
            image_width,
            image_height,
        ) >= 0.25:
            continue

        score = float(candidate.get("topq", 0.0))
        if candidate.get("top") == "I":
            score += 0.08
        score -= min(
            0.08,
            abs(candidate_x - h_x)
            / max(reference_height, 1e-6)
            * 0.01,
        )

        if best is None or score > best[0]:
            best = (score, candidate)

    if best is None:
        return result

    score, candidate = best
    replacement = dict(candidate)
    replacement["letter"] = "I"
    replacement["quality"] = float(score)
    replacement["shape_for_letter"] = float(candidate["sims"][8])
    replacement["missing"] = False

    for index, label in enumerate(result):
        if label["letter"] == "I":
            result[index] = replacement
            break

    return sorted(result, key=lambda item: ord(item["letter"]))

def candidate_word_penalty(candidate: dict) -> float:
    height = max(float(candidate["h"]), 1.0)
    left_gap = float(candidate["lg"])
    right_gap = float(candidate["rg"])
    penalty = 0.0

    if min(left_gap, right_gap) < 0.12 * height:
        penalty += 0.04
    if left_gap < 0.22 * height and right_gap < 0.22 * height:
        penalty += 0.12
    if right_gap < 0.16 * height and float(candidate["gutter"]) < 0.78:
        penalty += 0.06

    return penalty

def candidate_neighbor_word_penalty(
    candidate: dict,
    candidates: list[dict],
) -> float:
    x = float(candidate["x"])
    y = float(candidate["y"])
    width = float(candidate["w"])
    height = max(float(candidate["h"]), 1.0)
    x2 = x + width
    y2 = y + height
    left_neighbors: list[float] = []
    right_neighbors: list[float] = []

    for other in candidates:
        if other is candidate:
            continue
        other_height = max(float(other["h"]), 1.0)
        if not 0.55 * height <= other_height <= 1.45 * height:
            continue

        other_y = float(other["y"])
        other_y2 = other_y + other_height
        overlap_y = max(0.0, min(y2, other_y2) - max(y, other_y))
        if overlap_y / min(height, other_height) < 0.55:
            continue

        other_x = float(other["x"])
        other_x2 = other_x + float(other["w"])
        if other_x >= x2:
            gap = other_x - x2
            if gap <= 4.0 * height:
                right_neighbors.append(gap)
        elif other_x2 <= x:
            gap = x - other_x2
            if gap <= 4.0 * height:
                left_neighbors.append(gap)

    left_neighbors.sort()
    right_neighbors.sort()
    penalty = 0.0

    if (
        len(right_neighbors) >= 2
        and right_neighbors[0] <= 0.45 * height
    ):
        penalty += 0.30
    elif (
        len(right_neighbors) >= 3
        and right_neighbors[0] <= 0.75 * height
    ):
        penalty += 0.22

    if (
        left_neighbors
        and right_neighbors
        and left_neighbors[0] <= 0.45 * height
        and right_neighbors[0] <= 0.45 * height
    ):
        penalty += 0.24

    return min(0.42, penalty)

def candidate_edge_penalty(
    candidate: dict,
    image_width: int,
    image_height: int,
) -> float:
    x = float(candidate["x"])
    y = float(candidate["y"])
    width = float(candidate["w"])
    height = max(float(candidate["h"]), 1.0)
    penalty = 0.0

    if y > 0.94 * image_height and x > 0.18 * image_width:
        penalty += 0.30
    if (
        candidate["top"] == "I"
        and width <= 0.30 * height
        and y > 0.90 * image_height
        and x > 0.22 * image_width
    ):
        penalty += 0.18

    return penalty

def candidate_ocr_penalty(
    candidate: dict,
    ocr_words: list[dict] | None,
    reference_height: float,
) -> float:
    if not ocr_words:
        return 0.0

    x1 = float(candidate["x"])
    y1 = float(candidate["y"])
    x2 = x1 + float(candidate["w"])
    y2 = y1 + float(candidate["h"])
    center_x = 0.5 * (x1 + x2)
    center_y = 0.5 * (y1 + y2)
    candidate_area = max(1.0, (x2 - x1) * (y2 - y1))
    best_penalty = 0.0

    for word in ocr_words:
        normalized = str(word.get("normalized", ""))
        alpha_count = sum(char.isalpha() for char in normalized)
        if len(normalized) < 3 or alpha_count < 2:
            continue

        wx, wy, ww, wh = [
            float(value) for value in word.get("bbox", [0, 0, 0, 0])
        ]
        if ww <= 0 or wh <= 0:
            continue
        if ww > 10.0 * reference_height:
            continue
        if wh > 2.8 * reference_height:
            continue

        confidence = float(word.get("conf", -1.0))
        if confidence >= 0 and confidence < 60:
            continue

        padding = max(1.0, 0.12 * reference_height)
        word_x1 = wx - padding
        word_y1 = wy - padding
        word_x2 = wx + ww + padding
        word_y2 = wy + wh + padding
        center_inside = (
            word_x1 <= center_x <= word_x2
            and word_y1 <= center_y <= word_y2
        )

        overlap_x = max(0.0, min(x2, word_x2) - max(x1, word_x1))
        overlap_y = max(0.0, min(y2, word_y2) - max(y1, word_y1))
        overlap_ratio = overlap_x * overlap_y / candidate_area

        if center_inside or overlap_ratio >= 0.45:
            penalty = 0.44
            if len(normalized) >= 4:
                penalty += 0.10
            candidate_top = str(candidate.get("top", ""))
            if candidate_top.lower() in normalized.lower():
                penalty += 0.06
            best_penalty = max(best_penalty, penalty)

    return min(0.68, best_penalty)

def is_anchor_like_candidate(
    candidate: dict,
    reference_height: float,
    image_width: int,
    image_height: int,
    relaxed: bool,
    alphabet: str = LETTERS,
) -> bool:
    height = max(float(candidate["h"]), 1.0)
    ratio = height / max(reference_height, 1e-6)
    top_similarity = float(candidate["topsim"])
    quality = float(candidate["topq"])
    left_gap = float(candidate["lg"])
    right_gap = float(candidate["rg"])
    ocr_penalty = float(candidate.get("ocr_penalty", 0.0))

    if ocr_penalty >= 0.40:
        return False

    if relaxed:
        maximum_ratio = 1.52 if alphabet == LOWER_LETTERS else 1.38
        return (
            0.76 <= ratio <= maximum_ratio
            and quality >= 0.78
            and (
                top_similarity >= 0.82
                or quality >= 0.83
            )
            and (
                left_gap >= 0.38 * height
                or float(candidate["x"]) < 0.04 * image_width
            )
            and float(candidate["gutter"]) >= 0.50
            and candidate_edge_penalty(
                candidate,
                image_width,
                image_height,
            ) < 0.25
        )

    return (
        0.84 <= ratio <= 1.28
        and min(left_gap, right_gap) >= 0.20 * height
        and top_similarity >= 0.80
        and quality >= 0.74
        and candidate_edge_penalty(
            candidate,
            image_width,
            image_height,
        ) < 0.25
    )

def candidate_initial_letters(
    candidate: dict,
    alphabet: str,
) -> list[str]:
    letters = [str(candidate["top"])]
    if alphabet != LOWER_LETTERS:
        return letters

    height = max(float(candidate["h"]), 1.0)
    width = float(candidate["w"])
    if candidate["top"] == "l" and "f" in alphabet:
        f_similarity = float(candidate["sims"][alphabet.index("f")])
        if (
            width > 0.30 * height
            and min(float(candidate["lg"]), float(candidate["rg"]))
            >= 0.40 * height
            and f_similarity >= 0.78
            and float(candidate["topsim"]) - f_similarity <= 0.04
        ):
            letters.append("f")

    if width > 0.55 * height:
        return letters
    if min(float(candidate["lg"]), float(candidate["rg"])) < 0.40 * height:
        return letters

    top_similarity = float(candidate["topsim"])
    for letter in ("i", "j"):
        if letter not in alphabet or letter in letters:
            continue
        similarity = float(candidate["sims"][alphabet.index(letter)])
        if (
            similarity >= 0.76
            and top_similarity - similarity <= 0.055
        ):
            letters.append(letter)
    return letters

def is_lower_dotted_anchor_candidate(
    candidate: dict,
    reference_height: float,
    image_width: int,
    image_height: int,
) -> bool:
    height = max(float(candidate["h"]), 1.0)
    width = float(candidate["w"])
    ratio = height / max(reference_height, 1e-6)
    if not 0.82 <= ratio <= 1.60:
        return False
    if width > 0.65 * height:
        return False
    if min(float(candidate["lg"]), float(candidate["rg"])) < 0.45 * height:
        return False
    if float(candidate.get("ocr_penalty", 0.0)) >= 0.40:
        return False
    if float(candidate.get("neighbor_word_penalty", 0.0)) >= 0.24:
        return False
    if float(candidate.get("gutter", 0.0)) < 0.72:
        return False
    if candidate_edge_penalty(candidate, image_width, image_height) >= 0.25:
        return False

    i_similarity = float(candidate["sims"][LOWER_LETTERS.index("i")])
    j_similarity = float(candidate["sims"][LOWER_LETTERS.index("j")])
    dotted_similarity = max(i_similarity, j_similarity)
    top_similarity = float(candidate["topsim"])
    quality = float(candidate.get("topq", 0.0))
    return (
        dotted_similarity >= 0.74
        and top_similarity - dotted_similarity <= 0.12
        and quality >= 0.72
    )

def detect_panel_labels_for_alphabet(
    image: np.ndarray,
    ocr_words: list[dict] | None = None,
    alphabet: str = LETTERS,
    minimum_sequence_length: int | None = None,
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    float,
    int,
    np.ndarray,
    dict[str, dict],
]:
    image_height, image_width = image.shape[:2]
    candidates, _, _ = extract_label_candidates(image, alphabet)

    # 优先用左上区域的 A 估计顶层标签字号。
    a_candidates: list[tuple[float, dict]] = []
    for candidate in candidates:
        a_similarity = float(candidate["sims"][0])
        if (
            a_similarity > 0.65
            and candidate["x"] < 0.38 * image_width
            and candidate["y"] < 0.30 * image_height
        ):
            score = (
                candidate["base"]
                + 0.18 * (1.0 - candidate["x"] / image_width)
                + 0.18 * (1.0 - candidate["y"] / image_height)
                + 0.05 * a_similarity
            )
            a_candidates.append((score, candidate))

    if a_candidates:
        shape_specific = (
            [
                item
                for item in a_candidates
                if float(item[1]["sims"][0])
                >= float(item[1]["topsim"]) - 0.04
            ]
            if alphabet == LOWER_LETTERS
            else []
        )
        reference_height = float(
            max(
                shape_specific or a_candidates,
                key=lambda item: item[0],
            )[1]["h"]
        )
    else:
        eligible = [
            candidate["h"]
            for candidate in candidates
            if candidate["topsim"] > 0.72
            and candidate["base"] > 0.68
        ]
        reference_height = (
            quantile(eligible, 0.78)
            if eligible else 0.02 * image_height
        )

    initial: dict[str, dict] = {}

    for index, candidate in enumerate(candidates):
        ratio = candidate["h"] / max(reference_height, 1e-6)
        size_score = float(
            np.exp(
                -0.5
                * (
                    np.log(max(ratio, 1e-4)) / 0.24
                ) ** 2
            )
        )
        ocr_penalty = candidate_ocr_penalty(
            candidate,
            ocr_words,
            reference_height,
        )
        neighbor_word_penalty = candidate_neighbor_word_penalty(
            candidate,
            candidates,
        )
        quality = (
            0.66 * candidate["topsim"]
            + 0.16 * size_score
            + 0.08 * (1.0 - candidate["darkring"])
            + 0.05 * candidate["iso"]
            + 0.05 * candidate["gutter"]
        )
        quality -= candidate_word_penalty(candidate)
        quality -= neighbor_word_penalty
        quality -= ocr_penalty
        quality -= candidate_edge_penalty(
            candidate,
            image_width,
            image_height,
        )
        candidate["topq"] = quality
        candidate["idx"] = index
        candidate["size_score"] = size_score
        candidate["ocr_penalty"] = ocr_penalty
        candidate["neighbor_word_penalty"] = neighbor_word_penalty

        if not (
            is_anchor_like_candidate(
                candidate,
                reference_height,
                image_width,
                image_height,
                relaxed=False,
                alphabet=alphabet,
            )
            or is_anchor_like_candidate(
                candidate,
                reference_height,
                image_width,
                image_height,
                relaxed=True,
                alphabet=alphabet,
            )
            or (
                alphabet == LOWER_LETTERS
                and is_lower_dotted_anchor_candidate(
                    candidate,
                    reference_height,
                    image_width,
                    image_height,
                )
            )
        ):
            continue

        for letter in candidate_initial_letters(candidate, alphabet):
            if (
                letter not in initial
                or quality > initial[letter]["topq"]
            ):
                initial[letter] = candidate

    # 从高置信度字母锚点推断连续 A...N 序列终点。
    presence = np.array(
        [
            initial.get(letter, {}).get("topq", 0.0)
            for letter in alphabet
        ]
    )
    running_score = 0.0
    best_score = -1e9
    endpoint = 1

    for n in range(1, len(alphabet) + 1):
        quality = float(presence[n - 1])
        running_score += (
            min(0.52, quality - 0.56) if quality > 0 else -0.24
        )
        objective = running_score - (0.12 if quality == 0 else 0.0)
        if objective > best_score:
            best_score = objective
            endpoint = n

    strong_indices = np.flatnonzero(presence >= 0.78)
    for index in strong_indices[::-1]:
        prefix = presence[:index + 1]
        support = int(np.count_nonzero(prefix > 0))
        required = max(3, int(np.ceil(0.42 * (index + 1))))
        if support >= required:
            endpoint = max(endpoint, int(index) + 1)
            break

    for i in range(1, endpoint):
        if presence[i - 1] == 0 and presence[i] == 0:
            endpoint = max(1, i - 1)
            break

    while endpoint > 1 and presence[endpoint - 1] == 0:
        endpoint -= 1

    if minimum_sequence_length is not None:
        endpoint = max(endpoint, min(len(alphabet), minimum_sequence_length))

    sequence_length = endpoint

    pool: list[dict] = []
    for candidate in candidates:
        ratio = candidate["h"] / max(reference_height, 1e-6)
        if 0.70 <= ratio <= 1.55 and candidate["topsim"] > 0.48:
            pool.append(candidate)

    pool_size = len(pool)
    dummy_score = 0.61
    scores = np.full(
        (sequence_length, pool_size + sequence_length),
        dummy_score,
        np.float32,
    )

    for letter_index in range(sequence_length):
        letter = alphabet[letter_index]
        for candidate_index, candidate in enumerate(pool):
            ratio = candidate["h"] / max(reference_height, 1e-6)
            size_score = float(
                np.exp(
                    -0.5
                    * (
                        np.log(max(ratio, 1e-4)) / 0.27
                    ) ** 2
                )
            )
            shape_score = float(candidate["sims"][letter_index])
            quality = (
                0.66 * shape_score
                + 0.16 * size_score
                + 0.08 * (1.0 - candidate["darkring"])
                + 0.05 * candidate["iso"]
                + 0.05 * candidate["gutter"]
            )
            quality -= candidate_word_penalty(candidate)
            quality -= float(candidate.get("neighbor_word_penalty", 0.0))
            quality -= float(candidate.get("ocr_penalty", 0.0))
            quality -= candidate_edge_penalty(
                candidate,
                image_width,
                image_height,
            )

            if candidate["top"] == letter:
                quality += 0.08
            if (
                letter in initial
                and candidate["idx"] == initial[letter]["idx"]
            ):
                quality += 0.16
            if shape_score < 0.52:
                quality -= 0.16

            scores[letter_index, candidate_index] = quality

    rows, columns = linear_sum_assignment(-scores)
    detected: list[dict] = []
    assignment_rows: list[dict] = []

    for row, column in zip(rows, columns):
        letter = alphabet[row]
        if (
            column < pool_size
            and scores[row, column] >= dummy_score + 0.015
        ):
            candidate = dict(pool[column])
            candidate["letter"] = letter
            candidate["quality"] = float(scores[row, column])
            candidate["shape_for_letter"] = float(
                candidate["sims"][row]
            )
            candidate["missing"] = False
            detected.append(candidate)
            assignment_rows.append(candidate)
        else:
            assignment_rows.append(
                {
                    "letter": letter,
                    "quality": dummy_score,
                    "missing": True,
                }
            )

    detected.sort(key=lambda item: alphabet.index(item["letter"]))
    detected = prune_close_stacked_labels(
        detected,
        reference_height,
        alphabet,
    )
    detected = prune_trailing_spatial_outliers(
        detected,
        reference_height,
    )
    detected = repair_same_row_letter_swaps(
        detected,
        reference_height,
        image_width,
        alphabet,
    )
    if alphabet == UPPER_LETTERS:
        detected = repair_co_spatial_swap(
            detected,
            reference_height,
        )
        detected = repair_right_chart_false_i(
            detected,
            candidates,
            reference_height,
            image_width,
            image_height,
        )
    detected = prune_bottom_edge_labels(
        detected,
        reference_height,
        image_height,
    )
    if detected:
        sequence_length = max(
            alphabet.index(label["letter"]) for label in detected
        ) + 1

    return (
        detected,
        assignment_rows,
        candidates,
        reference_height,
        sequence_length,
        presence,
        initial,
    )

def score_label_detection(
    result: tuple[
        list[dict],
        list[dict],
        list[dict],
        float,
        int,
        np.ndarray,
        dict[str, dict],
    ],
    image_shape: tuple[int, ...],
    alphabet: str,
) -> float:
    labels, _, _, reference_height, _, presence, _ = result
    if not labels:
        return -100.0

    image_height, image_width = image_shape[:2]
    indices = sorted(
        {
            alphabet.index(label["letter"])
            for label in labels
            if label["letter"] in alphabet
        }
    )
    if not indices:
        return -100.0

    max_index = max(indices)
    expected = max_index + 1
    support = len(indices)
    missing = expected - support
    qualities = [float(label.get("quality", 0.0)) for label in labels]
    mean_quality = statistics.mean(qualities) if qualities else 0.0
    strong_count = sum(1 for value in qualities if value >= 0.78)
    continuity = support / max(expected, 1)

    first_bonus = 0.0
    first_label = next(
        (label for label in labels if label["letter"] == alphabet[0]),
        None,
    )
    if first_label is not None:
        first_x = float(first_label["x"])
        first_y = float(first_label["y"])
        if first_x < 0.22 * image_width and first_y < 0.22 * image_height:
            first_bonus = 0.55
        elif first_x < 0.36 * image_width and first_y < 0.32 * image_height:
            first_bonus = 0.28

    present_prefix = int(np.count_nonzero(presence[:expected] > 0.0))
    score = (
        1.05 * support
        + 1.85 * mean_quality
        + 1.30 * continuity
        + 0.10 * strong_count
        + 0.05 * present_prefix
        + first_bonus
        - 0.32 * missing
    )

    # Tiny labels that are far from the learned top-level size are usually
    # text fragments, so avoid letting a large count alone win the mode.
    size_outliers = 0
    for label in labels:
        ratio = float(label["h"]) / max(reference_height, 1e-6)
        if ratio < 0.70 or ratio > 1.55:
            size_outliers += 1
    score -= 0.18 * size_outliers
    return float(score)

def extended_sequence_endpoint(
    result: tuple[
        list[dict],
        list[dict],
        list[dict],
        float,
        int,
        np.ndarray,
        dict[str, dict],
    ],
) -> int | None:
    """Suggest a conservative retry when strong anchors survive truncation."""
    labels, _, _, _, sequence_length, presence, _ = result
    if len(labels) < 3:
        return None
    strong_after = [
        index
        for index in np.flatnonzero(presence >= 0.78)
        if index >= sequence_length
    ]
    if len(strong_after) < 2:
        return None

    nearby = [
        index
        for index in strong_after
        if index < sequence_length + 4
    ]
    if not nearby:
        return None
    return int(max(nearby)) + 1

def maybe_extend_label_result(
    image: np.ndarray,
    ocr_words: list[dict] | None,
    alphabet: str,
    baseline: tuple[
        list[dict],
        list[dict],
        list[dict],
        float,
        int,
        np.ndarray,
        dict[str, dict],
    ],
) -> tuple[
    tuple[
        list[dict],
        list[dict],
        list[dict],
        float,
        int,
        np.ndarray,
        dict[str, dict],
    ],
    dict[str, object],
]:
    baseline_score = score_label_detection(baseline, image.shape, alphabet)
    endpoint = extended_sequence_endpoint(baseline)
    diagnostics: dict[str, object] = {
        "strategy": "baseline",
        "baseline_sequence_length": baseline[4],
        "extended_sequence_length": None,
        "baseline_score": baseline_score,
        "extended_score": None,
    }
    if endpoint is None:
        return baseline, diagnostics

    extended = detect_panel_labels_for_alphabet(
        image,
        ocr_words,
        alphabet,
        minimum_sequence_length=endpoint,
    )
    extended_score = score_label_detection(extended, image.shape, alphabet)
    baseline_labels = baseline[0]
    extended_labels = extended[0]
    qualities = [float(label.get("quality", 0.0)) for label in extended_labels]
    mean_quality = statistics.mean(qualities) if qualities else 0.0
    adopt = (
        len(extended_labels) >= len(baseline_labels) + 2
        and mean_quality >= 0.72
        and extended_score >= baseline_score + 1.0
    )
    diagnostics.update(
        extended_sequence_length=endpoint,
        extended_score=extended_score,
        strategy="extended" if adopt else "baseline",
    )
    return (extended if adopt else baseline), diagnostics

def detect_panel_labels(
    image: np.ndarray,
    ocr_words: list[dict] | None = None,
    label_mode: str = "auto",
) -> tuple[
    list[dict],
    list[dict],
    list[dict],
    float,
    int,
    np.ndarray,
    dict[str, dict],
    dict[str, object],
]:
    from .layout import assess_detection_layout_evidence, make_content_mask
    if label_mode == "upper":
        baseline = detect_panel_labels_for_alphabet(
            image,
            ocr_words,
            UPPER_LETTERS,
        )
        result, strategy = maybe_extend_label_result(
            image, ocr_words, UPPER_LETTERS, baseline
        )
        score = score_label_detection(result, image.shape, UPPER_LETTERS)
        return (
            *result,
            {
                "requested": label_mode,
                "selected": "upper",
                "alphabet": UPPER_LETTERS,
                "score": score,
                "scores": {"upper": score},
                **strategy,
            },
        )

    if label_mode == "lower":
        baseline = detect_panel_labels_for_alphabet(
            image,
            ocr_words,
            LOWER_LETTERS,
        )
        result, strategy = maybe_extend_label_result(
            image, ocr_words, LOWER_LETTERS, baseline
        )
        score = score_label_detection(result, image.shape, LOWER_LETTERS)
        return (
            *result,
            {
                "requested": label_mode,
                "selected": "lower",
                "alphabet": LOWER_LETTERS,
                "score": score,
                "scores": {"lower": score},
                **strategy,
            },
        )

    upper_baseline = detect_panel_labels_for_alphabet(
        image,
        ocr_words,
        UPPER_LETTERS,
    )
    lower_baseline = detect_panel_labels_for_alphabet(
        image,
        ocr_words,
        LOWER_LETTERS,
    )
    upper_result, upper_strategy = maybe_extend_label_result(
        image, ocr_words, UPPER_LETTERS, upper_baseline
    )
    lower_result, lower_strategy = maybe_extend_label_result(
        image, ocr_words, LOWER_LETTERS, lower_baseline
    )
    upper_score = score_label_detection(
        upper_result,
        image.shape,
        UPPER_LETTERS,
    )
    lower_score = score_label_detection(
        lower_result,
        image.shape,
        LOWER_LETTERS,
    )
    upper_arbitration_score = float(upper_strategy["baseline_score"])
    lower_arbitration_score = float(lower_strategy["baseline_score"])
    lower_labels = lower_baseline[0]
    upper_first = next(
        (
            label
            for label in upper_baseline[0]
            if label["letter"] == UPPER_LETTERS[0]
        ),
        None,
    )
    lower_first = next(
        (
            label
            for label in lower_labels
            if label["letter"] == LOWER_LETTERS[0]
        ),
        None,
    )
    lower_origin_anchor = bool(
        lower_first is not None
        and float(lower_first.get("quality", 0.0)) >= 0.90
        and float(lower_first["x"]) < 0.08 * image.shape[1]
        and float(lower_first["y"]) < 0.08 * image.shape[0]
    )
    upper_origin_anchor = bool(
        upper_first is not None
        and float(upper_first.get("quality", 0.0)) >= 0.90
        and float(upper_first["x"]) < 0.08 * image.shape[1]
        and float(upper_first["y"]) < 0.08 * image.shape[0]
    )
    lower_mean_quality = (
        statistics.mean(
            float(label.get("quality", 0.0)) for label in lower_labels
        )
        if lower_labels
        else 0.0
    )
    lower_origin_override = (
        len(lower_labels) >= 3
        and lower_origin_anchor
        and not upper_origin_anchor
        and lower_mean_quality >= 0.92
        and lower_arbitration_score >= upper_arbitration_score - 0.40
    )
    score_prefers_lower = (
        len(lower_labels) >= 3
        and (
            lower_arbitration_score > upper_arbitration_score + 0.35
            or lower_origin_override
        )
    )
    layout_mask = make_content_mask(image)
    upper_layout = assess_detection_layout_evidence(
        image,
        layout_mask,
        upper_result,
        UPPER_LETTERS,
    )
    lower_layout = assess_detection_layout_evidence(
        image,
        layout_mask,
        lower_result,
        LOWER_LETTERS,
    )
    layout_override: str | None = None
    if lower_layout["strong"] and not upper_layout["strong"]:
        choose_lower = True
        layout_override = "lower"
    elif upper_layout["strong"] and not lower_layout["strong"]:
        choose_lower = False
        layout_override = "upper"
    else:
        choose_lower = score_prefers_lower
    selected_mode = "lower" if choose_lower else "upper"
    selected_alphabet = LOWER_LETTERS if choose_lower else UPPER_LETTERS
    selected_result = lower_result if choose_lower else upper_result
    selected_score = lower_score if choose_lower else upper_score
    selected_strategy = lower_strategy if choose_lower else upper_strategy

    return (
        *selected_result,
        {
            "requested": label_mode,
            "selected": selected_mode,
            "alphabet": selected_alphabet,
            "score": selected_score,
            "scores": {
                "upper": upper_score,
                "lower": lower_score,
            },
            "arbitration_scores": {
                "upper": upper_arbitration_score,
                "lower": lower_arbitration_score,
            },
            "arbitration_strategy": "baseline_before_extension",
            "lower_origin_override": lower_origin_override,
            "layout_override": layout_override,
            "layout_evidence": {
                "upper": upper_layout,
                "lower": lower_layout,
            },
            **selected_strategy,
        },
    )
