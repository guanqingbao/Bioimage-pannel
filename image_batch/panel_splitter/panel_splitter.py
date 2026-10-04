from __future__ import annotations

import argparse
import json
import statistics
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .output import build_overview, collect_inputs, process_image


def split_figure_panels(
    image_inputs: Sequence[str | Path],
    output_dir: str | Path,
    *,
    recursive: bool = False,
    overview: bool = True,
    ocr_filter: bool = False,
    ocr_engine: str = "auto",
    label_mode: str = "auto",
) -> list[dict[str, Any]]:
    """Split whole-figure images and write panel crops plus batch metadata."""
    destination = Path(output_dir).expanduser().resolve()
    inputs = collect_inputs([str(path) for path in image_inputs], recursive)
    if not inputs:
        return []

    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for index, image_path in enumerate(inputs, 1):
        preview_path, metadata = process_image(
            image_path,
            destination,
            ocr_filter=ocr_filter,
            ocr_engine=ocr_engine,
            label_mode=label_mode,
        )
        panels = metadata["panels"]
        confidences = [float(panel["confidence"]) for panel in panels.values()]
        labels = "".join(sorted(panels))
        low_confidence = sorted(
            letter
            for letter, panel in panels.items()
            if float(panel["confidence"]) < 0.70
        )
        record = {
            "source": str(image_path),
            "preview": str(preview_path),
            "labels": labels,
            "panel_count": len(panels),
            "mean_confidence": statistics.mean(confidences) if confidences else 0.0,
            "minimum_confidence": min(confidences) if confidences else 0.0,
            "low_confidence_panels": low_confidence,
            "unresolved_regions": metadata["diagnostics"],
            "split_decision": metadata.get("split_decision", {}),
            "figure_constraints": metadata.get("figure_constraints", {}),
            "detected_labels": "".join(sorted(metadata.get("detected_labels", {}))),
        }
        records.append(record)
        print(
            f"    [{index:03d}/{len(inputs):03d}] "
            f"{image_path.name}: labels={labels or '-'}, "
            f"panels={len(panels)}, "
            f"mean={record['mean_confidence']:.3f}, "
            f"low={','.join(low_confidence) or '-'}"
        )

    (destination / "batch_summary.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if overview:
        build_overview(records, destination / "overview.png")
    return records


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Split scientific whole-figure images into labeled top-level panels."
    )
    parser.add_argument("inputs", nargs="+", help="Image files or directories")
    parser.add_argument("-o", "--output-dir", type=Path, default=Path("panel_layout_results"))
    parser.add_argument("-r", "--recursive", action="store_true")
    parser.add_argument("--overview", action="store_true")
    parser.add_argument("--ocr-filter", action="store_true")
    parser.add_argument(
        "--ocr-engine", choices=("auto", "pytesseract", "tesseract"), default="auto"
    )
    parser.add_argument("--label-mode", choices=("auto", "upper", "lower"), default="auto")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    records = split_figure_panels(
        args.inputs,
        args.output_dir,
        recursive=args.recursive,
        overview=args.overview,
        ocr_filter=args.ocr_filter,
        ocr_engine=args.ocr_engine,
        label_mode=args.label_mode,
    )
    if not records:
        raise SystemExit("No supported images found.")
    print(f"Complete. Output: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
