"""JSON compatibility layer for the deterministic image pipeline.

The web application keeps the small JSON contract used by the original panel
editor while calling the copied PDF extractor and panel splitter directly.
No model provider or API key is required.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .panel_splitter import split_figure_panels as _split_figure_panels
from .pdf_figures import extract_pdf_figures as _extract_pdf_figures
from .pdf_figures import parse_page_selector, sanitize_name


def _collect_pdf_inputs(source: Path, recursive: bool) -> list[Path]:
    if source.is_file():
        return [source] if source.suffix.lower() == ".pdf" else []
    if not source.is_dir():
        return []
    iterator = source.rglob("*.pdf") if recursive else source.glob("*.pdf")
    return sorted(path.resolve() for path in iterator if path.is_file())


def extract_pdf_figures(
    input_path: str,
    output_root: str = "",
    recursive: bool = False,
    dpi: int = 300,
    pages: str = "",
) -> str:
    source = Path(input_path).expanduser().resolve()
    root = (
        Path(output_root).expanduser().resolve()
        if output_root.strip()
        else (source.parent if source.is_file() else source)
    )
    pdf_paths = _collect_pdf_inputs(source, recursive)
    if not pdf_paths:
        raise ValueError(f"No PDF files found: {source}")

    records: list[dict[str, Any]] = []
    selected_pages = parse_page_selector(pages)
    for pdf_path in pdf_paths:
        article_dir = root / sanitize_name(pdf_path.stem)
        result = _extract_pdf_figures(
            pdf_path=pdf_path,
            output_dir=article_dir,
            dpi=dpi,
            pages=selected_pages,
            overwrite=True,
        )
        payload = result.to_dict()
        records.append(
            {
                "pdf_path": str(pdf_path),
                "figures_dir": str(result.figures_dir),
                "figure_count": len(result.figures),
                "caption_count": len(result.captions),
                "metadata_path": str(result.metadata_path),
                "review_required_count": payload["review_required_count"],
                "review_items": payload["review_items"],
                "figure_quality": {
                    Path(figure.image_path).name: {
                        "figure_id": figure.figure_id,
                        "figure_number": figure.caption_figure_number,
                        "caption_kind": figure.caption_kind,
                        "caption": figure.caption,
                        "page_number": figure.page_number,
                        "width_px": figure.width_px,
                        "height_px": figure.height_px,
                        "dpi": figure.dpi,
                        "expected_panel_labels": figure.expected_panel_labels,
                        "native_panel_labels": figure.native_panel_labels,
                        "extraction_method": figure.extraction_method,
                        "quality_flags": figure.quality_flags,
                        "needs_review": figure.needs_review,
                    }
                    for figure in result.figures
                },
            }
        )
    return json.dumps({"pdf_count": len(records), "articles": records}, ensure_ascii=False)


def split_figure_panels(
    figures_path: str,
    output_dir: str = "",
    recursive: bool = False,
    overview: bool = True,
    ocr_filter: bool = False,
    ocr_engine: str = "auto",
    label_mode: str = "auto",
) -> str:
    source = Path(figures_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Figure input not found: {source}")
    destination = (
        Path(output_dir).expanduser().resolve()
        if output_dir.strip()
        else (
            source / "panel_layout_results"
            if source.is_dir()
            else source.parent / "panel_layout_results"
        )
    )
    records = _split_figure_panels(
        [source],
        destination,
        recursive=recursive,
        overview=overview,
        ocr_filter=ocr_filter,
        ocr_engine=ocr_engine,
        label_mode=label_mode,
    )
    return json.dumps(
        {
            "figures_path": str(source),
            "output_dir": str(destination),
            "summary_path": str(destination / "batch_summary.json"),
            "overview_path": str(destination / "overview.png") if overview else None,
            "processed_image_count": len(records),
            "total_panel_count": sum(record["panel_count"] for record in records),
            "results": records,
        },
        ensure_ascii=False,
    )
