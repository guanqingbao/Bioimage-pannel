from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import fitz
from PIL import Image, ImageDraw

from image_batch.pdf_figures.pdf_figure_extractor import extract_pdf_figures


class CrossPageFigureTests(unittest.TestCase):
    def test_clipped_image_and_next_page_caption_are_matched(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "cross_page.pdf"
            image = Image.new("RGB", (800, 500), "white")
            draw = ImageDraw.Draw(image)
            draw.rectangle((20, 20, 780, 480), outline="black", width=6)
            draw.rectangle((80, 80, 350, 420), fill="red")
            draw.rectangle((450, 80, 720, 420), fill="blue")
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")

            document = fitz.open()
            first = document.new_page(width=595, height=842)
            first.insert_image(
                fitz.Rect(70, 520, 525, 850),
                stream=image_bytes.getvalue(),
                keep_proportion=False,
            )
            second = document.new_page(width=595, height=842)
            second.insert_text(
                (72, 92),
                "Fig. 1. Two colored panels in the previous page image.",
                fontsize=10,
            )
            document.save(pdf_path)
            document.close()

            result = extract_pdf_figures(pdf_path, root / "output", dpi=96)
            matched = [
                figure for figure in result.figures
                if figure.caption_figure_number == "1"
            ]
            self.assertEqual(len(matched), 1)
            self.assertEqual(matched[0].page_number, 1)
            self.assertEqual(matched[0].caption_page_number, 2)
            self.assertEqual(matched[0].extraction_method, "embedded_raster")
            self.assertIn("cross_page_caption_match", matched[0].quality_flags)
            self.assertIn("source_image_clipped_by_page", matched[0].quality_flags)

    def test_caption_continuation_is_collected_on_next_page(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pdf_path = root / "continued_caption.pdf"
            image = Image.new("RGB", (800, 400), "green")
            image_bytes = BytesIO()
            image.save(image_bytes, format="PNG")

            document = fitz.open()
            first = document.new_page(width=595, height=842)
            first.insert_image(fitz.Rect(70, 430, 525, 720), stream=image_bytes.getvalue())
            first.insert_text(
                (72, 752),
                "Fig. 2. A printed figure with a caption that continues after the page turn.",
                fontsize=10,
            )
            first.insert_text(
                (72, 773),
                "The first page discusses the panels and the measured result.",
                fontsize=10,
            )
            second = document.new_page(width=595, height=842)
            second.insert_text(
                (72, 82),
                "(Middle columns: the next page continues the caption description)",
                fontsize=10,
            )
            second.insert_text(
                (72, 102),
                "and finishes the panel explanation before ordinary article text.",
                fontsize=10,
            )
            document.save(pdf_path)
            document.close()

            result = extract_pdf_figures(pdf_path, root / "output", dpi=96)
            matched = [
                figure for figure in result.figures
                if figure.caption_figure_number == "2"
            ]
            self.assertEqual(len(matched), 1)
            self.assertEqual(matched[0].page_number, 1)
            self.assertEqual(matched[0].caption_end_page_number, 2)
            self.assertIn("caption_spans_pages", matched[0].quality_flags)


if __name__ == "__main__":
    unittest.main()
