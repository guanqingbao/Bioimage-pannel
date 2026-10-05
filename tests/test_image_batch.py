from __future__ import annotations

import importlib.util
from io import BytesIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import fitz
from fastapi.testclient import TestClient
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
SPEC = importlib.util.spec_from_file_location("standalone_image_batch_app", PROJECT_ROOT / "app.py")
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("Could not import image-batch/app.py")
image_batch_app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(image_batch_app)


class ImageBatchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = TemporaryDirectory()
        self.data_root = Path(self.temporary.name) / "data"
        self.data_root.mkdir()
        self.data_patch = patch.object(image_batch_app, "DATA_DIR", self.data_root)
        self.db_patch = patch.object(
            image_batch_app,
            "BATCH_DB_PATH",
            self.data_root / "batches.sqlite3",
        )
        self.data_patch.start()
        self.db_patch.start()
        image_batch_app.initialize_batch_database()
        self.client = TestClient(image_batch_app.app)

    def tearDown(self) -> None:
        self.client.close()
        self.db_patch.stop()
        self.data_patch.stop()
        self.temporary.cleanup()

    def test_folder_endpoint_creates_a_persistent_paginated_queue(self) -> None:
        source_dir = Path(self.temporary.name) / "pdfs"
        source_dir.mkdir()
        (source_dir / "one.pdf").write_bytes(b"%PDF-1.7\n")
        (source_dir / "two.pdf").write_bytes(b"%PDF-1.7\n")

        with patch.object(image_batch_app, "_start_batch") as start_batch:
            response = self.client.post(
                "/api/batches/folder",
                json={"folder": str(source_dir), "recursive": False, "concurrency": 2},
            )

        self.assertEqual(response.status_code, 200, response.text)
        payload = response.json()
        self.assertEqual(payload["total"], 2)
        self.assertEqual(payload["counts"]["queued"], 2)
        self.assertEqual(len(payload["items"]), 2)
        self.assertEqual(payload["source_kind"], "folder")
        self.assertEqual({item["source_group"] for item in payload["items"]}, {"根目录"})
        self.assertTrue(start_batch.called)

        paged = self.client.get(f"/api/batches/{payload['batch_id']}?offset=1&limit=1")
        self.assertEqual(paged.status_code, 200, paged.text)
        self.assertEqual(paged.json()["items_returned"], 1)
        self.assertEqual(paged.json()["items_total"], 2)

    def test_server_folder_browser_is_restricted_to_configured_roots(self) -> None:
        source_root = Path(self.temporary.name) / "pdfs"
        category = source_root / "group-a"
        category.mkdir(parents=True)
        (source_root / "root.pdf").write_bytes(b"%PDF-1.7\n")
        (category / "nested.pdf").write_bytes(b"%PDF-1.7\n")
        outside = Path(self.temporary.name) / "outside"
        outside.mkdir()
        (outside / "private.pdf").write_bytes(b"%PDF-1.7\n")

        with patch.object(
            image_batch_app,
            "PDF_SOURCE_ROOTS",
            (source_root.resolve(),),
        ):
            root_response = self.client.get("/api/server-folders")
            nested_response = self.client.get(
                "/api/server-folders",
                params={"path": str(category)},
            )
            forbidden_browser = self.client.get(
                "/api/server-folders",
                params={"path": str(outside)},
            )
            with patch.object(image_batch_app, "_start_batch") as start_batch:
                forbidden_batch = self.client.post(
                    "/api/batches/folder",
                    json={"folder": str(outside), "recursive": False, "concurrency": 1},
                )

        self.assertEqual(root_response.status_code, 200, root_response.text)
        root_payload = root_response.json()
        self.assertTrue(root_payload["configured"])
        self.assertEqual(root_payload["current"], str(source_root.resolve()))
        self.assertEqual(root_payload["pdf_count"], 1)
        self.assertEqual(root_payload["directories"][0]["name"], "group-a")
        self.assertEqual(root_payload["directories"][0]["pdf_count"], 1)
        self.assertEqual(nested_response.status_code, 200, nested_response.text)
        self.assertEqual(nested_response.json()["parent"], str(source_root.resolve()))
        self.assertEqual(forbidden_browser.status_code, 403)
        self.assertEqual(forbidden_batch.status_code, 403)
        start_batch.assert_not_called()

    def test_recursive_folder_batch_records_relative_directory_groups(self) -> None:
        source_root = Path(self.temporary.name) / "pdfs"
        first_group = source_root / "experiment-a"
        second_group = source_root / "experiment-b" / "day-1"
        first_group.mkdir(parents=True)
        second_group.mkdir(parents=True)
        (first_group / "one.pdf").write_bytes(b"%PDF-1.7\n")
        (second_group / "two.pdf").write_bytes(b"%PDF-1.7\n")

        with patch.object(
            image_batch_app,
            "PDF_SOURCE_ROOTS",
            (source_root.resolve(),),
        ), patch.object(image_batch_app, "_start_batch"):
            response = self.client.post(
                "/api/batches/folder",
                json={"folder": str(source_root), "recursive": True, "concurrency": 2},
            )

        self.assertEqual(response.status_code, 200, response.text)
        items = {item["filename"]: item for item in response.json()["items"]}
        self.assertEqual(items["one.pdf"]["source_group"], "experiment-a")
        self.assertEqual(items["one.pdf"]["source_relative_path"], "experiment-a/one.pdf")
        self.assertEqual(items["two.pdf"]["source_group"], "experiment-b/day-1")
        self.assertEqual(
            items["two.pdf"]["source_relative_path"],
            "experiment-b/day-1/two.pdf",
        )

    def test_batch_worker_writes_result_and_metrics(self) -> None:
        source = Path(self.temporary.name) / "paper.pdf"
        source.write_bytes(b"%PDF-1.7\n")
        job_id = "a" * 32
        state = image_batch_app.create_batch_state(
            [
                {
                    "item_id": "i000001",
                    "job_id": job_id,
                    "filename": source.name,
                    "source_path": str(source),
                    "size_bytes": source.stat().st_size,
                }
            ],
            concurrency=1,
            source_kind="folder",
            source_label=str(source.parent),
        )

        result = {
            "job_id": job_id,
            "message": "处理完成",
            "metrics": {"figures": 2, "panels": 5, "needs_review": 1},
            "records": [],
            "documents": [],
            "artifacts": [],
        }
        with patch.object(image_batch_app, "run_pipeline", return_value=result):
            image_batch_app._execute_batch(state["batch_id"], ["i000001"], 1)

        completed = image_batch_app.load_batch_state(state["batch_id"])
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(completed["completed_count"], 1)
        self.assertEqual(completed["items"][0]["metrics"]["panels"], 5)
        response = self.client.get(f"/api/jobs/{job_id}/result")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json()["metrics"]["figures"], 2)

    def test_batch_history_can_include_manual_review_progress(self) -> None:
        job_id = "e" * 32
        job_root = self.data_root / job_id
        source_path = job_root / "upload" / "whole.png"
        panel_dir = job_root / "results" / "panels" / "figure_1"
        source_path.parent.mkdir(parents=True)
        panel_dir.mkdir(parents=True)
        Image.new("RGB", (80, 60), "white").save(source_path)
        preview_path = panel_dir / "preview.png"
        Image.new("RGB", (80, 60), "white").save(preview_path)

        state = image_batch_app.create_batch_state(
            [
                {
                    "item_id": "i000001",
                    "job_id": job_id,
                    "filename": "reviewed.pdf",
                    "source_path": str(source_path),
                    "size_bytes": 1,
                }
            ],
            concurrency=1,
            source_kind="upload",
            source_label="review batch",
        )
        image_batch_app.initialize_editor_state(
            job_id,
            [
                {
                    "source": str(source_path),
                    "preview": str(preview_path),
                    "labels": "",
                    "detected_labels": "",
                    "figure_constraints": {"expected_labels": []},
                }
            ],
        )
        editor_state_path = job_root / "editor_state.json"
        editor_state = json.loads(editor_state_path.read_text(encoding="utf-8"))
        editor_state["figures"]["r001"].update(
            {"review_status": "verified", "annotation_complete": True}
        )
        image_batch_app.write_json_atomic(editor_state_path, editor_state)

        plain = self.client.get(f"/api/batches/{state['batch_id']}")
        reviewed = self.client.get(
            f"/api/batches/{state['batch_id']}?include_review=true"
        )
        self.assertEqual(plain.status_code, 200, plain.text)
        self.assertNotIn("review_state", plain.json()["items"][0])
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        item = reviewed.json()["items"][0]
        self.assertEqual(item["review_state"], "verified")
        self.assertEqual(item["verified_count"], 1)
        self.assertEqual(item["figure_count"], 1)

    def test_interrupted_processing_is_requeued_on_restart(self) -> None:
        source = Path(self.temporary.name) / "interrupted.pdf"
        source.write_bytes(b"%PDF-1.7\n")
        state = image_batch_app.create_batch_state(
            [
                {
                    "item_id": "i000001",
                    "job_id": "c" * 32,
                    "filename": source.name,
                    "source_path": str(source),
                    "size_bytes": source.stat().st_size,
                }
            ],
            concurrency=1,
            source_kind="folder",
            source_label=str(source.parent),
        )
        with image_batch_app._batch_connection() as connection:
            connection.execute(
                "UPDATE batches SET status = 'running' WHERE batch_id = ?",
                (state["batch_id"],),
            )
            connection.execute(
                "UPDATE batch_items SET status = 'processing', started_at = ? WHERE batch_id = ?",
                (image_batch_app.utc_now(), state["batch_id"]),
            )

        with patch.object(image_batch_app, "_start_batch") as start_batch:
            recovered_count = image_batch_app.recover_interrupted_batches()

        recovered = image_batch_app.load_batch_state(state["batch_id"])
        self.assertEqual(recovered_count, 1)
        self.assertEqual(recovered["status"], "queued")
        self.assertEqual(recovered["counts"]["queued"], 1)
        self.assertIsNone(recovered["items"][0]["started_at"])
        self.assertTrue(start_batch.called)

    def test_original_pdf_and_highlighted_source_page_are_available(self) -> None:
        job_id = "f" * 32
        job_root = self.data_root / job_id
        upload_dir = job_root / "upload"
        figure_dir = job_root / "results" / "extracted" / "paper" / "figures"
        panel_dir = job_root / "results" / "panels" / "figure_1"
        upload_dir.mkdir(parents=True)
        figure_dir.mkdir(parents=True)
        panel_dir.mkdir(parents=True)

        source_pdf = upload_dir / "paper.pdf"
        document = fitz.open()
        page = document.new_page(width=200, height=300)
        page.draw_rect(fitz.Rect(20, 30, 180, 200), color=(0, 0, 0), width=1)
        document.save(source_pdf)
        document.close()

        figure_path = figure_dir / "figure_1.png"
        preview_path = panel_dir / "preview.png"
        Image.new("RGB", (320, 340), "white").save(figure_path, dpi=(300, 300))
        Image.new("RGB", (320, 340), "white").save(preview_path)
        image_batch_app.write_json_atomic(
            figure_dir / "figures_metadata.json",
            {
                "figures": [
                    {
                        "figure_id": "figure_1",
                        "page_number": 1,
                        "bbox_pt": [20, 30, 180, 200],
                        "image_path": str(figure_path),
                    }
                ]
            },
        )
        image_batch_app.initialize_editor_state(
            job_id,
            [
                {
                    "source": str(figure_path),
                    "preview": str(preview_path),
                    "figure_id": "figure_1",
                    "labels": "",
                    "detected_labels": "",
                    "figure_constraints": {"expected_labels": []},
                }
            ],
        )
        image_batch_app.create_batch_state(
            [
                {
                    "item_id": "i000001",
                    "job_id": job_id,
                    "filename": source_pdf.name,
                    "source_path": str(source_pdf),
                    "size_bytes": source_pdf.stat().st_size,
                }
            ],
            concurrency=1,
            source_kind="upload",
            source_label="PDF review",
        )
        image_batch_app.write_json_atomic(
            job_root / "job_result.json",
            {
                "job_id": job_id,
                "message": "处理完成",
                "metrics": {},
                "records": [{"record_id": "r001", "page_number": 1}],
                "documents": [],
                "artifacts": [],
            },
        )

        result = self.client.get(f"/api/jobs/{job_id}/result")
        self.assertEqual(result.status_code, 200, result.text)
        record = result.json()["records"][0]
        self.assertTrue(record["source_pdf_available"])
        self.assertEqual(record["source_pdf_url"], f"/api/jobs/{job_id}/source-pdf")
        self.assertIn("source-page-preview", record["source_page_preview_url"])

        pdf_response = self.client.get(f"/api/jobs/{job_id}/source-pdf")
        self.assertEqual(pdf_response.status_code, 200, pdf_response.text)
        self.assertEqual(pdf_response.headers["content-type"], "application/pdf")
        self.assertTrue(pdf_response.headers["content-disposition"].startswith("inline"))
        self.assertTrue(pdf_response.content.startswith(b"%PDF"))

        preview_response = self.client.get(
            f"/api/jobs/{job_id}/figures/r001/source-page-preview",
            params={"dpi": 120},
        )
        self.assertEqual(preview_response.status_code, 200, preview_response.text)
        self.assertEqual(preview_response.headers["content-type"], "image/png")
        with Image.open(BytesIO(preview_response.content)) as preview:
            self.assertEqual(preview.size, (334, 500))
            red, green, blue = preview.convert("RGB").getpixel((33, 100))
            self.assertGreater(red, 180)
            self.assertLess(green, 100)
            self.assertLess(blue, 100)

        invalid_dpi = self.client.get(
            f"/api/jobs/{job_id}/figures/r001/source-page-preview",
            params={"dpi": 300},
        )
        self.assertEqual(invalid_dpi.status_code, 400)

    def test_manual_crop_keeps_automatic_and_manual_versions(self) -> None:
        job_id = "b" * 32
        job_root = self.data_root / job_id
        source_path = job_root / "upload" / "whole.png"
        panel_dir = job_root / "results" / "panels" / "figure_1"
        source_path.parent.mkdir(parents=True)
        panel_dir.mkdir(parents=True)
        Image.new("RGB", (100, 80), "white").save(source_path, dpi=(300, 300))
        preview_path = panel_dir / "preview.png"
        Image.new("RGB", (100, 80), "white").save(preview_path)
        Image.new("RGB", (40, 40), "red").save(panel_dir / "panel_A.png")
        image_batch_app.initialize_editor_state(
            job_id,
            [
                {
                    "source": str(source_path),
                    "preview": str(preview_path),
                    "labels": "A",
                    "detected_labels": "A",
                    "figure_constraints": {"expected_labels": ["A"]},
                }
            ],
        )

        response = self.client.post(
            f"/api/jobs/{job_id}/figures/r001/panel-versions",
            json={
                "label": "A",
                "x0": 50,
                "y0": 0,
                "x1": 100,
                "y1": 40,
                "expected_current_version": 0,
                "confirmed": True,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual([entry["version"] for entry in response.json()["versions"]], [0, 1])
        manifest = json.loads(
            (panel_dir / "manual_versions" / "A" / "versions.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["current_version"], 1)

    def test_verified_figure_builds_unsplit_yolo_training_collection(self) -> None:
        job_id = "d" * 32
        job_root = self.data_root / job_id
        source_path = job_root / "upload" / "whole.png"
        panel_dir = job_root / "results" / "panels" / "figure_1"
        source_path.parent.mkdir(parents=True)
        panel_dir.mkdir(parents=True)
        Image.new("RGB", (100, 80), "white").save(source_path, dpi=(300, 300))
        preview_path = panel_dir / "preview.png"
        Image.new("RGB", (100, 80), "white").save(preview_path)
        image_batch_app.initialize_editor_state(
            job_id,
            [
                {
                    "source": str(source_path),
                    "preview": str(preview_path),
                    "labels": "",
                    "detected_labels": "",
                    "figure_constraints": {"expected_labels": []},
                }
            ],
        )
        crop = self.client.post(
            f"/api/jobs/{job_id}/figures/r001/panel-versions",
            json={
                "label": "A",
                "x0": 50,
                "y0": 0,
                "x1": 100,
                "y1": 40,
                "expected_current_version": None,
                "confirmed": True,
            },
        )
        self.assertEqual(crop.status_code, 200, crop.text)

        review = self.client.post(
            f"/api/jobs/{job_id}/figures/r001/review",
            json={"status": "verified", "reviewer": "tester", "confirmed": True},
        )
        self.assertEqual(review.status_code, 200, review.text)
        self.assertTrue(review.json()["ready_for_training"])

        rebuild = self.client.post("/api/training-collection/rebuild")
        self.assertEqual(rebuild.status_code, 200, rebuild.text)
        summary = rebuild.json()
        self.assertEqual(summary["samples"], 1)
        self.assertEqual(summary["ready_for_training"], 1)
        self.assertEqual(summary["verified_panels"], 1)

        collection_root = self.data_root / "training_collection"
        annotations = list((collection_root / "annotations").glob("*.json"))
        labels = list((collection_root / "ready" / "labels").glob("*.txt"))
        images = list((collection_root / "ready" / "images").glob("*.png"))
        self.assertEqual(len(annotations), 1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(images), 1)
        annotation = json.loads(annotations[0].read_text(encoding="utf-8"))
        self.assertEqual(annotation["bbox_format"], "xyxy")
        self.assertEqual(annotation["coordinate_space"], "original_figure_pixels")
        self.assertEqual(annotation["review_status"], "verified")
        self.assertTrue(annotation["annotation_complete"])
        self.assertEqual(annotation["panels"][0]["bbox_xyxy"], [50, 0, 100, 40])
        self.assertEqual(
            labels[0].read_text(encoding="utf-8"),
            "0 0.75000000 0.25000000 0.50000000 0.50000000\n",
        )

        deleted = self.client.request(
            "DELETE",
            f"/api/jobs/{job_id}/figures/r001/panels",
            json={
                "label": "A",
                "expected_current_version": 1,
                "confirmed": True,
            },
        )
        self.assertEqual(deleted.status_code, 200, deleted.text)
        self.assertEqual(deleted.json()["review_status"], "proposed")
        self.assertEqual(deleted.json()["panel_targets"], [])
        self.assertEqual(list((collection_root / "ready" / "labels").glob("*.txt")), [])
        self.assertEqual(list((collection_root / "ready" / "images").glob("*.png")), [])

    def test_verified_no_split_figure_builds_empty_yolo_label(self) -> None:
        job_id = "e" * 32
        job_root = self.data_root / job_id
        source_path = job_root / "upload" / "whole.png"
        panel_dir = job_root / "results" / "panels" / "figure_1"
        source_path.parent.mkdir(parents=True)
        panel_dir.mkdir(parents=True)
        Image.new("RGB", (100, 80), "white").save(source_path, dpi=(300, 300))
        preview_path = panel_dir / "preview.png"
        Image.new("RGB", (100, 80), "white").save(preview_path)
        image_batch_app.initialize_editor_state(
            job_id,
            [
                {
                    "source": str(source_path),
                    "preview": str(preview_path),
                    "labels": "",
                    "detected_labels": "",
                    "figure_constraints": {"expected_labels": []},
                }
            ],
        )

        review = self.client.post(
            f"/api/jobs/{job_id}/figures/r001/review",
            json={
                "status": "verified",
                "annotation_mode": "no_split",
                "reviewer": "tester",
                "confirmed": True,
            },
        )
        self.assertEqual(review.status_code, 200, review.text)
        self.assertEqual(review.json()["annotation_mode"], "no_split")
        self.assertTrue(review.json()["ready_for_training"])

        rebuild = self.client.post("/api/training-collection/rebuild")
        self.assertEqual(rebuild.status_code, 200, rebuild.text)
        summary = rebuild.json()
        self.assertEqual(summary["ready_for_training"], 1)
        self.assertEqual(summary["no_split"], 1)
        self.assertEqual(summary["verified_panels"], 0)

        collection_root = self.data_root / "training_collection"
        annotations = list((collection_root / "annotations").glob("*.json"))
        labels = list((collection_root / "ready" / "labels").glob("*.txt"))
        images = list((collection_root / "ready" / "images").glob("*.png"))
        self.assertEqual(len(annotations), 1)
        self.assertEqual(len(labels), 1)
        self.assertEqual(len(images), 1)
        annotation = json.loads(annotations[0].read_text(encoding="utf-8"))
        self.assertEqual(annotation["annotation_mode"], "no_split")
        self.assertEqual(annotation["panels"], [])
        self.assertEqual(annotation["human_panel_count"], 0)
        self.assertEqual(labels[0].read_text(encoding="utf-8"), "")


if __name__ == "__main__":
    unittest.main()
