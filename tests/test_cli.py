from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main as app


build_parser = app.build_parser
clean_input_path = app.clean_input_path
resolve_source_path = app.resolve_source_path
validate_args = app.validate_args


class CliValidationTests(unittest.TestCase):
    def parse(self, values: list[str]):
        parser = build_parser()
        return validate_args(parser, parser.parse_args(values))

    def test_webcam_options(self) -> None:
        args = self.parse(["--webcam", "--camera-index", "0"])
        self.assertTrue(args.webcam)
        self.assertEqual(args.camera_index, 0)

    def test_rejects_source_and_webcam_together(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse(["image.jpg", "--webcam"])

    def test_rejects_invalid_confidence(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse(["image.jpg", "--confidence", "1.1"])

    def test_does_not_offer_file_output_option(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse(["image.jpg", "--save-output"])

    def test_rejects_negative_camera_index(self) -> None:
        with self.assertRaises(SystemExit):
            self.parse(["--webcam", "--camera-index", "-1"])

    def test_accepts_custom_model(self) -> None:
        args = self.parse(["image.jpg", "--model", "model-khac.pt"])
        self.assertEqual(args.model, "model-khac.pt")

    def test_rejects_missing_source_before_loading_pipeline(self) -> None:
        with self.assertRaises(FileNotFoundError):
            resolve_source_path("file-khong-ton-tai.jpg")


class TemporarySourceFixture:
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_dir = Path(self.temp_dir.name)
        self.media_dir = self.base_dir / "BienSoXe"
        self.media_dir.mkdir()
        self.root_image = self.base_dir / "anh-goc.jpg"
        self.sample_image = self.media_dir / "xe_02.jpg"
        self.sample_video = self.media_dir / "video_1.mp4"
        self.root_image.touch()
        self.sample_image.touch()
        self.sample_video.touch()

        self.base_dir_patch = patch.object(app, "BASE_DIR", self.base_dir)
        self.base_dir_patch.start()

    def tearDown(self) -> None:
        self.base_dir_patch.stop()
        self.temp_dir.cleanup()


class SourceResolutionTests(TemporarySourceFixture, unittest.TestCase):
    def test_resolves_relative_path_from_base_dir(self) -> None:
        self.assertEqual(resolve_source_path("anh-goc.jpg"), self.root_image.resolve())

    def test_resolves_path_inside_biensoxe_from_base_dir(self) -> None:
        self.assertEqual(
            resolve_source_path("BienSoXe/xe_02.jpg"), self.sample_image.resolve()
        )

    def test_resolves_bare_filename_from_biensoxe(self) -> None:
        self.assertEqual(resolve_source_path("xe_02.jpg"), self.sample_image.resolve())

    def test_strips_single_and_double_quotes_from_path(self) -> None:
        for quoted_path in (f'"{self.sample_image}"', f"'{self.sample_image}'"):
            with self.subTest(quoted_path=quoted_path):
                self.assertEqual(
                    resolve_source_path(quoted_path), self.sample_image.resolve()
                )

    def test_accepts_powershell_call_operator_with_single_quotes(self) -> None:
        pasted_path = f"& '{self.sample_image}'"
        self.assertEqual(resolve_source_path(pasted_path), self.sample_image.resolve())

    def test_accepts_powershell_call_operator_with_double_quotes(self) -> None:
        pasted_path = f'& "{self.sample_video}"'
        self.assertEqual(resolve_source_path(pasted_path), self.sample_video.resolve())

    def test_rejects_empty_path(self) -> None:
        for value in ("", "   ", "&", "&   "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    clean_input_path(value)


class InteractiveMenuTests(TemporarySourceFixture, unittest.TestCase):
    def test_accepts_image_path(self) -> None:
        with patch("builtins.input", return_value="xe_02.jpg"):
            self.assertEqual(
                app.ask_for_source(), (str(self.sample_image.resolve()), False)
            )

    def test_menu_accepts_powershell_style_image_path(self) -> None:
        pasted_path = f"& '{self.sample_image}'"
        with patch("builtins.input", return_value=pasted_path):
            self.assertEqual(
                app.ask_for_source(), (str(self.sample_image.resolve()), False)
            )

    def test_accepts_video_path(self) -> None:
        with patch("builtins.input", return_value="video_1.mp4"):
            self.assertEqual(
                app.ask_for_source(), (str(self.sample_video.resolve()), False)
            )

    def test_cam_selects_webcam(self) -> None:
        with patch("builtins.input", return_value="cam"):
            self.assertEqual(app.ask_for_source(), (None, True))

    def test_q_exits_cleanly(self) -> None:
        with patch("builtins.input", return_value="q"):
            with self.assertRaises(SystemExit) as raised:
                app.ask_for_source()
        self.assertEqual(raised.exception.code, 0)

    def test_invalid_path_then_accepts_image(self) -> None:
        with patch(
            "builtins.input", side_effect=["khong-ton-tai.jpg", "xe_02.jpg"]
        ):
            with patch("builtins.print") as mocked_print:
                self.assertEqual(
                    app.ask_for_source(), (str(self.sample_image.resolve()), False)
                )

        printed_text = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in mocked_print.call_args_list
        )
        self.assertIn("Không tìm thấy file", printed_text)

    def test_invalid_path_then_accepts_video(self) -> None:
        with patch(
            "builtins.input", side_effect=["khong-ton-tai.mp4", "video_1.mp4"]
        ):
            with patch("builtins.print") as mocked_print:
                self.assertEqual(
                    app.ask_for_source(), (str(self.sample_video.resolve()), False)
                )

        printed_text = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in mocked_print.call_args_list
        )
        self.assertIn("Không tìm thấy file", printed_text)

    def test_camera_word_selects_webcam(self) -> None:
        with patch("builtins.input", return_value="camera"):
            self.assertEqual(app.ask_for_source(), (None, True))

    def test_bad_input_then_cam_continues(self) -> None:
        with patch("builtins.input", side_effect=["main.py", "cam"]):
            with patch("builtins.print") as mocked_print:
                self.assertEqual(app.ask_for_source(), (None, True))

        printed_text = "\n".join(
            " ".join(str(value) for value in call.args)
            for call in mocked_print.call_args_list
        )
        self.assertIn("Chương trình chỉ nhận file ảnh hoặc video", printed_text)

    def test_main_reuses_one_pipeline_for_multiple_menu_choices(self) -> None:
        fake_model = self.base_dir / "best.pt"
        fake_model.touch()

        class FakePipeline:
            instances = 0
            processed: list[str] = []

            def __init__(self, model_path, confidence):
                self.model_path = model_path
                self.confidence = confidence
                FakePipeline.instances += 1

            def process(self, source, preview, ocr_interval):
                FakePipeline.processed.append(str(source))
                return {
                    "kind": "image",
                    "frames_processed": 1,
                    "detections_total": 0,
                    "recognized_texts": [],
                    "new_plates": [],
                    "duplicate_plates": [],
                }

            def process_webcam(self, camera_index, preview, ocr_interval):
                raise AssertionError("Không dùng webcam trong test này")

        with (
            patch.object(app, "MODEL_PATH", fake_model),
            patch.object(app.sys, "argv", ["main.py"]),
            patch.object(
                app,
                "ask_for_source",
                side_effect=[
                    (str(self.root_image), False),
                    (str(self.sample_video), False),
                    SystemExit(0),
                ],
            ),
            patch("plate_recognition.RecognitionPipeline", FakePipeline),
            patch("builtins.print"),
        ):
            with self.assertRaises(SystemExit) as raised:
                app.main()

        self.assertEqual(raised.exception.code, 0)
        self.assertEqual(FakePipeline.instances, 1)
        self.assertEqual(len(FakePipeline.processed), 2)


class ProjectPythonBootstrapTests(unittest.TestCase):
    def test_does_not_relaunch_when_already_using_project_venv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_python = Path(temp_dir) / "python.exe"
            fake_python.touch()
            with (
                patch.object(app.sys, "platform", "win32"),
                patch.object(app.sys, "executable", str(fake_python)),
                patch.object(app, "VENV_PYTHON", fake_python),
                patch.object(app.subprocess, "run") as mocked_run,
            ):
                app.ensure_project_python()

        mocked_run.assert_not_called()

    def test_relaunches_with_project_venv_and_forwards_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_dir = Path(temp_dir)
            current_python = base_dir / "global-python.exe"
            project_python = base_dir / ".venv-train" / "Scripts" / "python.exe"
            current_python.touch()
            project_python.parent.mkdir(parents=True)
            project_python.touch()
            completed = subprocess.CompletedProcess([], 7)

            with (
                patch.object(app.sys, "platform", "win32"),
                patch.object(app.sys, "executable", str(current_python)),
                patch.object(app.sys, "argv", ["main.py", "--help"]),
                patch.object(app, "BASE_DIR", base_dir),
                patch.object(app, "VENV_PYTHON", project_python),
                patch.object(
                    app.subprocess, "run", return_value=completed
                ) as mocked_run,
            ):
                with self.assertRaises(SystemExit) as raised:
                    app.ensure_project_python()

            self.assertEqual(raised.exception.code, 7)
            mocked_run.assert_called_once_with(
                [
                    str(project_python),
                    str(Path(app.__file__).resolve()),
                    "--help",
                ],
                cwd=str(base_dir),
                check=False,
            )


if __name__ == "__main__":
    unittest.main()
