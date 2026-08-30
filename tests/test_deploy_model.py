from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import deploy_model


class DeployModelTests(unittest.TestCase):
    def test_validation_failure_does_not_replace_current_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            candidate = root / "candidate.pt"
            destination = root / "models" / "best.pt"
            evaluation = root / "evaluation.json"
            candidate.write_bytes(b"new-model")
            destination.parent.mkdir(parents=True)
            destination.write_bytes(b"old-model")
            evaluation.write_text(
                json.dumps(
                    {
                        "model_sha256": hashlib.sha256(b"new-model").hexdigest(),
                        "selected": {"threshold": 0.85},
                    }
                ),
                encoding="utf-8",
            )

            argv = [
                "deploy_model.py",
                "--model",
                str(candidate),
                "--evaluation",
                str(evaluation),
                "--destination",
                str(destination),
            ]
            with (
                patch.object(sys, "argv", argv),
                patch.object(
                    deploy_model,
                    "split_metadata",
                    side_effect=RuntimeError("invalid split"),
                ),
            ):
                with self.assertRaises(RuntimeError):
                    deploy_model.main()

            self.assertEqual(destination.read_bytes(), b"old-model")


if __name__ == "__main__":
    unittest.main()
