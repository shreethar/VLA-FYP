import json
import tempfile
import unittest
from pathlib import Path

from train.stage4.merge_and_upload import (
    _prepare_output_directory,
    resolve_stage4_checkpoint,
)


def _make_checkpoint(path):
    adapter = path / "student_lora"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (path / "spatial_parameters.pt").write_bytes(b"spatial")
    return path


class MergeAndUploadTest(unittest.TestCase):
    def test_resolve_stage4_run_prefers_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            older = _make_checkpoint(run / "step_000100")
            _make_checkpoint(run / "step_000200")
            (run / "best_checkpoint.json").write_text(
                json.dumps({"checkpoint_path": str(older)}),
                encoding="utf-8",
            )

            selected, source = resolve_stage4_checkpoint(run)

            self.assertEqual(selected, older.resolve())
            self.assertEqual(source, "best_checkpoint.json")

    def test_resolve_stage4_run_falls_back_to_latest_complete_step(self):
        with tempfile.TemporaryDirectory() as temporary:
            run = Path(temporary) / "run"
            _make_checkpoint(run / "step_000100")
            latest = _make_checkpoint(run / "step_000200")

            selected, source = resolve_stage4_checkpoint(run)

            self.assertEqual(selected, latest.resolve())
            self.assertEqual(source, "latest_complete_step_fallback")

    def test_output_directory_must_be_empty(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "merged"
            output.mkdir()
            (output / "existing.txt").write_text(
                "do not overwrite", encoding="utf-8"
            )

            with self.assertRaisesRegex(FileExistsError, "not empty"):
                _prepare_output_directory(output)


if __name__ == "__main__":
    unittest.main()
