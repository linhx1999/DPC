"""Exercise shell argument forwarding and CLI defaults without model requests."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class LauncherBudgetTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        for name in ["exp_bird.sh", "scripts/run_gen_baseline.sh", "scripts/run_dpc_selection.sh", "scripts/run_eval_ex.sh"]:
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / name, target)
        self.data = self.root / "data.json"
        self.data.write_text("[]")
        self.candidates = self.root / "candidates.json"
        self.candidates.write_text("{}")
        self.db = self.root / "databases"
        self.db.mkdir()
        self.calls = self.root / "calls.jsonl"
        fake = self.root / "fake_python"
        fake.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "with open(os.environ['TEST_CALLS'], 'a') as f:\n"
            "    f.write(json.dumps(sys.argv[1:]) + '\\n')\n"
            "if sys.argv[1:4] == ['-m', 'baseline.provenance', 'start']:\n"
            "    print(str(Path(os.environ['TEST_CALLS']).parent / 'session'))\n"
        )
        fake.chmod(0o755)
        self.env = dict(os.environ)
        for key in ["MAX_TOKENS", "PRED_SQLS_PATH", "RUN_ID", "RUNS", "DPC_SESSION", "OUTPUT_DIR"]:
            self.env.pop(key, None)
        self.env.update({
            "PYTHON_BIN": str(fake), "TEST_CALLS": str(self.calls),
            "DATA_PATH": str(self.data), "DB_ROOT_PATH": str(self.db),
            "OUTPUT_DIR": str(self.root / "output"), "MODEL_NAME": "test-model",
        })

    def run_shell(self, script, **env):
        subprocess.run(["bash", script], cwd=self.root, env=self.env | env, check=True, capture_output=True, text=True)
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_bird_launcher_passes_shared_budget_and_allows_override(self):
        for override, expected in [({}, "16384"), ({"MAX_TOKENS": "32768"}, "32768")]:
            with self.subTest(expected=expected):
                self.calls.unlink(missing_ok=True)
                calls = self.run_shell("exp_bird.sh", **override)
                for entry in ["baseline/run_sql_generation.py", "baseline/run_dpc_selection.py"]:
                    call, = [c for c in calls if c[0] == entry]
                    self.assertEqual(call[call.index("--max_tokens") + 1], expected)

    def test_external_candidates_skip_generation_with_new_selection_budget(self):
        calls = self.run_shell("exp_bird.sh", PRED_SQLS_PATH=str(self.candidates))
        self.assertFalse(any(c[0] == "baseline/run_sql_generation.py" for c in calls))
        call, = [c for c in calls if c[0] == "baseline/run_dpc_selection.py"]
        self.assertEqual(call[call.index("--max_tokens") + 1], "16384")
        self.assertEqual(call[call.index("--pred_sqls_path") + 1], str(self.candidates))

    def test_standalone_selection_shell_defaults_to_16384(self):
        call, = self.run_shell("scripts/run_dpc_selection.sh")
        self.assertEqual(call[call.index("--max_tokens") + 1], "16384")

    def test_python_selection_default_is_saved_in_session_config(self):
        session = self.root / "session"
        session.mkdir()
        env = self.env | {"DPC_SESSION": str(session), "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([
            sys.executable, "baseline/run_dpc_selection.py", "--dataset_type", "bird",
            "--data_path", str(self.data), "--db_root_path", str(self.db),
            "--pred_sqls_path", str(self.candidates), "--output_path", str(self.root / "selected.json"),
        ], cwd=ROOT, env=env, check=True, capture_output=True, text=True)
        self.assertEqual(json.loads((session / "selection.json").read_text())["max_tokens"], 16384)


if __name__ == "__main__":
    unittest.main()
