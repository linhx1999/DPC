"""Run manifests and per-question checkpoints for the experiment launcher."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from baseline.common import save_json_atomic


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def candidates_digest(candidates):
    payload = json.dumps(candidates, ensure_ascii=False, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def record_config(stage, args):
    session = os.getenv("DPC_SESSION")
    if not session:
        return
    config = vars(args).copy()
    config.pop("api_key", None)
    if "base_url" in config:
        endpoint = urlsplit(config["base_url"] or os.getenv("OPENAI_BASE_URL", ""))
        config["base_url"] = endpoint._replace(netloc=endpoint.netloc.split("@")[-1],
                                               query="", fragment="").geturl()
    for key in ("data_path", "pred_sqls_path", "pred_path"):
        if key in config and Path(config[key]).is_file():
            config[key + "_sha256"] = digest(config[key])
    save_json_atomic(Path(session) / (stage + ".json"), config)


def question_path(stage, qid):
    key = hashlib.sha256(str(qid).encode()).hexdigest()
    return Path(os.environ["DPC_USAGE_DIR"]) / "telemetry" / stage / (key + ".json")


def record_question(stage, qid, started, **fields):
    session = os.getenv("DPC_SESSION")
    if session:
        save_json_atomic(question_path(stage, qid), {
            "question_id": str(qid), "session_id": Path(session).name,
            "seconds": time.monotonic() - started, **fields,
        })


@lru_cache
def dataset_identity(session, stage):
    config = json.loads((Path(session) / (stage + ".json")).read_text())
    return config.get("data_path_sha256"), config["db_root_path"]


def matches_dataset(record, stage, session):
    original = Path(session).parent / record["session_id"]
    return dataset_identity(original, stage) == dataset_identity(session, "evaluation")


def read_question(stage, qid):
    path = question_path(stage, qid)
    return json.loads(path.read_text()) if path.is_file() else None


def git(*args):
    result = subprocess.run(["git", *args], capture_output=True, text=True)
    return result.stdout.rstrip() if result.returncode == 0 else None


def start(output, run_id, split, external):
    session = output / "sessions" / uuid4().hex
    source_files = [Path("exp_bird.sh"), Path("uv.lock"), Path("pyproject.toml")]
    for directory in ("baseline", "dpc", "evaluation", "scripts"):
        source_files.extend(p for p in Path(directory).rglob("*") if p.suffix in {".py", ".sh"})
    save_json_atomic(session / "manifest.json", {
        "session_id": session.name, "run_id": run_id, "split": split,
        "started_at": utc_now(), "started_monotonic": time.monotonic(), "status": "running",
        "candidate_source": "external" if external else "generation",
        "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
        "source_sha256": {str(p): digest(p) for p in sorted(source_files)},
        "python": platform.python_version(), "openai_client": version("openai"),
        "model_revision": os.getenv("MODEL_REVISION"), "server_version": os.getenv("MODEL_SERVER_VERSION"),
    })
    print(session)


def finish(session, code):
    path = session / "manifest.json"
    data = json.loads(path.read_text())
    data.update(finished_at=utc_now(), seconds=time.monotonic() - data.pop("started_monotonic"),
                exit_code=code, status="complete" if code == 0 else "failed")
    save_json_atomic(path, data)
    sessions = [json.loads(p.read_text()) for p in session.parent.glob("*/manifest.json")]
    finished = [s for s in sessions if "seconds" in s]
    save_json_atomic(session.parent.parent / "metrics" / "timing.json", {
        "last_session": session.name, "last_session_seconds": data["seconds"],
        "recorded_wall_seconds": sum(s["seconds"] for s in finished),
        "finished_sessions": len(finished), "unfinished_sessions": len(sessions) - len(finished),
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    begin = actions.add_parser("start")
    begin.add_argument("output", type=Path)
    begin.add_argument("run_id")
    begin.add_argument("split")
    begin.add_argument("--external", action="store_true")
    end = actions.add_parser("finish")
    end.add_argument("session", type=Path)
    end.add_argument("code", type=int)
    args = parser.parse_args()
    if args.action == "start":
        start(args.output, args.run_id, args.split, args.external)
    else:
        finish(args.session, args.code)
