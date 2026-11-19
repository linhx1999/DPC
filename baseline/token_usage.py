"""Persist model-reported token counts for a DPC experiment run."""
import argparse
import json
import os
from pathlib import Path
from uuid import uuid4

from baseline.common import save_json_atomic


TOKEN_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")


def usage_recorder(stage, question_id, model):
    output = os.environ.get("DPC_USAGE_DIR")
    if not output:
        return None

    def record(usage, status, **details):
        counts = {field: getattr(usage, field, None) for field in TOKEN_FIELDS}
        row = {"stage": stage, "question_id": str(question_id), "model": model,
               "session_id": Path(os.environ["DPC_SESSION"]).name if os.getenv("DPC_SESSION") else None,
               "status": status, **counts, **details}
        path = Path(output) / "telemetry" / "calls" / f"{stage}-{uuid4().hex}.json"
        save_json_atomic(path, row)

    return record


def summarize(output):
    def counters():
        return {"calls": 0, "failed_calls": 0, "calls_missing_usage": 0, "outcome_counts": {},
                **dict.fromkeys(TOKEN_FIELDS)}

    def add(stats, row):
        stats["calls"] += 1
        stats["failed_calls"] += row["status"] == "failed"
        stats["calls_missing_usage"] += any(row[k] is None for k in TOKEN_FIELDS)
        outcome = row.get("outcome") or "unclassified"
        stats["outcome_counts"][outcome] = stats["outcome_counts"].get(outcome, 0) + 1
        for field in TOKEN_FIELDS:
            if row[field] is not None:
                stats[field] = (stats[field] or 0) + row[field]

    total = counters()
    stages = {name: counters() for name in ("generation", "selection")}
    questions = {}
    for path in (output / "telemetry" / "calls").glob("*.json"):
        row = json.loads(path.read_text(encoding="utf-8"))
        stage = row["stage"]
        by_stage = questions.setdefault(row["question_id"], {})
        add(total, row)
        add(stages[stage], row)
        add(by_stage.setdefault(stage, counters()), row)

    for stats in [total, *stages.values()]:
        if not stats["calls"]:
            stats.update(dict.fromkeys(TOKEN_FIELDS, 0))

    save_json_atomic(output / "metrics" / "summary.json", {
        "scope": "Recorded calls in this run directory; token sums include only provider-reported values.",
        "total": total, "by_stage": stages, "by_question": questions,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
