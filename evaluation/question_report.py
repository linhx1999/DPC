"""Write a dataset-ordered EX report with paired initial/final decisions."""
from collections import Counter
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

from baseline.common import save_json_atomic
from baseline.provenance import candidates_digest, matches_dataset, read_question


def write_report(output, session, tasks, evaluated, candidates):
    rows = []
    outcomes = Counter()
    seconds = {"generation": [], "selection": [], "duel_selection": []}
    selection_tokens = {"all": [], "duel": []}
    status_counts = {key: Counter() for key in ("prediction_status", "gold_status", "champion_status")}
    usage = json.loads((output / "metrics/summary.json").read_text())["by_question"]
    for task in tasks:
        qid = task["qid"]
        result = evaluated[qid]
        selection = task["selection"]
        generation = read_question("generation", qid)
        if generation and (generation.get("candidates_sha256") != candidates_digest(candidates.get(qid))
                           or not matches_dataset(generation, "generation", session)):
            generation = None
        tokens = usage.get(qid, {})
        row = {"question_id": qid, "db_id": task["db_id"], "difficulty": task["difficulty"],
               "prediction_present": task["prediction_present"], "candidate_count": None,
               "selected_sql": task["pred_sql"], "generation": generation, "selection": selection,
               "selection_trace_status": task["selection_trace_status"],
               "tokens": tokens, **{k: v for k, v in result.items() if k not in ("qid", "difficulty")}}
        if qid in candidates:
            row["candidate_count"] = len(candidates[qid])
        if generation:
            seconds["generation"].append(generation["seconds"])
        if selection:
            seconds["selection"].append(selection["seconds"])
            selection_usage = tokens.get("selection", {})
            if not selection_usage.get("calls_missing_usage"):
                cost = selection_usage.get("total_tokens", 0)
                selection_tokens["all"].append(cost)
                if selection.get("duel_triggered"):
                    selection_tokens["duel"].append(cost)
            if selection.get("duel_triggered"):
                seconds["duel_selection"].append(selection["seconds"])
        for key, counts in status_counts.items():
            counts[result[key]] += 1
        if result["champion_correct"] is not None and result["gold_status"] == "ok":
            if result["champion_correct"] == result["correct"]:
                outcomes["both_correct" if result["correct"] else "both_wrong"] += 1
            else:
                outcomes["fixed" if result["correct"] else "regressed"] += 1
        else:
            outcomes["unpaired"] += 1
        rows.append(row)

    def mean(values):
        return {"n": len(values), "mean": sum(values) / len(values) if values else None}

    report = {
        "expected": len(rows), "prediction_keys_present": sum(r["prediction_present"] for r in rows),
        "scored": len(rows),
        "execution_pairs_succeeded": sum(r["prediction_status"] == r["gold_status"] == "ok" for r in rows),
        "missing_prediction_keys": sum(not r["prediction_present"] for r in rows),
        "nonempty_predictions": sum(bool(r["selected_sql"]) for r in rows),
        "correct": sum(r["correct"] for r in rows),
        "accuracy_percent": 100 * sum(r["correct"] for r in rows) / len(rows) if rows else None,
        "execution_status_counts": status_counts,
        "paired": {k: outcomes[k] for k in ("both_correct", "both_wrong", "fixed", "regressed", "unpaired")},
        "net_correct_on_paired": outcomes["fixed"] - outcomes["regressed"],
        "seconds_per_question": {k: mean(v) for k, v in seconds.items()},
        "selection_tokens_per_question": {k: mean(v) for k, v in selection_tokens.items()},
        "evaluation_session": session.name,
    }
    # Keep the original evaluation when a later session re-evaluates a run.
    for directory in (session, output):
        with NamedTemporaryFile("w", dir=directory, delete=False, encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        Path(stream.name).replace(directory / "per_question.jsonl")
        save_json_atomic(directory / "evaluation_summary.json", report)
