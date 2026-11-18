import os
import sys
import json
import argparse
import logging
import multiprocessing
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from typing import List, Dict, Any, Tuple, Optional

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from baseline.provenance import candidates_digest, matches_dataset, read_question, record_config
from evaluation.question_report import write_report
from dpc.datasets.spider_loader import SpiderLoader
from dpc.datasets.bird_loader import BirdLoader
from dpc.utils.db_utils import execute_sql as execute_sql_rows

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Eval-EX")

def execute_checked(sql, db_path, timeout):
    if not sql:
        return None, "missing", None
    try:
        return set(execute_sql_rows(db_path, sql, timeout=timeout)), "ok", None
    except TimeoutError as exc:
        return None, "timeout", str(exc)
    except Exception as exc:
        return None, "error", str(exc)


def execute_sql(sql: str, db_path: str, timeout: int = 30) -> Optional[set]:
    return execute_checked(sql, db_path, timeout)[0]


def process_sample_ex(task):
    db_path, timeout = task["db_path"], task["sql_timeout"]
    pred, ps, pe = execute_checked(task["pred_sql"], db_path, timeout)
    gold, gs, ge = execute_checked(task["gold_sql"], db_path, timeout)
    champion_sql = (task.get("selection") or {}).get("champion_sql")
    if champion_sql and champion_sql == task["pred_sql"]:
        champion, cs, ce = pred, ps, pe
    else:
        champion, cs, ce = execute_checked(champion_sql, db_path, timeout)
    executable_count = (task.get("selection") or {}).get("executable_count")
    if executable_count is None and task.get("unchecked_candidates") is not None:
        executable_count = sum(execute_checked(sql, db_path, timeout)[1] == "ok"
                               for sql in task["unchecked_candidates"])
    return {"qid": task["qid"], "difficulty": task["difficulty"], "executable_count": executable_count,
            "correct": ps == gs == "ok" and pred == gold,
            "champion_correct": (cs == gs == "ok" and champion == gold) if champion_sql else None,
            "prediction_status": ps, "prediction_error": pe, "gold_status": gs, "gold_error": ge,
            "champion_status": cs, "champion_error": ce}


def main():
    parser = argparse.ArgumentParser(description="Evaluate SQL execution accuracy.")
    parser.add_argument("--pred_path", type=str, required=True, help="Path to predicted SQLs JSON file.")
    parser.add_argument("--dataset_type", type=str, choices=["spider", "bird"], required=True, help="Type of dataset.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to dataset file.")
    parser.add_argument("--db_root_path", type=str, required=True, help="Path to database root directory.")
    parser.add_argument("--timeout", type=int, default=30, help="Timeout for SQL execution.")
    parser.add_argument("--num_workers", type=int, default=min(multiprocessing.cpu_count(), 8), help="Number of parallel workers.")
    
    args = parser.parse_args()
    session = os.getenv("DPC_SESSION")
    record_config("evaluation", args)

    # 1. Load Dataset
    if args.dataset_type.lower() == "spider":
        loader = SpiderLoader(args.data_path, args.db_root_path)
    elif args.dataset_type.lower() == "bird":
        loader = BirdLoader(args.data_path, args.db_root_path)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    # 2. Load Predicted SQLs
    if not os.path.exists(args.pred_path):
        raise FileNotFoundError(f"Predicted SQLs file not found: {args.pred_path}")
    
    with open(args.pred_path, 'r', encoding='utf-8') as f:
        pred_map = json.load(f)
    
    # Standardized format: {"qid": "sql"}
    # Ensure all keys are strings
    pred_map = {str(k): v for k, v in pred_map.items()}

    candidates = {}
    if session:
        config = json.loads((Path(session) / "selection.json").read_text())
        candidates = json.loads(Path(config["pred_sqls_path"]).read_text())

    # 3. Prepare Tasks
    tasks = []
    for i in range(len(loader)):
        item = loader.get_item(i)
        qid = str(item.question_id)
        
        if not session and qid not in pred_map:
            continue
            
        selection = read_question("selection", qid) if session else None
        trace_status = "missing"
        if selection:
            same_inputs = selection.get("candidates_sha256") == candidates_digest(candidates.get(qid))
            same_prediction = qid in pred_map and selection.get("selected_sql") == pred_map[qid]
            if same_inputs and same_prediction and matches_dataset(selection, "selection", session):
                trace_status = "matched"
            else:
                trace_status = "mismatch"
                selection = None
        tasks.append({
            "qid": qid,
            "pred_sql": pred_map.get(qid),
            "db_id": item.db_id,
            "prediction_present": qid in pred_map,
            "selection": selection,
            "unchecked_candidates": candidates.get(qid) if (selection or {}).get("executable_count") is None else None,
            "selection_trace_status": trace_status,
            "gold_sql": item.ground_truth,
            "db_path": loader.get_db_path(item.db_id),
            "sql_timeout": args.timeout,
            "difficulty": item.difficulty
        })

    # 4. Parallel Evaluation
    total = len(tasks)
    correct = 0
    evaluated = {}
    diff_stats = {} # {difficulty: {"correct": 0, "total": 0}}
    
    logger.info(f"Starting parallel execution evaluation for {total} samples...")
    
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {executor.submit(process_sample_ex, task): task for task in tasks}
        for future in tqdm(as_completed(futures), total=total, desc="Evaluating EX"):
            res = future.result()
            if session:
                evaluated[res["qid"]] = res
            diff = res["difficulty"]
            if diff not in diff_stats:
                diff_stats[diff] = {"correct": 0, "total": 0}
            
            diff_stats[diff]["total"] += 1
            if res["correct"]:
                correct += 1
                diff_stats[diff]["correct"] += 1

    if session:
        write_report(Path(os.environ["DPC_USAGE_DIR"]), Path(session), tasks, evaluated, candidates)

    # 5. Report
    accuracy = (correct / total) * 100 if total > 0 else 0
    logger.info("=" * 30)
    logger.info(f"Overall Execution Accuracy: {accuracy:.2f}%")
    logger.info(f"Total Evaluated: {total}")
    logger.info(f"Correct: {correct}")
    
    if diff_stats:
        logger.info("-" * 20)
        logger.info("Difficulty Breakdown:")
        # Sort difficulties for consistent output
        for diff in sorted(diff_stats.keys()):
            c = diff_stats[diff]["correct"]
            t = diff_stats[diff]["total"]
            acc = (c / t) * 100 if t > 0 else 0
            logger.info(f"  {diff:12}: {acc:6.2f}% ({c}/{t})")
            
    logger.info("=" * 30)

if __name__ == "__main__":
    main()
