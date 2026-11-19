#!/usr/bin/env bash
set -euo pipefail

# 参数初始化。相对路径均以 DPC 项目根目录为基准。
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd -- "$REPO_DIR"

# BIRD 数据集根目录，包含 dev/ 和 minidev/。
export DATA_ROOT="${DATA_ROOT:-../datasets/bird}"
SPLIT="${SPLIT:-minidev}"

export MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:8001/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-EMPTY}"
export NUM_WORKERS="${WORKERS:-${NUM_WORKERS:-8}}"

# 论文 §4.1：DPC 采样温度 0.7，自纠错上限 3 次。
# 每题 5 个候选；生成和选择统一使用 16384 输出 tokens，供推理模型使用。
GEN_TEMPERATURE="${GEN_TEMPERATURE:-0.6}"
GEN_MAX_TOKENS="${MAX_TOKENS:-16384}"
DPC_TEMPERATURE="${DPC_TEMPERATURE:-0.6}"
export MAX_TOKENS="${MAX_TOKENS:-16384}"
export MAX_CORRECTION_ATTEMPTS="${MAX_CORRECTION_ATTEMPTS:-3}"

RUN_ID="${RUN_ID:-run1}"
RUNS="${RUNS:-1}"
OUTPUT_BASE="${OUTPUT_DIR:-}"

# 参数拼接。其余 DPC 参数通过 scripts/run_dpc_selection.sh 的环境变量设置。
while (($#)); do
    case "$1" in
        --split) SPLIT="${2:?--split requires a value}"; shift 2 ;;
        --split=*) SPLIT="${1#*=}"; shift ;;
        --runs) RUNS="${2:?--runs requires a value}"; shift 2 ;;
        --runs=*) RUNS="${1#*=}"; shift ;;
        --output_dir) OUTPUT_BASE="${2:?--output_dir requires a value}"; shift 2 ;;
        --output_dir=*) OUTPUT_BASE="${1#*=}"; shift ;;
        -h|--help)
            echo "Usage: ./exp_bird.sh [--split dev|minidev] [--runs N] [--output_dir DIR]"
            exit 0 ;;
        *) echo "Unknown option: $1 (see --help)" >&2; exit 2 ;;
    esac
done
case "$SPLIT" in
    dev) split_dir="$DATA_ROOT/dev"; questions=dev.json ;;
    minidev) split_dir="$DATA_ROOT/minidev"; questions=mini_dev_sqlite.json ;;
    *) echo "Unsupported BIRD split: $SPLIT" >&2; exit 2 ;;
esac
export DATASET_TYPE=bird
export DATA_PATH="${DATA_PATH:-$split_dir/$questions}"
export DB_ROOT_PATH="${DB_ROOT_PATH:-$split_dir/dev_databases}"
[[ -r "$DATA_PATH" ]] || { echo "Missing BIRD questions: $DATA_PATH" >&2; exit 1; }
[[ -d "$DB_ROOT_PATH" ]] || { echo "Missing BIRD database directory: $DB_ROOT_PATH" >&2; exit 1; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "RUNS must be a positive integer" >&2; exit 2; }
if ((RUNS > 1)); then
    if [[ "$RUN_ID" =~ ^(.*[^0-9])?([0-9]+)$ ]]; then
        prefix="${BASH_REMATCH[1]}"
        start=$((10#${BASH_REMATCH[2]}))
    else
        echo "Multiple runs require a RUN_ID ending in a number" >&2
        exit 2
    fi
fi

default_python=python3
if [[ -x .venv/bin/python ]]; then default_python=.venv/bin/python; fi

# 顺序生成候选、运行 DPC 选择并评测 EX。生成和选择按题目 ID 续跑。
for ((i=0; i<RUNS; i++)); do
    current_id="$RUN_ID"
    if ((RUNS > 1)); then current_id="$prefix$((start + i))"; fi
    run_dir="${OUTPUT_BASE:-artifacts/dpc_bird_${SPLIT}_${MODEL_NAME//\//_}_${current_id}}"
    if [[ -n "$OUTPUT_BASE" ]] && ((RUNS > 1)); then run_dir="$OUTPUT_BASE/$current_id"; fi
    mkdir -p -- "$run_dir"
    exec {run_lock}>"$run_dir/.run.lock"
    flock "$run_lock"
    echo "Run $((i + 1))/$RUNS: $current_id -> $run_dir" >&2
    export DPC_USAGE_DIR="$run_dir"
    python_bin="${PYTHON_BIN:-$default_python}"
    source_args=()
    if [[ -n "${PRED_SQLS_PATH:-}" ]]; then source_args+=(--external); fi
    DPC_SESSION=$("$python_bin" -m baseline.provenance start "$run_dir" "$current_id" "$SPLIT" "${source_args[@]}")
    export DPC_SESSION
    trap '"$python_bin" -m baseline.provenance finish "$DPC_SESSION" "$?"' EXIT
    candidates="${PRED_SQLS_PATH:-$run_dir/candidates.json}"
    if [[ -z "${PRED_SQLS_PATH:-}" ]]; then
        TEMPERATURE="$GEN_TEMPERATURE" MAX_TOKENS="$GEN_MAX_TOKENS" OUTPUT_PATH="$candidates" \
            bash scripts/run_gen_baseline.sh
    fi
    TEMPERATURE="$DPC_TEMPERATURE" PRED_SQLS_PATH="$candidates" OUTPUT_PATH="$run_dir/selected.json" \
        bash scripts/run_dpc_selection.sh
    "${PYTHON_BIN:-$default_python}" -m baseline.token_usage "$run_dir"
    PRED_PATH="$run_dir/selected.json" bash scripts/run_eval_ex.sh 2>&1 | tee "$run_dir/eval.log"
    "$python_bin" -m baseline.provenance finish "$DPC_SESSION" 0
    trap - EXIT
    exec {run_lock}>&-
done
