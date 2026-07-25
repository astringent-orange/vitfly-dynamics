#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_DIR/../.." && pwd)"
COLLECTION_ROOT="${VITFLY_ABLATION_ROOT:-$SCRIPT_DIR/train_set/vitfly_ablation}"
LEGACY_DIR="${VITFLY_ABLATION_LEGACY_DIR:-$SCRIPT_DIR/train_set/vitfly_original_rtf2_phase1505_retry1}"
TARGET="${VITFLY_ABLATION_TARGET:-408}"
PHASE_BASE="${VITFLY_ABLATION_START_PHASE:-1101}"
REAL_TIME_FACTOR="${VITFLY_REAL_TIME_FACTOR:-2}"
PYTHON_BIN="${VITFLY_PYTHON:-/home/mian/miniconda3/envs/pubflight/bin/python3.8}"
STATUS_FILE="$COLLECTION_ROOT/collection_queue.tsv"

mkdir -p "$COLLECTION_ROOT"

count_trajectories() {
  local count=0
  if [ -d "$LEGACY_DIR" ]; then
    count=$((count + $(find "$LEGACY_DIR" -mindepth 2 -maxdepth 2 -name data.csv -type f | wc -l)))
  fi
  count=$((count + $(find "$COLLECTION_ROOT" -mindepth 3 -maxdepth 3 -path '*/phase*/*/data.csv' -type f | wc -l)))
  printf '%d\n' "$count"
}

batch_complete() {
  local summary="$1/collection_summary.json"
  [ -s "$summary" ] && grep -q '"collection_runs": 1' "$summary"
}

next_phase() {
  PHASE_BASE=$((PHASE_BASE + 101))
  if [ "$PHASE_BASE" -eq 1505 ]; then
    PHASE_BASE=$((PHASE_BASE + 101))
  fi
}

if [ ! -f "$STATUS_FILE" ]; then
  printf 'timestamp\tphase_seed_base\tstatus\tretained\tcumulative\n' > "$STATUS_FILE"
fi

cd "$WORKSPACE_DIR"
# Catkin setup scripts reference optional shell variables, so nounset must be
# enabled only after the ROS environment has been loaded.
source devel/setup.bash
set -u
cd "$REPO_DIR"

while true; do
  valid_count="$(count_trajectories)"
  if [ "$valid_count" -ge "$TARGET" ]; then
    printf '%s\t-\tcomplete\t-\t%s\n' "$(date --iso-8601=seconds)" "$valid_count" >> "$STATUS_FILE"
    echo "[VITFLY COLLECTOR] Target reached: $valid_count/$TARGET valid trajectories."
    exit 0
  fi

  batch_dir="$COLLECTION_ROOT/phase${PHASE_BASE}"
  if batch_complete "$batch_dir"; then
    echo "[VITFLY COLLECTOR] phase${PHASE_BASE} already complete; skipping."
    next_phase
    continue
  fi
  if [ -d "$batch_dir" ] && find "$batch_dir" -mindepth 1 -print -quit | grep -q .; then
    incomplete_dir="${batch_dir}_incomplete_$(date +%Y%m%d_%H%M%S)"
    mv "$batch_dir" "$incomplete_dir"
    echo "[VITFLY COLLECTOR] Preserved incomplete batch as $incomplete_dir"
  fi

  mkdir -p "$batch_dir"
  evaluation_path="$batch_dir/evaluation_phase${PHASE_BASE}.yaml"
  collection_log="$batch_dir/collection.log"
  printf '%s\t%s\trunning\t-\t%s\n' "$(date --iso-8601=seconds)" "$PHASE_BASE" "$valid_count" >> "$STATUS_FILE"
  echo "[VITFLY COLLECTOR] Starting phase${PHASE_BASE}; current=$valid_count target=$TARGET"

  export VITFLY_PYTHON="$PYTHON_BIN"
  export VITFLY_REAL_TIME_FACTOR="$REAL_TIME_FACTOR"
  export VITFLY_DATASET_DIR="$batch_dir"
  export VITFLY_EVALUATION_PATH="$evaluation_path"
  unset VITFLY_DYNAMIC_PHASE_SEED

  set +e
  bash launch_evaluation.bash 101 state expert=vitfly phase_seed_base="$PHASE_BASE" \
    > >(tee "$collection_log") 2>&1
  launch_status=$?
  set -e

  retained=$(find "$batch_dir" -mindepth 2 -maxdepth 2 -name data.csv -type f | wc -l)
  valid_count="$(count_trajectories)"
  if [ "$launch_status" -ne 0 ]; then
    printf '%s\t%s\tfailed:%s\t%s\t%s\n' "$(date --iso-8601=seconds)" "$PHASE_BASE" "$launch_status" "$retained" "$valid_count" >> "$STATUS_FILE"
    echo "[VITFLY COLLECTOR] phase${PHASE_BASE} failed with status $launch_status; stopping for inspection."
    exit "$launch_status"
  fi

  printf '%s\t%s\tcomplete\t%s\t%s\n' "$(date --iso-8601=seconds)" "$PHASE_BASE" "$retained" "$valid_count" >> "$STATUS_FILE"
  echo "[VITFLY COLLECTOR] phase${PHASE_BASE} retained=$retained cumulative=$valid_count"
  next_phase
done
