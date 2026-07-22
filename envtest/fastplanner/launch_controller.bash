#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_WS="${VITFLY_MAIN_WORKSPACE:-$(cd "$REPO_ROOT/../.." && pwd)}"
PLANNER_WS="${VITFLY_FASTPLANNER_WORKSPACE:-$(cd "$MAIN_WS/../fastplanner_ws" 2>/dev/null && pwd)}"
PYTHON_BIN="${VITFLY_PYTHON:-python3}"
PLANNER_PID=""
BRIDGE_PID=""
cleaned=0

source_if_exists() {
  if [ -f "$1" ]; then
    # shellcheck disable=SC1090
    source "$1"
  fi
}

cleanup() {
  local status="$1"
  [ "$cleaned" -eq 1 ] && return
  cleaned=1
  trap - EXIT INT TERM
  if [ -n "$BRIDGE_PID" ] && kill -0 "$BRIDGE_PID" 2>/dev/null; then
    kill -INT "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
  fi
  if [ -n "$PLANNER_PID" ] && kill -0 "$PLANNER_PID" 2>/dev/null; then
    kill -INT "$PLANNER_PID" 2>/dev/null || true
    sleep 1
    kill -TERM "$PLANNER_PID" 2>/dev/null || true
    wait "$PLANNER_PID" 2>/dev/null || true
  fi
  exit "$status"
}
trap 'cleanup 130' INT TERM
trap 'cleanup $?' EXIT

source_if_exists /opt/ros/noetic/setup.bash
source_if_exists "$MAIN_WS/devel/setup.bash"
source_if_exists "$PLANNER_WS/devel/setup.bash"

if ! command -v roslaunch >/dev/null 2>&1 || ! rospack find plan_manage >/dev/null 2>&1; then
  echo "[FASTPLANNER] planner overlay is not built or sourced" >&2
  exit 3
fi

desired_speed="${VITFLY_DES_VEL:-5.0}"
roslaunch vitfly_planner_bridge fastplanner_benchmark.launch \
  desired_speed:="$desired_speed" start_bridge:=false >"${VITFLY_PLANNER_LOG:-/tmp/vitfly_fastplanner.log}" 2>&1 &
PLANNER_PID="$!"

"$PYTHON_BIN" "$REPO_ROOT/planner_bridge/scripts/planner_bridge.py" \
  --planner fastplanner --desired-speed "$desired_speed" \
  --command-topic /fastplanner/position_cmd &
BRIDGE_PID="$!"
wait "$BRIDGE_PID"
bridge_status=$?
BRIDGE_PID=""
if [ "$bridge_status" -eq 0 ]; then
  exit 0
elif [ "$bridge_status" -eq 2 ]; then
  exit 2
fi
exit 3
