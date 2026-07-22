#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_WS="${VITFLY_MAIN_WORKSPACE:-$(cd "$REPO_ROOT/../.." && pwd)}"
WORKSPACE="${VITFLY_EGOPLANNER_WORKSPACE:-$MAIN_WS/../egoplanner_ws}"
SOURCE="$WORKSPACE/src/ego-planner"
EGO_COMMIT="bfda51284c8c1b476043255a8145ef925a3778a5"

source /opt/ros/noetic/setup.bash
mkdir -p "$WORKSPACE/src"

if [ ! -d "$SOURCE/.git" ]; then
  git clone https://github.com/ZJU-FAST-Lab/ego-planner.git "$SOURCE"
fi
git -C "$SOURCE" fetch --depth=1 origin "$EGO_COMMIT"
git -C "$SOURCE" checkout --detach "$EGO_COMMIT"

apply_once() {
  local source_dir="$1" patch_file="$2"
  if git -C "$source_dir" apply --check --unidiff-zero --recount "$patch_file" >/dev/null 2>&1; then
    git -C "$source_dir" apply --unidiff-zero --recount "$patch_file"
  elif git -C "$source_dir" apply --reverse --check --unidiff-zero --recount "$patch_file" >/dev/null 2>&1; then
    : # already applied
  else
    echo "[EGOPLANNER BUILD] patch cannot be applied: $patch_file" >&2
    return 1
  fi
}

apply_once "$SOURCE" "$SCRIPT_DIR/patches/ego_planner_goal_height.patch"
apply_once "$SOURCE" "$SCRIPT_DIR/patches/ego_planner_high_speed_compatibility.patch"
ln -sfn "$REPO_ROOT/planner_bridge" "$WORKSPACE/src/vitfly_planner_bridge"

if command -v catkin >/dev/null 2>&1; then
  (cd "$WORKSPACE" && catkin build ego_planner quadrotor_msgs vitfly_planner_bridge \
    --cmake-args -DCMAKE_BUILD_TYPE=Release)
else
  echo "[EGOPLANNER BUILD] catkin command is unavailable" >&2
  exit 1
fi

test -f "$WORKSPACE/devel/setup.bash"
echo "[EGOPLANNER BUILD] Ready: $WORKSPACE/devel/setup.bash"
