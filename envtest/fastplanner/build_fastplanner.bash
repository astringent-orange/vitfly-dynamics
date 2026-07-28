#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_WS="${VITFLY_MAIN_WORKSPACE:-$(cd "$REPO_ROOT/../.." && pwd)}"
WORKSPACE="${VITFLY_FASTPLANNER_WORKSPACE:-$REPO_ROOT/../../../.planner_workspaces/fastplanner}"
SOURCE="$WORKSPACE/src/Fast-Planner"
NLOPT_SOURCE="$WORKSPACE/deps/nlopt"
NLOPT_PREFIX="${VITFLY_NLOPT_PREFIX:-$WORKSPACE/deps/nlopt-install}"
VENDOR_SOURCE="$REPO_ROOT/third_party/Fast-Planner"
VENDOR_NLOPT="$REPO_ROOT/third_party/nlopt"
FAST_COMMIT="41be219fe4ecc43bf0e0c2b42a523f8755ccc0bd"
NLOPT_COMMIT="09b3c2a6da71cabcb98d2c8facc6b83d2321ed71"

source /opt/ros/noetic/setup.bash
mkdir -p "$WORKSPACE/src" "$WORKSPACE/deps"

if [ "$(git -C "$VENDOR_SOURCE" rev-parse HEAD 2>/dev/null || true)" != "$FAST_COMMIT" ]; then
  echo "[FASTPLANNER BUILD] initialize third_party/Fast-Planner at $FAST_COMMIT first" >&2
  exit 2
fi

if [ "$(git -C "$VENDOR_NLOPT" rev-parse HEAD 2>/dev/null || true)" != "$NLOPT_COMMIT" ]; then
  echo "[FASTPLANNER BUILD] initialize third_party/nlopt at $NLOPT_COMMIT first" >&2
  exit 2
fi

prepare_worktree() {
  local vendor="$1" target="$2" commit="$3"
  if [ -e "$target" ] && [ ! -d "$target/.git" ]; then
    echo "[FASTPLANNER BUILD] generated source is not a git worktree: $target" >&2
    exit 2
  fi
  if [ ! -d "$target/.git" ]; then
    git clone --local "$vendor" "$target"
  fi
  git -C "$target" reset --hard "$commit" >/dev/null
  git -C "$target" clean -fdx >/dev/null
}

prepare_worktree "$VENDOR_SOURCE" "$SOURCE" "$FAST_COMMIT"
prepare_worktree "$VENDOR_NLOPT" "$NLOPT_SOURCE" "$NLOPT_COMMIT"

if [ ! -f "$NLOPT_PREFIX/lib/libnlopt.so" ]; then
  cmake -S "$NLOPT_SOURCE" -B "$NLOPT_SOURCE/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$NLOPT_PREFIX"
  cmake --build "$NLOPT_SOURCE/build" --target install -j"${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
fi

apply_once() {
  local source_dir="$1" patch_file="$2"
  if git -C "$source_dir" apply --check --whitespace=fix --unidiff-zero --recount "$patch_file" >/dev/null 2>&1; then
    git -C "$source_dir" apply --whitespace=fix --unidiff-zero --recount "$patch_file"
  elif git -C "$source_dir" apply --reverse --check --unidiff-zero --recount "$patch_file" >/dev/null 2>&1; then
    : # already applied
  else
    echo "[FASTPLANNER BUILD] patch cannot be applied: $patch_file" >&2
    return 1
  fi
}

apply_once "$SOURCE" "$SCRIPT_DIR/patches/fast_planner_nlopt_prefix.patch"
apply_once "$SOURCE" "$SCRIPT_DIR/patches/fast_planner_goal_height.patch"
ln -sfn "$REPO_ROOT/planner_bridge" "$WORKSPACE/src/vitfly_planner_bridge"

export VITFLY_NLOPT_PREFIX="$NLOPT_PREFIX"
if command -v catkin >/dev/null 2>&1; then
  if [ ! -d "$WORKSPACE/.catkin_tools" ]; then
    (cd "$WORKSPACE" && catkin init)
  fi
  (cd "$WORKSPACE" && catkin config --extend "$MAIN_WS/devel" --merge-devel \
    --jobs "${CATKIN_BUILD_JOBS:-2}")
  (cd "$WORKSPACE" && catkin build plan_manage quadrotor_msgs vitfly_planner_bridge \
    --cmake-args -DCMAKE_BUILD_TYPE=Release
  )
else
  echo "[FASTPLANNER BUILD] catkin command is unavailable" >&2
  exit 1
fi

test -f "$WORKSPACE/devel/setup.bash"
echo "[FASTPLANNER BUILD] Ready: $WORKSPACE/devel/setup.bash"
