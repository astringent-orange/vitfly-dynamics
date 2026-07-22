#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
MAIN_WS="${VITFLY_MAIN_WORKSPACE:-$(cd "$REPO_ROOT/../.." && pwd)}"
WORKSPACE="${VITFLY_FASTPLANNER_WORKSPACE:-$MAIN_WS/../fastplanner_ws}"
SOURCE="$WORKSPACE/src/Fast-Planner"
NLOPT_SOURCE="$WORKSPACE/deps/nlopt"
NLOPT_PREFIX="${VITFLY_NLOPT_PREFIX:-$WORKSPACE/deps/nlopt-install}"
FAST_COMMIT="41be219fe4ecc43bf0e0c2b42a523f8755ccc0bd"
NLOPT_COMMIT="09b3c2a6da71cabcb98d2c8facc6b83d2321ed71"

source /opt/ros/noetic/setup.bash
mkdir -p "$WORKSPACE/src" "$WORKSPACE/deps"

if [ ! -d "$SOURCE/.git" ]; then
  git clone https://github.com/HKUST-Aerial-Robotics/Fast-Planner.git "$SOURCE"
fi
git -C "$SOURCE" fetch --depth=1 origin "$FAST_COMMIT"
git -C "$SOURCE" checkout --detach "$FAST_COMMIT"

if [ ! -d "$NLOPT_SOURCE/.git" ]; then
  git clone https://github.com/stevengj/nlopt.git "$NLOPT_SOURCE"
fi
git -C "$NLOPT_SOURCE" fetch --depth=1 origin "$NLOPT_COMMIT"
git -C "$NLOPT_SOURCE" checkout --detach "$NLOPT_COMMIT"

if [ ! -f "$NLOPT_PREFIX/lib/libnlopt.so" ]; then
  cmake -S "$NLOPT_SOURCE" -B "$NLOPT_SOURCE/build" \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON \
    -DCMAKE_INSTALL_PREFIX="$NLOPT_PREFIX"
  cmake --build "$NLOPT_SOURCE/build" --target install -j"${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
fi

apply_once() {
  local source_dir="$1" patch_file="$2"
  if git -C "$source_dir" apply --check --unidiff-zero --recount "$patch_file" >/dev/null 2>&1; then
    git -C "$source_dir" apply --unidiff-zero --recount "$patch_file"
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
  (cd "$WORKSPACE" && catkin build plan_manage quadrotor_msgs vitfly_planner_bridge \
    --cmake-args -DCMAKE_BUILD_TYPE=Release)
else
  echo "[FASTPLANNER BUILD] catkin command is unavailable" >&2
  exit 1
fi

test -f "$WORKSPACE/devel/setup.bash"
echo "[FASTPLANNER BUILD] Ready: $WORKSPACE/devel/setup.bash"
