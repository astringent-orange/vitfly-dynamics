#!/usr/bin/env python3
"""Policy registry and launch metadata for benchmark controllers.

The ViTFly adapters and the two external ROS planner wrappers share the same
benchmark lifecycle.  Planner-specific message conversion is isolated in
``planner_bridge``; the result schema and immutable case manifests remain
common to every policy.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Protocol


SUPPORTED = {"vitfly_neural", "vitfly_legacy", "fastplanner_ros", "egoplanner_ros"}
PLANNER_ADAPTERS = {"fastplanner_ros", "egoplanner_ros"}
PLANNER_COMMAND_TOPIC = "/kingfisher/dodgeros_pilot/velocity_command"
PLANNER_COMMAND_TYPE = "geometry_msgs/TwistStamped"
PLANNER_FRAME = "world"
PLANNER_DEFAULTS = {
    "fastplanner_ros": {
        "launch_command": "envtest/fastplanner/launch_controller.bash",
        "ready_topic": "/fastplanner/ready",
    },
    "egoplanner_ros": {
        "launch_command": "envtest/egoplanner/launch_controller.bash",
        "ready_topic": "/egoplanner/ready",
    },
}


class PlannerLifecycle(Protocol):
    """Interface a future FastPlanner/EGO-Planner ROS bridge must implement."""

    def validate(self, policy_config: Dict[str, Any]) -> None:
        ...

    def start(self, case_config: Dict[str, Any]) -> None:
        ...

    def wait_ready(self, timeout: float) -> bool:
        ...

    def stop(self) -> None:
        ...

    def metadata(self) -> Dict[str, Any]:
        ...


def checkpoint_sha256(path):
    path = Path(path)
    if not path.is_file():
        return "missing"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_policy(policy):
    required = ("id", "adapter")
    missing = [key for key in required if not policy.get(key)]
    if missing:
        raise ValueError(f"policy missing fields: {', '.join(missing)}")
    if policy["adapter"] not in SUPPORTED:
        raise ValueError(f"unsupported policy adapter: {policy['adapter']}")
    offset = int(policy.get("frame_offset", 0))
    if offset not in (0, 1, 2):
        raise ValueError("frame_offset must be 0, 1, or 2")
    adapter = policy["adapter"]
    enabled = bool(policy.get("enabled", True))
    if adapter.startswith("vitfly") and not str(policy.get("checkpoint", "")).strip():
        raise ValueError(f"{policy['id']} requires a checkpoint")
    if adapter in PLANNER_ADAPTERS:
        command_topic = policy.get("command_topic", PLANNER_COMMAND_TOPIC)
        command_type = policy.get("command_type", PLANNER_COMMAND_TYPE)
        command_frame = policy.get("command_frame", PLANNER_FRAME)
        if command_topic != PLANNER_COMMAND_TOPIC:
            raise ValueError(f"{policy['id']} must publish to {PLANNER_COMMAND_TOPIC}")
        if command_type != PLANNER_COMMAND_TYPE or command_frame != PLANNER_FRAME:
            raise ValueError(f"{policy['id']} must publish world-frame {PLANNER_COMMAND_TYPE}")
        defaults = PLANNER_DEFAULTS[adapter]
        if enabled and not str(policy.get("launch_command", defaults["launch_command"])).strip():
            raise ValueError(f"enabled planner {policy['id']} requires launch_command")
        policy.setdefault("launch_command", defaults["launch_command"])
        policy.setdefault("ready_topic", defaults["ready_topic"])
        policy.setdefault("command_topic", PLANNER_COMMAND_TOPIC)
        policy.setdefault("command_type", PLANNER_COMMAND_TYPE)
        policy.setdefault("command_frame", PLANNER_FRAME)
    checkpoint = policy.get("checkpoint", "")
    return dict(
        policy,
        enabled=enabled,
        frame_offset=offset,
        checkpoint_sha256=checkpoint_sha256(checkpoint) if checkpoint else "not_applicable",
    )


def policy_environment(policy):
    """Environment variables consumed by the existing launch script."""
    adapter = policy["adapter"]
    if adapter in PLANNER_ADAPTERS:
        if not policy.get("enabled", False):
            raise RuntimeError(f"{policy['id']} is disabled in the benchmark configuration")
        launch_command = str(policy.get("launch_command", PLANNER_DEFAULTS[adapter]["launch_command"]))
        launch_path = Path(launch_command)
        if not launch_path.is_absolute():
            launch_path = Path(__file__).resolve().parents[2] / launch_path
        if not launch_path.is_file():
            raise FileNotFoundError(f"planner launch wrapper does not exist: {launch_path}")
        return {
            "VITFLY_BENCHMARK_MODE": "1",
            "VITFLY_PLANNER_ADAPTER": adapter,
            "VITFLY_PLANNER_LAUNCH": str(launch_path.resolve()),
            "VITFLY_PLANNER_READY_TOPIC": str(policy.get("ready_topic", PLANNER_DEFAULTS[adapter]["ready_topic"])),
            "VITFLY_PLANNER_COMMAND_TOPIC": str(policy.get("command_topic", PLANNER_COMMAND_TOPIC)),
            "VITFLY_PLANNER_COMMAND_TYPE": str(policy.get("command_type", PLANNER_COMMAND_TYPE)),
            "VITFLY_PLANNER_FRAME": str(policy.get("command_frame", PLANNER_FRAME)),
            "VITFLY_MAIN_WORKSPACE": str(Path(__file__).resolve().parents[4]),
            "VITFLY_OFFSET": "0",
        }
    if adapter not in ("vitfly_neural", "vitfly_legacy"):
        raise ValueError(f"unsupported executable adapter: {adapter}")
    return {
        "VITFLY_OFFSET": str(policy.get("frame_offset", 0)),
        "VITFLY_MODEL_PATH": str(Path(policy["checkpoint"]).resolve()),
        "VITFLY_BENCHMARK_MODE": "1",
        "VITFLY_ALLOW_LEGACY_CHECKPOINT": "1" if adapter == "vitfly_legacy" else "0",
    }


def policy_index(policies):
    result = {}
    for raw in policies:
        checked = validate_policy(raw)
        if checked["id"] in result:
            raise ValueError(f"duplicate policy id: {checked['id']}")
        result[checked["id"]] = checked
    return result
