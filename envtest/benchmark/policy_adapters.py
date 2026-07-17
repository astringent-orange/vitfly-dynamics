#!/usr/bin/env python3
"""Policy registry and launch metadata for benchmark controllers.

Only the two ViTFly adapters are executable in this milestone.  The ROS
planner adapter names are reserved so future FastPlanner/EGO-Planner bridges
can be added without changing the case manifest or result schema.
"""

import hashlib
from pathlib import Path
from typing import Any, Dict, Protocol


SUPPORTED = {"vitfly_neural", "vitfly_legacy", "fastplanner_ros", "egoplanner_ros"}
PLANNER_ADAPTERS = {"fastplanner_ros", "egoplanner_ros"}
PLANNER_COMMAND_TOPIC = "/kingfisher/dodgeros_pilot/feedthrough_command"


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
        command_type = policy.get("command_type", "dodgeros_msgs/Command")
        command_mode = int(policy.get("command_mode", 2))
        command_frame = policy.get("command_frame", "world")
        if command_topic != PLANNER_COMMAND_TOPIC:
            raise ValueError(f"{policy['id']} must publish to {PLANNER_COMMAND_TOPIC}")
        if command_type != "dodgeros_msgs/Command" or command_mode != 2 or command_frame != "world":
            raise ValueError(f"{policy['id']} must publish world-frame dodgeros_msgs/Command mode 2")
        if enabled and not str(policy.get("launch_command", "")).strip():
            raise ValueError(f"enabled planner {policy['id']} requires launch_command")
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
    if adapter not in ("vitfly_neural", "vitfly_legacy"):
        if not policy.get("enabled", False):
            raise RuntimeError(f"{policy['id']} is disabled because its ROS adapter is not integrated")
        raise NotImplementedError(f"{adapter} must implement planner launch and command conversion")
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
