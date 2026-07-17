#!/usr/bin/env python3
"""Run paired Flightmare cases and write resumable benchmark results."""

import argparse
import csv
import hashlib
import json
import os
import signal
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from .policy_adapters import policy_environment, policy_index
except ImportError:
    from policy_adapters import policy_environment, policy_index


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "envtest" / "benchmark" / "configs" / "forest_benchmark_v1.yaml"
DEFAULT_CASES = ROOT / "envtest" / "benchmark" / "manifests" / "ablation_validation_cases.csv"
DEFAULT_ABLATION_OUTPUT = ROOT / "results" / "ablation"
RESULT_FIELDS = [
    "experiment_id", "case_id", "policy_id", "checkpoint_sha256", "scenario_id",
    "map_id", "desired_speed", "forest_density", "tree_count", "dynamic_profile",
    "dynamic_speed_mps", "phase_seed", "goal_reached", "success", "collision",
    "collision_count", "flight_time", "termination_elapsed_time", "termination_reason",
    "runner_returncode", "completed_at_utc",
]
ABLATION_MANIFEST = "ablation_validation_cases.csv"


def load(path):
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with open(temporary, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def read_results(path):
    return read_csv(path) if path.is_file() else []


def hash_scene(path):
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def validate_case_scene(case, cfg):
    """Refuse to run a case if its frozen scene was changed or removed."""
    scene_root = ROOT / "flightmare" / "flightpy" / "configs" / "vision"
    scene = scene_root / case["scene_path"]
    if not scene.is_dir():
        raise FileNotFoundError(f"case {case['case_id']} scene is missing: {scene}")
    required = ("static_obstacles.csv", "dynamic_obstacles.yaml", "astar_path.csv")
    missing = [name for name in required if not (scene / name).is_file()]
    if missing:
        raise FileNotFoundError(f"case {case['case_id']} scene missing: {', '.join(missing)}")
    actual = hash_scene(scene)
    expected = case.get("scene_hash", "")
    if expected and actual != expected:
        raise RuntimeError(
            f"immutable scene hash mismatch for {case['case_id']}: expected {expected}, got {actual}"
        )


def evaluation_result(path):
    if not path.is_file() or not path.read_text().strip():
        return None
    with open(path) as stream:
        value = yaml.safe_load(stream) or {}
    if "rollout_1" in value:
        value = value["rollout_1"] or {}
    if not isinstance(value, dict):
        return None
    crashes = int(value.get("number_crashes", value.get("collision_count", 0)))
    has_time = value.get("time_to_finish") not in (None, "")
    success = bool(value.get("Success", False))
    return {
        "goal_reached": int(value.get("goal_reached", has_time)),
        "success": int(success),
        "collision": int(crashes > 0),
        "collision_count": crashes,
        "flight_time": value.get("time_to_finish"),
        "termination_elapsed_time": value.get("termination_elapsed_time", value.get("time_to_finish")),
        "termination_reason": value.get("termination_reason") or ("goal_reached" if success else ("collision" if crashes else "timeout")),
    }


def select_policies(cfg, requested):
    policies = policy_index(cfg.get("policies", []))
    selected = []
    for policy_id in requested:
        if policy_id not in policies:
            raise ValueError(f"unknown policy id: {policy_id}")
        selected.append(policies[policy_id])
    if not selected:
        selected = list(policies.values())
    unique = {}
    for policy in selected:
        unique[policy["id"]] = policy
    return list(unique.values())


def resolve_output_path(cases_path, policies, requested_output=None, now=None):
    """Enforce one-policy ablations and provide their timestamped output path."""
    is_ablation = Path(cases_path).name == ABLATION_MANIFEST
    if is_ablation and len(policies) != 1:
        raise ValueError("ablation runs require exactly one --policy")
    if requested_output:
        return Path(requested_output), False
    if not is_ablation:
        raise ValueError("--output is required outside an ablation run")
    current = now or datetime.now()
    timestamp = current.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_ABLATION_OUTPUT / f"{policies[0]['id']}_{timestamp}", True


def run_one(cfg, policy, case, output, dry_run=False, runner_timeout=420.0):
    evaluation_path = output / "rollout_logs" / f"{policy['id']}__{case['case_id']}__evaluation.yaml"
    env = os.environ.copy()
    env.update(policy_environment(policy))
    env.update({
        "VITFLY_ENV_LEVEL": cfg.get("output_level", "forest_benchmark_v1"),
        "VITFLY_ENV_FOLDER": case["scene_id"],
        "VITFLY_ENV_SEED": str(int(case["map_id"]) + 10),
        "VITFLY_DYNAMIC_PHASE_SEED": str(case["phase_seed"]),
        "VITFLY_DES_VEL": str(case["desired_speed"]),
        "VITFLY_EVALUATION_PATH": str(evaluation_path.resolve()),
        "VITFLY_POLICY_CONFIG": str((output / "policies.yaml").resolve()),
        "VITFLY_CASE_CONFIG": str((output / "benchmark_cases.csv").resolve()),
        "VITFLY_EVALUATION_PROFILE": str(case.get("evaluation_profile", "strict")),
    })
    profile = (cfg.get("evaluation_profiles") or {}).get(case.get("evaluation_profile", "strict"), {})
    env.update({
        "VITFLY_EVAL_START_X": str(profile.get("start_x", 0.5)),
        "VITFLY_EVAL_GOAL_X": str(profile.get("goal_x", 60.0)),
        "VITFLY_EVAL_TIMEOUT_SECONDS": str(profile.get("timeout_seconds", 60.0)),
        "VITFLY_EVAL_COLLISION_MARGIN": str(profile.get("collision_margin", 0.0)),
        "VITFLY_EVAL_BOUNDING_BOX": ",".join(str(value) for value in profile.get("bounding_box", [-5, 65, -10, 10, 0, 10])),
    })
    command = ["bash", "launch_evaluation.bash", "1", "vision", "fixed_env"]
    print(f"[BENCHMARK] {policy['id']} {case['case_id']}")
    if dry_run:
        print("[BENCHMARK] command:", " ".join(command))
        return {
            "goal_reached": "", "success": "", "collision": "", "collision_count": "",
            "flight_time": "", "termination_elapsed_time": "", "termination_reason": "dry_run",
            "runner_returncode": 0,
        }
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=runner_timeout)
    except subprocess.TimeoutExpired:
        # launch_evaluation starts ROS children; terminate the complete process
        # group so a timed-out case cannot poison the next paired rollout.
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except OSError:
            pass
        process.wait()
        return {
            "goal_reached": 0, "success": 0, "collision": 0, "collision_count": 0,
            "flight_time": "", "termination_elapsed_time": "", "termination_reason": "timeout",
            "runner_returncode": 124,
        }
    completed = subprocess.CompletedProcess(command, returncode)
    summary = evaluation_result(evaluation_path)
    if summary is None:
        summary = {
            "goal_reached": 0, "success": 0, "collision": 0, "collision_count": 0,
            "flight_time": "", "termination_elapsed_time": "",
            "termination_reason": "controller_error" if completed.returncode else "missing_result",
        }
    elif completed.returncode != 0:
        summary["success"] = 0
        summary["termination_reason"] = "controller_error"
    summary["runner_returncode"] = completed.returncode
    summary["evaluation_yaml"] = str(evaluation_path)
    return summary


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"Benchmark config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--cases", default=DEFAULT_CASES, help=f"Case manifest (default: {DEFAULT_CASES})")
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", help="Result directory; defaults to a timestamped directory for ablation runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--runner-timeout", type=float, default=420.0)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    cfg = load(args.config)
    policies = select_policies(cfg, args.policy)
    try:
        output, used_default_output = resolve_output_path(
            args.cases,
            policies,
            requested_output=args.output,
        )
    except ValueError as error:
        parser.error(str(error))
    cases = read_csv(args.cases)
    if args.scenario:
        cases = [case for case in cases if case["scenario_id"] in set(args.scenario)]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("case manifest selection is empty")
    for case in cases:
        validate_case_scene(case, cfg)
    output.mkdir(parents=True, exist_ok=True)
    if used_default_output:
        print(f"[BENCHMARK] default output: {output}")
    result_path = output / "results.csv"
    rows = read_results(result_path)
    completed_keys = {(row["policy_id"], row["case_id"]) for row in rows}
    if not (output / "config.yaml").exists():
        shutil.copy2(args.config, output / "config.yaml")
        shutil.copy2(args.cases, output / "benchmark_cases.csv")
        with open(output / "policies.yaml", "w") as stream:
            yaml.safe_dump(policies, stream, sort_keys=False)
        (output / "git_state.txt").write_text(
            subprocess.run(["git", "status", "--short", "--branch"], cwd=ROOT, text=True, capture_output=True).stdout
        )

    for policy in policies:
        for case in cases:
            key = (policy["id"], case["case_id"])
            if args.resume and key in completed_keys:
                continue
            result = run_one(
                cfg, policy, case, output,
                dry_run=args.dry_run,
                runner_timeout=args.runner_timeout,
            )
            now = datetime.now(timezone.utc).isoformat()
            row = {
                "experiment_id": cfg.get("experiment_id", "benchmark"),
                "case_id": case["case_id"], "policy_id": policy["id"],
                "checkpoint_sha256": policy.get("checkpoint_sha256", "missing"),
                "scenario_id": case["scenario_id"], "map_id": case["map_id"],
                "desired_speed": case["desired_speed"], "forest_density": case["forest_density"],
                "tree_count": case["tree_count"], "dynamic_profile": case["dynamic_profile"],
                "dynamic_speed_mps": case["dynamic_speed_mps"], "phase_seed": case["phase_seed"],
                **{field: result.get(field, "") for field in RESULT_FIELDS if field not in {
                    "experiment_id", "case_id", "policy_id", "checkpoint_sha256", "scenario_id",
                    "map_id", "desired_speed", "forest_density", "tree_count", "dynamic_profile",
                    "dynamic_speed_mps", "phase_seed", "completed_at_utc",
                }},
                "completed_at_utc": now,
            }
            rows.append(row)
            write_csv(result_path, RESULT_FIELDS, rows)
            log_path = output / "rollout_logs" / f"{policy['id']}__{case['case_id']}.csv"
            write_csv(log_path, RESULT_FIELDS, [row])
    print(f"[BENCHMARK] wrote {result_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
