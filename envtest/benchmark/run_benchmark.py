#!/usr/bin/env python3
"""Run paired Flightmare cases and write resumable benchmark results."""

import argparse
import csv
import hashlib
import os
import signal
import shutil
import subprocess
import threading
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
    "experiment_id", "case_id", "policy_id", "source_policy_id", "adapter", "frame_offset",
    "checkpoint_sha256", "scenario_id",
    "map_id", "desired_speed", "forest_density", "tree_count", "dynamic_profile",
    "dynamic_speed_mps", "phase_seed", "goal_reached", "success", "collision",
    "collision_count", "flight_time", "termination_elapsed_time", "termination_reason",
    "runner_returncode", "attempt_count", "completed_at_utc",
]
ABLATION_MANIFEST = "ablation_validation_cases.csv"
INFRASTRUCTURE_REASONS = {"simulator_error", "runner_timeout", "missing_result"}
ACCEPTED_RUNNER_RETURNCODES = {0, 2}


def load(path):
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def normalize_policy_paths(cfg):
    """Resolve repository-relative checkpoint paths independently of the caller's cwd."""
    for policy in cfg.get("policies", []):
        checkpoint = policy.get("checkpoint")
        if checkpoint and not Path(checkpoint).is_absolute():
            policy["checkpoint"] = str(ROOT / checkpoint)
    return cfg


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


def validate_case_scene(case):
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


def empty_result(reason, returncode):
    return {
        "goal_reached": 0,
        "success": 0,
        "collision": 0,
        "collision_count": 0,
        "flight_time": "",
        "termination_elapsed_time": "",
        "termination_reason": reason,
        "runner_returncode": returncode,
    }


def terminate_process_group(process, interrupt_timeout=45.0, terminate_timeout=10.0):
    """Stop a benchmark launch group without leaving ROS children behind."""
    if process.poll() is not None:
        return process.returncode
    try:
        process_group = os.getpgid(process.pid)
    except OSError:
        process_group = None
    for sig, timeout in (
        (signal.SIGINT, interrupt_timeout),
        (signal.SIGTERM, terminate_timeout),
        (signal.SIGKILL, terminate_timeout),
    ):
        if process.poll() is not None:
            break
        try:
            if process_group is not None:
                os.killpg(process_group, sig)
            else:
                process.send_signal(sig)
        except (OSError, ProcessLookupError):
            pass
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
    return process.poll()


def stream_process_output(process, log_stream):
    """Copy one attempt's combined output to both the terminal and its log."""
    if process.stdout is None:
        return
    for line in iter(process.stdout.readline, ""):
        print(line, end="", flush=True)
        log_stream.write(line)
        log_stream.flush()
    process.stdout.close()


def run_attempt(cfg, policy, case, output, attempt, runner_timeout=420.0):
    evaluation_path = output / "rollout_logs" / f"{policy['id']}__{case['case_id']}__evaluation.yaml"
    attempt_log = output / "rollout_logs" / f"{policy['id']}__{case['case_id']}__attempt_{attempt}.log"
    evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    evaluation_path.unlink(missing_ok=True)
    env = os.environ.copy()
    env.update(policy_environment(policy))
    env.update({
        "VITFLY_BENCHMARK_MODE": "1",
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
    command = ["bash", "launch_evaluation.bash", "1", "vision", "fixed_env", "benchmark_mode"]
    print(f"[BENCHMARK] {policy['id']} {case['case_id']} attempt={attempt}")
    with open(attempt_log, "w") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            start_new_session=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            bufsize=1,
        )
        output_thread = threading.Thread(
            target=stream_process_output,
            args=(process, log_stream),
            daemon=True,
        )
        output_thread.start()
        try:
            returncode = process.wait(timeout=runner_timeout)
        except subprocess.TimeoutExpired:
            terminate_process_group(process)
            output_thread.join(timeout=5)
            return empty_result("runner_timeout", 124)
        except KeyboardInterrupt:
            terminate_process_group(process)
            output_thread.join(timeout=5)
            raise
        output_thread.join(timeout=5)

    if returncode == 130:
        raise KeyboardInterrupt
    summary = evaluation_result(evaluation_path)
    if returncode == 0 and summary is None:
        return empty_result("missing_result", 0)
    if returncode == 2:
        if summary is None:
            summary = empty_result("controller_error", 2)
        summary["success"] = 0
        summary["termination_reason"] = "controller_error"
    elif returncode == 3:
        return empty_result("simulator_error", 3)
    elif returncode != 0:
        return empty_result("simulator_error", returncode)
    summary["runner_returncode"] = returncode
    return summary


def attempt_is_retryable(result):
    return (
        int(result.get("runner_returncode", 0)) in {3, 124}
        or result.get("termination_reason") == "missing_result"
    )


def run_one(
    cfg,
    policy,
    case,
    output,
    runner_timeout=420.0,
    simulator_retries=1,
    starting_attempt=0,
):
    attempts = simulator_retries + 1
    for local_attempt in range(1, attempts + 1):
        attempt = starting_attempt + local_attempt
        result = run_attempt(
            cfg,
            policy,
            case,
            output,
            attempt,
            runner_timeout=runner_timeout,
        )
        result["attempt_count"] = attempt
        if not attempt_is_retryable(result) or local_attempt == attempts:
            return result
        print(
            f"[BENCHMARK] infrastructure failure ({result['termination_reason']}); "
            f"clean retry {local_attempt}/{simulator_retries}"
        )


def result_needs_rerun(row):
    """Return true for incomplete/infrastructure rows that --resume must replace."""
    reason = row.get("termination_reason", "")
    if reason in INFRASTRUCTURE_REASONS:
        return True
    try:
        returncode = int(row.get("runner_returncode", 0))
    except (TypeError, ValueError):
        return True
    return returncode not in ACCEPTED_RUNNER_RETURNCODES


def upsert_result(rows, new_row):
    key = (new_row["policy_id"], new_row["case_id"])
    return [
        row for row in rows
        if (row.get("policy_id"), row.get("case_id")) != key
    ] + [new_row]


def prior_attempt_count(row):
    try:
        return max(0, int(row.get("attempt_count") or 0))
    except (TypeError, ValueError):
        return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help=f"Benchmark config (default: {DEFAULT_CONFIG})")
    parser.add_argument("--cases", default=DEFAULT_CASES, help=f"Case manifest (default: {DEFAULT_CASES})")
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--policy-alias", help="Stable result ID for a single selected policy")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument(
        "--case-id",
        action="append",
        default=[],
        help="Run one exact manifest case; repeat for multiple cases",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", help="Result directory; defaults to a timestamped directory for ablation runs")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-timeout", type=float, default=420.0)
    parser.add_argument(
        "--simulator-retries",
        type=int,
        default=1,
        help="Clean retries after a simulator infrastructure failure",
    )
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.simulator_retries < 0:
        parser.error("--simulator-retries must be nonnegative")

    cfg = normalize_policy_paths(load(args.config))
    policies = select_policies(cfg, args.policy)
    if args.policy_alias:
        if len(policies) != 1:
            parser.error("--policy-alias requires exactly one selected --policy")
        if not args.policy_alias.replace("_", "").isalnum():
            parser.error("--policy-alias may contain only letters, numbers, and underscores")
        source = policies[0]
        policies = [dict(source, id=args.policy_alias, source_policy_id=source["id"])]
    else:
        policies = [dict(policy, source_policy_id=policy.get("source_policy_id", policy["id"])) for policy in policies]
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
    if args.case_id:
        requested_case_ids = set(args.case_id)
        available_case_ids = {case["case_id"] for case in cases}
        unknown_case_ids = requested_case_ids - available_case_ids
        if unknown_case_ids:
            parser.error(f"unknown case id: {sorted(unknown_case_ids)[0]}")
        cases = [case for case in cases if case["case_id"] in requested_case_ids]
    if args.limit is not None:
        cases = cases[:args.limit]
    if not cases:
        raise ValueError("case manifest selection is empty")
    for case in cases:
        validate_case_scene(case)
    output.mkdir(parents=True, exist_ok=True)
    if used_default_output:
        print(f"[BENCHMARK] default output: {output}")
    result_path = output / "results.csv"
    rows = read_results(result_path)
    previous_rows = {
        (row.get("policy_id"), row.get("case_id")): row
        for row in rows
    }
    completed_keys = {
        (row["policy_id"], row["case_id"])
        for row in rows
        if not result_needs_rerun(row)
    }
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
                runner_timeout=args.runner_timeout,
                simulator_retries=args.simulator_retries,
                starting_attempt=prior_attempt_count(previous_rows.get(key, {})),
            )
            now = datetime.now(timezone.utc).isoformat()
            row = {
                "experiment_id": cfg.get("experiment_id", "benchmark"),
                "case_id": case["case_id"], "policy_id": policy["id"],
                "source_policy_id": policy["source_policy_id"],
                "adapter": policy["adapter"],
                "frame_offset": policy.get("frame_offset", 0),
                "checkpoint_sha256": policy.get("checkpoint_sha256", "missing"),
                "scenario_id": case["scenario_id"], "map_id": case["map_id"],
                "desired_speed": case["desired_speed"], "forest_density": case["forest_density"],
                "tree_count": case["tree_count"], "dynamic_profile": case["dynamic_profile"],
                "dynamic_speed_mps": case["dynamic_speed_mps"], "phase_seed": case["phase_seed"],
                **{field: result.get(field, "") for field in RESULT_FIELDS if field not in {
                    "experiment_id", "case_id", "policy_id", "source_policy_id", "adapter", "frame_offset",
                    "checkpoint_sha256", "scenario_id",
                    "map_id", "desired_speed", "forest_density", "tree_count", "dynamic_profile",
                    "dynamic_speed_mps", "phase_seed", "completed_at_utc",
                }},
                "completed_at_utc": now,
            }
            rows = upsert_result(rows, row)
            write_csv(result_path, RESULT_FIELDS, rows)
            log_path = output / "rollout_logs" / f"{policy['id']}__{case['case_id']}.csv"
            write_csv(log_path, RESULT_FIELDS, [row])
            if result_needs_rerun(row):
                print(
                    f"[BENCHMARK] stopped after infrastructure failure: "
                    f"{policy['id']} {case['case_id']} ({row['termination_reason']})"
                )
                raise SystemExit(3)
    print(f"[BENCHMARK] wrote {result_path} rows={len(rows)}")


if __name__ == "__main__":
    main()
