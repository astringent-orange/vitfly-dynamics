#!/usr/bin/env python3
"""Run one model over the immutable main-comparison test manifest."""

import argparse
from datetime import datetime
from pathlib import Path

try:
    from .policy_adapters import PLANNER_ADAPTERS, policy_index
    from .run_benchmark import ROOT, load, main as run_benchmark_main, normalize_policy_paths
except ImportError:
    from policy_adapters import PLANNER_ADAPTERS, policy_index
    from run_benchmark import ROOT, load, main as run_benchmark_main, normalize_policy_paths


DEFAULT_CONFIG = ROOT / "envtest" / "benchmark" / "configs" / "forest_benchmark_v1.yaml"
DEFAULT_CASES = ROOT / "envtest" / "benchmark" / "manifests" / "comparison_test_cases.csv"
DEFAULT_OUTPUT_ROOT = ROOT / "results" / "comparison"
POLICY_CHOICES = ("single", "adjacent", "skip_one", "vitfly", "fastplanner", "egoplanner")


def resolve_policy(cfg, requested_policy):
    """Return (source policy id, stable result id) for one comparison policy."""
    comparison = cfg.get("design", {}).get("comparison_models", {})
    best = comparison.get("best", {})
    if requested_policy in tuple(best.get("candidates", ())):
        source_policy = requested_policy
        result_id = best["result_id"]
    else:
        if requested_policy not in comparison:
            raise ValueError(f"comparison policy is not configured: {requested_policy}")
        spec = comparison[requested_policy]
        source_policy = spec["source_policy"]
        result_id = spec["result_id"]

    policies = policy_index(cfg.get("policies", []))
    if source_policy not in policies:
        raise ValueError(f"source policy is not configured: {source_policy}")
    policy = policies[source_policy]
    if policy["adapter"] in PLANNER_ADAPTERS and not policy.get("enabled", False):
        raise ValueError(
            f"{requested_policy} is not integrated: configure its ROS adapter and set enabled: true"
        )
    return source_policy, result_id


def default_output_path(result_id, now=None):
    current = now or datetime.now()
    return DEFAULT_OUTPUT_ROOT / f"{result_id}_{current.strftime('%Y%m%d_%H%M%S')}"


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--policy", required=True, choices=POLICY_CHOICES)
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="Benchmark configuration")
    parser.add_argument("--cases", default=DEFAULT_CASES, help="Immutable comparison manifest")
    parser.add_argument("--scenario", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", help="Result directory; timestamped under results/comparison by default")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--runner-timeout", type=float, default=420.0)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = normalize_policy_paths(load(args.config))
    try:
        source_policy, result_id = resolve_policy(cfg, args.policy)
    except ValueError as error:
        parser.error(str(error))

    output = Path(args.output) if args.output else default_output_path(result_id)
    print(f"[COMPARISON] model={result_id} source_policy={source_policy}")
    print(f"[COMPARISON] output={output}")
    benchmark_args = [
        "--config", str(args.config),
        "--cases", str(args.cases),
        "--policy", source_policy,
        "--policy-alias", result_id,
        "--output", str(output),
        "--runner-timeout", str(args.runner_timeout),
    ]
    for scenario in args.scenario:
        benchmark_args.extend(("--scenario", scenario))
    if args.limit is not None:
        benchmark_args.extend(("--limit", str(args.limit)))
    if args.resume:
        benchmark_args.append("--resume")
    run_benchmark_main(benchmark_args)


if __name__ == "__main__":
    main()
