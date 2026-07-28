#!/usr/bin/env python3
"""Static preflight audit for external planner comparison runs."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml

try:
    from .policy_adapters import PLANNER_ADAPTERS, policy_index
    from .run_benchmark import DEFAULT_CASES, DEFAULT_CONFIG, ROOT, load
except ImportError:
    from policy_adapters import PLANNER_ADAPTERS, policy_index
    from run_benchmark import DEFAULT_CASES, DEFAULT_CONFIG, ROOT, load


EXPECTED_COMMITS = {
    "fastplanner_ros": "41be219fe4ecc43bf0e0c2b42a523f8755ccc0bd",
    "egoplanner_ros": "bfda51284c8c1b476043255a8145ef925a3778a5",
}


def git_head(path):
    try:
        return subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True, text=True, capture_output=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def patch_applied(source_dir, patch_path):
    """Return whether a generated worktree contains the recorded patch."""
    try:
        return subprocess.run(
            [
                "git", "-C", str(source_dir), "apply", "--reverse", "--check",
                "--unidiff-zero", "--recount", str(patch_path),
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        ).returncode == 0
    except OSError:
        return False


def audit(config_path, cases_path, requested):
    cfg = load(config_path)
    policies = policy_index(cfg.get("policies", []))
    selected = requested or ["fastplanner", "egoplanner"]
    checks = []
    for policy_id in selected:
        if policy_id not in policies:
            checks.append({"policy": policy_id, "ok": False, "error": "missing policy"})
            continue
        policy = policies[policy_id]
        adapter = policy["adapter"]
        ok = adapter in PLANNER_ADAPTERS and bool(policy.get("enabled"))
        row = {"policy": policy_id, "adapter": adapter, "ok": ok}
        wrapper = policy.get("launch_command", "")
        wrapper_path = Path(wrapper)
        if not wrapper_path.is_absolute():
            wrapper_path = ROOT / wrapper_path
        row["wrapper"] = str(wrapper_path)
        row["wrapper_exists"] = wrapper_path.is_file()
        row["ok"] = row["ok"] and row["wrapper_exists"]
        patch_paths = []
        for patch_name in policy.get("patches", []):
            patch_path = Path(patch_name)
            if not patch_path.is_absolute():
                patch_path = ROOT / patch_path
            patch_paths.append({"path": str(patch_path), "exists": patch_path.is_file()})
        row["patches"] = patch_paths
        row["patches_exist"] = all(item["exists"] for item in patch_paths)
        row["ok"] = row["ok"] and row["patches_exist"]
        if adapter in EXPECTED_COMMITS:
            is_fast = adapter == "fastplanner_ros"
            workspace_var = "VITFLY_FASTPLANNER_WORKSPACE" if is_fast else "VITFLY_EGOPLANNER_WORKSPACE"
            row["workspace_env"] = workspace_var
            row["expected_commit"] = EXPECTED_COMMITS[adapter]
            row["configured_commit"] = policy.get("upstream_commit", "")
            row["configured_commit_matches"] = row["configured_commit"] == row["expected_commit"]
            submodule = ROOT / "third_party" / ("Fast-Planner" if is_fast else "ego-planner")
            row["submodule_path"] = str(submodule)
            row["submodule_commit"] = git_head(submodule)
            row["submodule_initialized"] = bool(row["submodule_commit"])
            row["submodule_commit_matches"] = row["submodule_commit"] == row["expected_commit"]
            default_workspace = ROOT.parents[1].parent / ".planner_workspaces" / (
                "fastplanner" if is_fast else "egoplanner"
            )
            workspace = Path(os.environ.get(workspace_var, str(default_workspace)))
            source_dir = workspace / "src" / ("Fast-Planner" if is_fast else "ego-planner")
            row["workspace"] = str(workspace)
            row["source_commit"] = git_head(source_dir)
            row["commit_matches"] = row["source_commit"] == row["expected_commit"]
            row["devel_setup_exists"] = (workspace / "devel" / "setup.bash").is_file()
            row["patches_applied"] = all(
                patch_applied(source_dir, item["path"])
                for item in patch_paths
            )
            row["ok"] = (
                row["ok"] and row["configured_commit_matches"] and
                row["submodule_initialized"] and row["submodule_commit_matches"] and
                row["commit_matches"] and row["devel_setup_exists"] and
                row["patches_applied"]
            )
        checks.append(row)

    cases = []
    with open(cases_path, newline="") as stream:
        import csv
        cases = list(csv.DictReader(stream))
    case_ids = [row.get("case_id", "") for row in cases]
    checks.append({
        "manifest": str(cases_path),
        "case_count": len(cases),
        "unique_case_count": len(set(case_ids)),
        "ok": bool(cases) and len(case_ids) == len(set(case_ids)),
    })
    return {
        "config": str(config_path),
        "cases": str(cases_path),
        "checks": checks,
        "ready": all(check.get("ok", False) for check in checks),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--policy", action="append", default=[])
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "preflight.json")
    args = parser.parse_args(argv)
    report = audit(args.config, args.cases, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ready": report["ready"], "output": str(args.output)}, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
