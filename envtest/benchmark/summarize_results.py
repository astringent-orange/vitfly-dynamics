#!/usr/bin/env python3
"""Summarize benchmark results for manual model comparison."""

import argparse
import csv
import json
import random
from pathlib import Path

SUMMARY_FIELDS = [
    "policy_id", "scenario_id", "total", "success_count", "success_rate",
    "collision_count", "collision_rate", "successful_time_mean",
    "successful_time_median", "success_ci_low", "success_ci_high",
]
PAIRED_FIELDS = [
    "scenario_id", "policy_a", "policy_b", "paired_total",
    "success_difference_mean", "collision_difference_mean",
    "successful_time_difference_mean",
]


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, fields, rows):
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def bootstrap(values, repeats=2000, seed=0):
    values = [float(value) for value in values]
    if not values:
        return "", ""
    if len(set(values)) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    samples = []
    for _ in range(repeats):
        draw = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(draw) / len(draw))
    samples.sort()
    return samples[int(0.025 * len(samples))], samples[int(0.975 * len(samples))]


def as_int(value):
    if value in (None, ""):
        return 0
    return int(float(value))


def summarize(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["policy_id"], row["scenario_id"]), []).append(row)
    summaries = []
    for (policy, scenario), group in sorted(groups.items()):
        successes = [as_int(row.get("success", 0)) for row in group]
        collisions = [as_int(row.get("collision", 0)) for row in group]
        times = [float(row["flight_time"]) for row in group if row.get("success") == "1" and row.get("flight_time") not in ("", None)]
        ci_low, ci_high = bootstrap(successes)
        summaries.append({
            "policy_id": policy,
            "scenario_id": scenario,
            "total": len(group),
            "success_count": sum(successes),
            "success_rate": sum(successes) / len(group) if group else "",
            "collision_count": sum(collisions),
            "collision_rate": sum(collisions) / len(group) if group else "",
            "successful_time_mean": sum(times) / len(times) if times else "",
            "successful_time_median": sorted(times)[len(times) // 2] if times else "",
            "success_ci_low": ci_low,
            "success_ci_high": ci_high,
        })
    return summaries


def paired_comparisons(rows):
    """Return paired policy deltas for identical immutable case IDs."""
    by_case = {}
    for row in rows:
        by_case.setdefault((row["scenario_id"], row["case_id"]), {})[row["policy_id"]] = row
    grouped = {}
    for (scenario, _case_id), policies in by_case.items():
        names = sorted(policies)
        for index, policy_a in enumerate(names):
            for policy_b in names[index + 1:]:
                left, right = policies[policy_a], policies[policy_b]
                key = (scenario, policy_a, policy_b)
                entry = grouped.setdefault(key, {"success": [], "collision": [], "time": []})
                entry["success"].append(as_int(right.get("success", 0)) - as_int(left.get("success", 0)))
                entry["collision"].append(as_int(right.get("collision", 0)) - as_int(left.get("collision", 0)))
                if left.get("flight_time") not in ("", None) and right.get("flight_time") not in ("", None):
                    entry["time"].append(float(right["flight_time"]) - float(left["flight_time"]))
    output = []
    for (scenario, policy_a, policy_b), values in sorted(grouped.items()):
        output.append({
            "scenario_id": scenario,
            "policy_a": policy_a,
            "policy_b": policy_b,
            "paired_total": len(values["success"]),
            "success_difference_mean": sum(values["success"]) / len(values["success"]),
            "collision_difference_mean": sum(values["collision"]) / len(values["collision"]),
            "successful_time_difference_mean": (
                sum(values["time"]) / len(values["time"])
                if values["time"] else ""
            ),
        })
    return output


def plot_summaries(summaries, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SUMMARY] matplotlib unavailable; skipped plots")
        return
    policies = sorted({row["policy_id"] for row in summaries})
    scenarios = sorted({row["scenario_id"] for row in summaries})
    for metric, filename, ylabel in (
        ("success_rate", "success_rate_by_scenario.png", "Success rate"),
        ("collision_rate", "collision_rate_by_scenario.png", "Collision rate"),
        ("successful_time_median", "flight_time_by_scenario.png", "Median flight time (s)"),
    ):
        fig, ax = plt.subplots(figsize=(max(8, len(scenarios) * 1.3), 5))
        for policy in policies:
            values = []
            for scenario in scenarios:
                match = next((row for row in summaries if row["policy_id"] == policy and row["scenario_id"] == scenario), None)
                values.append(float(match[metric]) if match and match[metric] not in ("", None) else float("nan"))
            ax.plot(scenarios, values, marker="o", label=policy)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Scenario")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.autofmt_xdate(rotation=30)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result_path = Path(args.results)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(result_path)
    summaries = summarize(rows)
    write_csv(output / "summary.csv", SUMMARY_FIELDS, summaries)
    paired = paired_comparisons(rows)
    write_csv(output / "paired_model_differences.csv", PAIRED_FIELDS, paired)
    with open(output / "summary.json", "w") as stream:
        json.dump({"rows": summaries, "paired_model_differences": paired, "source": str(result_path)}, stream, indent=2)
    plot_summaries(summaries, output)
    print(f"[SUMMARY] wrote {output / 'summary.csv'} groups={len(summaries)}")


if __name__ == "__main__":
    main()
