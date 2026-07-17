#!/usr/bin/env python3
"""Summarize benchmark results for manual model comparison."""

import argparse
import csv
import json
import random
import statistics
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
POLICY_ORDER = ("single", "adjacent", "skip_one")
FACTOR_SPECS = (
    {
        "filename": "ablation_dynamic_speed.png",
        "title": "Dynamic obstacle speed sweep",
        "xlabel": "Dynamic obstacle speed (m/s)",
        "x_values": (1.0, 2.0, 3.0),
        "tick_labels": ("1", "2", "3"),
        "scenarios": ("dynamic_speed_1mps", "baseline", "dynamic_speed_3mps"),
    },
    {
        "filename": "ablation_forest_density.png",
        "title": "Forest density sweep",
        "xlabel": "Number of trees",
        "x_values": (50.0, 100.0, 150.0),
        "tick_labels": ("50", "100", "150"),
        "scenarios": ("forest_density_low", "baseline", "forest_density_high"),
    },
    {
        "filename": "ablation_flight_speed.png",
        "title": "Flight speed sweep",
        "xlabel": "Desired flight speed (m/s)",
        "x_values": (3.0, 5.0, 7.0),
        "tick_labels": ("3", "5", "7"),
        "scenarios": ("flight_speed_3", "baseline", "flight_speed_7"),
    },
)
LEGACY_PLOT_FILENAMES = (
    "success_rate_by_scenario.png",
    "collision_rate_by_scenario.png",
    "flight_time_by_scenario.png",
)


def read_csv(path):
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


def read_result_files(paths):
    """Merge per-policy result files while rejecting duplicate rollouts."""
    rows = []
    seen = set()
    for path in paths:
        for row in read_csv(path):
            key = (row["policy_id"], row["case_id"])
            if key in seen:
                raise ValueError(f"duplicate result for policy/case: {key[0]} / {key[1]}")
            seen.add(key)
            rows.append(row)
    return rows


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
            "successful_time_median": statistics.median(times) if times else "",
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


def ordered_policies(summaries):
    present = {row["policy_id"] for row in summaries}
    return [policy for policy in POLICY_ORDER if policy in present]


def build_factor_figure(summaries, spec):
    """Build one three-panel single-factor figure from summary rows."""
    import matplotlib.pyplot as plt

    lookup = {(row["policy_id"], row["scenario_id"]): row for row in summaries}
    policies = ordered_policies(summaries)
    x_values = list(spec["x_values"])
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

    for policy in policies:
        success_rates = []
        ci_lower = []
        ci_upper = []
        collision_rates = []
        flight_times = []
        for scenario_id in spec["scenarios"]:
            row = lookup.get((policy, scenario_id))
            if row is None:
                success_rates.append(float("nan"))
                ci_lower.append(0.0)
                ci_upper.append(0.0)
                collision_rates.append(float("nan"))
                flight_times.append(float("nan"))
                continue
            success = float(row["success_rate"])
            success_rates.append(success)
            ci_lower.append(success - float(row["success_ci_low"]))
            ci_upper.append(float(row["success_ci_high"]) - success)
            collision_rates.append(float(row["collision_rate"]))
            median_time = row.get("successful_time_median")
            flight_times.append(float(median_time) if median_time not in ("", None) else float("nan"))

        axes[0].errorbar(
            x_values,
            success_rates,
            yerr=[ci_lower, ci_upper],
            marker="o",
            capsize=3,
            label=policy,
        )
        axes[1].plot(x_values, collision_rates, marker="o", label=policy)
        axes[2].plot(x_values, flight_times, marker="o", label=policy)

    axes[0].set_ylabel("Success rate")
    axes[0].set_ylim(-0.05, 1.05)
    axes[0].legend()
    axes[1].set_ylabel("Collision rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[2].set_ylabel("Median successful\nflight time (s)")
    axes[2].set_xlabel(spec["xlabel"])
    axes[2].set_xticks(x_values)
    axes[2].set_xticklabels(spec["tick_labels"])
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle(spec["title"])
    fig.tight_layout()
    return fig, axes


def plot_factor_sweeps(summaries, output):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SUMMARY] matplotlib unavailable; skipped plots")
        return
    for filename in LEGACY_PLOT_FILENAMES:
        (output / filename).unlink(missing_ok=True)
    present = {row["policy_id"] for row in summaries}
    if not set(POLICY_ORDER).issubset(present):
        print("[SUMMARY] skipped ablation plots until single, adjacent, and skip_one are present")
        return
    for spec in FACTOR_SPECS:
        fig, _axes = build_factor_figure(summaries, spec)
        fig.savefig(output / spec["filename"], dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result_paths = [Path(path) for path in args.results]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_result_files(result_paths)
    summaries = summarize(rows)
    write_csv(output / "summary.csv", SUMMARY_FIELDS, summaries)
    paired = paired_comparisons(rows)
    write_csv(output / "paired_model_differences.csv", PAIRED_FIELDS, paired)
    with open(output / "summary.json", "w") as stream:
        sources = [str(path) for path in result_paths]
        json.dump({
            "rows": summaries,
            "paired_model_differences": paired,
            "source": sources[0] if len(sources) == 1 else sources,
            "sources": sources,
        }, stream, indent=2)
    plot_factor_sweeps(summaries, output)
    print(f"[SUMMARY] wrote {output / 'summary.csv'} groups={len(summaries)}")


if __name__ == "__main__":
    main()
