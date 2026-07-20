#!/usr/bin/env python3
"""Summarize benchmark results for manual model comparison."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ABLATION_RESULTS_ROOT = ROOT / "results" / "ablation"
DEFAULT_TABLE_OUTPUT = ABLATION_RESULTS_ROOT / "table"
DEFAULT_CASES = ROOT / "envtest" / "benchmark" / "manifests" / "ablation_validation_cases.csv"
INFRASTRUCTURE_REASONS = {"simulator_error", "runner_timeout", "missing_result"}
SUMMARY_FIELDS = [
    "policy_id", "scenario_id", "total", "success_count", "success_rate",
    "collision_count", "collision_rate", "successful_time_mean",
    "successful_time_median", "success_ci_low", "success_ci_high",
    "dynamic_encounter_rate", "dynamic_collision_count",
    "min_dynamic_clearance_mean", "altitude_min_mean", "altitude_max_mean",
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
        "xlabel": "Dynamic obstacle speed (m/s)",
        "xlabel_zh": "动态障碍物速度（米/秒）",
        "short_label_zh": "动态",
        "tick_labels_zh": ("1", "2", "3", "4"),
        "x_values": (1.0, 2.0, 3.0, 4.0),
        "tick_labels": ("1", "2", "3", "4"),
        "scenarios": ("dynamic_speed_1mps", "dynamic_speed_2mps", "dynamic_speed_3mps", "dynamic_speed_4mps"),
    },
    {
        "filename": "ablation_flight_speed.png",
        "xlabel": "Desired flight speed (m/s)",
        "xlabel_zh": "设定飞行速度（米/秒）",
        "short_label_zh": "飞行",
        "tick_labels_zh": ("2", "4", "6", "8"),
        "x_values": (2.0, 4.0, 6.0, 8.0),
        "tick_labels": ("2", "4", "6", "8"),
        "scenarios": ("flight_speed_2", "flight_speed_4", "flight_speed_6", "flight_speed_8"),
    },
)
LEGACY_PLOT_FILENAMES = (
    "success_rate_by_scenario.png",
    "collision_rate_by_scenario.png",
    "flight_time_by_scenario.png",
    "ablation_forest_density.png",
    "comparison_forest_density.png",
)
POLICY_DISPLAY_NAMES_ZH = {
    "single": "单帧",
    "adjacent": "相邻双帧",
    "skip_one": "隔帧双帧",
    "best_ours": "本文模型",
    "vitfly": "原版模型",
    "fastplanner": "快速规划器",
    "egoplanner": "局部规划器",
}
CHINESE_FONT_PATHS = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"),
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


def validate_result_integrity(rows, expected_case_ids, policy_order):
    """Reject incomplete, duplicate, or infrastructure-contaminated results."""
    expected = set(expected_case_ids)
    expected_policies = set(policy_order)
    factors = {
        float(row["real_time_factor"])
        for row in rows
        if row.get("real_time_factor") not in (None, "")
    }
    if len(factors) > 1:
        raise ValueError(
            f"mixed real_time_factor values in one summary: {sorted(factors)}"
        )
    by_policy = {policy: set() for policy in policy_order}
    seen = set()
    for row in rows:
        policy = row.get("policy_id", "")
        case_id = row.get("case_id", "")
        if policy not in expected_policies:
            raise ValueError(f"unexpected policy in results: {policy}")
        key = (policy, case_id)
        if key in seen:
            raise ValueError(f"duplicate result for policy/case: {policy} / {case_id}")
        seen.add(key)
        by_policy[policy].add(case_id)

        reason = row.get("termination_reason", "")
        if reason in INFRASTRUCTURE_REASONS:
            raise ValueError(
                f"infrastructure failure in results: {policy} / {case_id} / {reason}"
            )
        raw_returncode = row.get("runner_returncode", "0")
        try:
            returncode = int(raw_returncode or 0)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"invalid runner return code for {policy} / {case_id}: {raw_returncode}"
            ) from error
        if returncode not in {0, 2}:
            raise ValueError(
                f"unknown runner return code for {policy} / {case_id}: {returncode}"
            )
        if returncode == 2 and reason != "controller_error":
            raise ValueError(
                f"runner return code 2 must be controller_error: {policy} / {case_id}"
            )

    for policy, actual in by_policy.items():
        if actual != expected:
            missing = len(expected - actual)
            extra = len(actual - expected)
            raise ValueError(
                f"incomplete results for {policy}: {len(actual)}/{len(expected)} cases, "
                f"missing={missing}, extra={extra}"
            )


def latest_named_result_paths(root, policy_order):
    """Find the most recently modified result file for every named policy."""
    paths = []
    for policy in policy_order:
        candidates = [path for path in Path(root).glob(f"{policy}_*/results.csv") if path.is_file()]
        if not candidates:
            raise FileNotFoundError(f"no results found for {policy} under {root}")
        paths.append(max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path))))
    return paths


def latest_policy_result_paths(root=ABLATION_RESULTS_ROOT):
    """Find the most recently modified result file for every ablation policy."""
    return latest_named_result_paths(root, POLICY_ORDER)


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
        diagnostic_rows = [
            row for row in group
            if row.get("dynamic_encounter_count") not in ("", None)
        ]
        encounter_rate = (
            sum(float(row["dynamic_encounter_count"]) > 0 for row in diagnostic_rows)
            / len(diagnostic_rows)
            if diagnostic_rows else ""
        )
        dynamic_collision_count = (
            sum(as_int(row.get("dynamic_collision", 0)) for row in diagnostic_rows)
            if diagnostic_rows else ""
        )
        clearances = [
            float(row["min_dynamic_clearance"])
            for row in diagnostic_rows
            if row.get("min_dynamic_clearance") not in ("", None)
        ]
        altitude_mins = [
            float(row["altitude_min"])
            for row in diagnostic_rows
            if row.get("altitude_min") not in ("", None)
        ]
        altitude_maxs = [
            float(row["altitude_max"])
            for row in diagnostic_rows
            if row.get("altitude_max") not in ("", None)
        ]
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
            "dynamic_encounter_rate": encounter_rate,
            "dynamic_collision_count": dynamic_collision_count,
            "min_dynamic_clearance_mean": (
                sum(clearances) / len(clearances) if clearances else ""
            ),
            "altitude_min_mean": (
                sum(altitude_mins) / len(altitude_mins) if altitude_mins else ""
            ),
            "altitude_max_mean": (
                sum(altitude_maxs) / len(altitude_maxs) if altitude_maxs else ""
            ),
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


def ordered_policies(summaries, policy_order=POLICY_ORDER):
    present = {row["policy_id"] for row in summaries}
    return [policy for policy in policy_order if policy in present]


def chinese_font_properties():
    """Load a Chinese-capable font without relying on Matplotlib's font cache."""
    from matplotlib.font_manager import FontProperties

    font_path = next((path for path in CHINESE_FONT_PATHS if path.exists()), None)
    return FontProperties(fname=str(font_path)) if font_path else FontProperties()


def apply_font_properties(text_items, font_properties):
    """Apply a font file while preserving each text object's existing style."""
    for text_item in text_items:
        font_size = text_item.get_fontsize()
        font_weight = text_item.get_fontweight()
        font_style = text_item.get_fontstyle()
        text_item.set_fontproperties(font_properties)
        text_item.set_fontsize(font_size)
        text_item.set_fontweight(font_weight)
        text_item.set_fontstyle(font_style)


def compact_proportion_limits(values, include_zero=False, tick_step=0.04):
    """Return compact, rounded limits for proportion-valued plot data."""
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0, 1.0
    padding = tick_step / 2.0
    lower = 0.0 if include_zero else max(
        0.0,
        math.floor((min(finite) - padding) / tick_step) * tick_step,
    )
    upper = min(
        1.0,
        math.ceil((max(finite) + padding) / tick_step) * tick_step,
    )
    if upper <= lower:
        upper = min(1.0, lower + tick_step)
        lower = max(0.0, upper - tick_step)
    return lower, upper


def factor_policy_series(summaries, spec, policy_order=POLICY_ORDER):
    """Return aligned metric series for every policy in a factor sweep."""
    lookup = {(row["policy_id"], row["scenario_id"]): row for row in summaries}
    series = []
    for policy in ordered_policies(summaries, policy_order):
        success_rates = []
        collision_rates = []
        flight_times = []
        for scenario_id in spec["scenarios"]:
            row = lookup.get((policy, scenario_id))
            if row is None:
                success_rates.append(float("nan"))
                collision_rates.append(float("nan"))
                flight_times.append(float("nan"))
                continue
            success_rates.append(float(row["success_rate"]))
            collision_rates.append(float(row["collision_rate"]))
            median_time = row.get("successful_time_median")
            flight_times.append(
                float(median_time) if median_time not in ("", None) else float("nan")
            )
        series.append((policy, success_rates, collision_rates, flight_times))
    return series


def build_factor_figure(summaries, spec, policy_order=POLICY_ORDER):
    """Build one three-panel single-factor figure from summary rows."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, MultipleLocator

    x_values = list(spec["x_values"])
    chinese_font = chinese_font_properties()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), sharex=True)
    success_axis_values = []
    collision_axis_values = []

    for policy, success_rates, collision_rates, flight_times in factor_policy_series(
        summaries, spec, policy_order
    ):
        success_axis_values.extend(success_rates)
        collision_axis_values.extend(collision_rates)
        display_name = POLICY_DISPLAY_NAMES_ZH.get(policy, policy)
        axes[0].plot(x_values, success_rates, marker="o", label=display_name)
        axes[1].plot(x_values, collision_rates, marker="o", label=display_name)
        axes[2].plot(x_values, flight_times, marker="o", label=display_name)

    axes[0].set_ylim(*compact_proportion_limits(success_axis_values))
    axes[0].yaxis.set_major_locator(MultipleLocator(0.04))
    axes[1].set_ylim(*compact_proportion_limits(collision_axis_values, include_zero=True))
    axes[1].yaxis.set_major_locator(MultipleLocator(0.04))
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=5))
    panel_labels = (
        "成功率",
        "碰撞率",
        "成功飞行时间中位数（秒）",
    )
    for axis, panel_label in zip(axes, panel_labels):
        axis.set_xticks(x_values)
        axis.set_xticklabels(spec["tick_labels"])
        axis.grid(True, alpha=0.3)
        axis.text(
            0.5,
            -0.19,
            panel_label,
            ha="center",
            va="top",
            fontweight="semibold",
            transform=axis.transAxes,
        )
    fig.suptitle(spec["xlabel_zh"], y=0.98)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        bbox_to_anchor=(0.5, 0.015),
        fontsize=12,
        markerscale=1.25,
        handlelength=2.3,
    )
    apply_font_properties([
        fig._suptitle,
        *[text for axis in axes for text in axis.texts],
        *[text for axis in axes for text in axis.get_xticklabels()],
        *[text for axis in axes for text in axis.get_yticklabels()],
        *fig.legends[0].get_texts(),
    ], chinese_font)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.93))
    return fig, axes


def plot_factor_sweeps(
    summaries,
    output,
    policy_order=POLICY_ORDER,
    factor_specs=FACTOR_SPECS,
    legacy_plot_filenames=LEGACY_PLOT_FILENAMES,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[SUMMARY] matplotlib unavailable; skipped plots")
        return
    for filename in legacy_plot_filenames:
        (output / filename).unlink(missing_ok=True)
    for spec in factor_specs:
        (output / spec["filename"]).unlink(missing_ok=True)
    present = {row["policy_id"] for row in summaries}
    if not set(policy_order).issubset(present):
        missing = ", ".join(policy for policy in policy_order if policy not in present)
        print(f"[SUMMARY] skipped plots; missing policies: {missing}")
        return
    for spec in factor_specs:
        fig, _axes = build_factor_figure(summaries, spec, policy_order)
        fig.savefig(output / spec["filename"], dpi=160)
        plt.close(fig)


def write_summary_outputs(
    rows,
    result_paths,
    output,
    policy_order=POLICY_ORDER,
    factor_specs=FACTOR_SPECS,
):
    """Write common CSV/JSON statistics and the requested factor figures."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
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
    plot_factor_sweeps(
        summaries,
        output,
        policy_order=policy_order,
        factor_specs=factor_specs,
    )
    return summaries, paired


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", action="append", help="Result CSV; repeat to merge multiple policies")
    parser.add_argument("--output", default=DEFAULT_TABLE_OUTPUT)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result_paths = (
            [Path(path) for path in args.results]
            if args.results
            else latest_policy_result_paths()
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    output = Path(args.output)
    for path in result_paths:
        print(f"[SUMMARY] source: {path}")
    rows = read_result_files(result_paths)
    expected_case_ids = [row["case_id"] for row in read_csv(DEFAULT_CASES)]
    try:
        validate_result_integrity(rows, expected_case_ids, POLICY_ORDER)
    except ValueError as error:
        parser.error(str(error))
    summaries, _paired = write_summary_outputs(rows, result_paths, output)
    print(f"[SUMMARY] wrote {output / 'summary.csv'} groups={len(summaries)}")


if __name__ == "__main__":
    main()
