#!/usr/bin/env python3
"""Combine the latest four main-comparison model results and draw factor plots."""

import argparse
from pathlib import Path

try:
    from .run_benchmark import ROOT
    from .summarize_results import (
        FACTOR_SPECS,
        latest_named_result_paths,
        read_csv,
        read_result_files,
        validate_result_integrity,
        write_summary_outputs,
    )
except ImportError:
    from run_benchmark import ROOT
    from summarize_results import (
        FACTOR_SPECS,
        latest_named_result_paths,
        read_csv,
        read_result_files,
        validate_result_integrity,
        write_summary_outputs,
    )


COMPARISON_RESULTS_ROOT = ROOT / "results" / "comparison"
DEFAULT_TABLE_OUTPUT = COMPARISON_RESULTS_ROOT / "table"
DEFAULT_CASES = ROOT / "envtest" / "benchmark" / "manifests" / "comparison_test_cases.csv"
COMPARISON_POLICY_ORDER = ("best_ours", "vitfly", "fastplanner", "egoplanner")
COMPARISON_FACTOR_SPECS = tuple(
    dict(
        spec,
        filename=spec["filename"].replace("ablation_", "comparison_"),
        title=spec["title"].replace("sweep", "main comparison"),
    )
    for spec in FACTOR_SPECS
)


def latest_comparison_result_paths(root=COMPARISON_RESULTS_ROOT):
    return latest_named_result_paths(root, COMPARISON_POLICY_ORDER)


def validate_comparison_coverage(rows, expected_case_ids=None):
    """Require all four policies to contain the same complete immutable case set."""
    expected = set(expected_case_ids or (row["case_id"] for row in read_csv(DEFAULT_CASES)))
    try:
        validate_result_integrity(rows, expected, COMPARISON_POLICY_ORDER)
    except ValueError as error:
        if str(error).startswith("incomplete results"):
            raise ValueError(str(error).replace("incomplete results", "incomplete comparison results", 1)) from error
        raise


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--results", action="append", help="Result CSV; repeat for explicit model runs")
    parser.add_argument("--output", default=DEFAULT_TABLE_OUTPUT)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result_paths = (
            [Path(path) for path in args.results]
            if args.results
            else latest_comparison_result_paths()
        )
    except FileNotFoundError as error:
        parser.error(str(error))
    for path in result_paths:
        print(f"[COMPARISON SUMMARY] source: {path}")
    rows = read_result_files(result_paths)
    try:
        validate_comparison_coverage(rows)
    except ValueError as error:
        parser.error(str(error))
    summaries, _paired = write_summary_outputs(
        rows,
        result_paths,
        args.output,
        policy_order=COMPARISON_POLICY_ORDER,
        factor_specs=COMPARISON_FACTOR_SPECS,
    )
    print(f"[COMPARISON SUMMARY] wrote {Path(args.output) / 'summary.csv'} groups={len(summaries)}")


if __name__ == "__main__":
    main()
