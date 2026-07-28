#!/usr/bin/env python3
"""Append or replace one complete policy summary without changing other rows."""

import argparse
import csv
from pathlib import Path

from summarize_results import SUMMARY_FIELDS, read_csv, scenario_sort_key, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--results", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--output-policy")
    parser.add_argument("--expected-total", type=int, default=500)
    args = parser.parse_args()

    result_rows = read_csv(args.results)
    policy_rows = [row for row in result_rows if row.get("policy_id") == args.policy]
    if len(policy_rows) != args.expected_total:
        raise ValueError(
            f"{args.policy} has {len(policy_rows)}/{args.expected_total} result rows"
        )
    case_ids = [row["case_id"] for row in policy_rows]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"{args.policy} contains duplicate case IDs")

    output_policy = args.output_policy or args.policy
    new_rows = summarize(policy_rows)
    for row in new_rows:
        row["policy_id"] = output_policy
    summary_path = Path(args.summary)
    existing_rows = read_csv(summary_path)
    existing_policy_order = []
    for row in existing_rows:
        if row["policy_id"] not in existing_policy_order:
            existing_policy_order.append(row["policy_id"])
    policy_order = existing_policy_order + (
        [] if output_policy in existing_policy_order else [output_policy]
    )
    order_index = {policy: index for index, policy in enumerate(policy_order)}

    merged = [row for row in existing_rows if row["policy_id"] != output_policy]
    merged.extend(new_rows)
    merged.sort(
        key=lambda row: (
            order_index.get(row["policy_id"], len(order_index)),
            scenario_sort_key(row["scenario_id"]),
        )
    )
    with summary_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SUMMARY_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(merged)
    print(
        f"[SUMMARY MERGE] wrote {summary_path}: rows={len(merged)} "
        f"policy={args.policy} output_policy={output_policy}"
    )


if __name__ == "__main__":
    main()
