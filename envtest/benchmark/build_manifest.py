#!/usr/bin/env python3
"""Build immutable scene and paired benchmark case manifests."""

import argparse
import csv
import hashlib
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
VISION_ROOT = ROOT / "flightmare" / "flightpy" / "configs" / "vision"
PROFILES = {
    "dynamic_off": 0.0,
    "dynamic_collection": 1.0,
    "dynamic_high": 1.5,
}

CASE_FIELDS = [
    "case_id", "split", "scenario_id", "scene_id", "map_id", "phase_seed",
    "desired_speed", "forest_density", "tree_count", "dynamic_profile",
    "dynamic_speed_multiplier", "scene_path", "scene_hash", "evaluation_profile",
]
SCENE_FIELDS = [
    "scene_id", "map_id", "forest_density", "tree_count", "dynamic_profile",
    "scene_path", "scene_hash",
]


def load(path):
    with open(path) as stream:
        return yaml.safe_load(stream) or {}


def hash_scene(path):
    digest = hashlib.sha256()
    for child in sorted(path.rglob("*")):
        if child.is_file():
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(child.read_bytes())
    return digest.hexdigest()


def scene_path(cfg, map_id, density, profile):
    scene_id = f"map_{map_id:03d}_density_{density}_{profile}"
    path = VISION_ROOT / cfg.get("output_level", "forest_benchmark_v1") / scene_id
    if not path.is_dir():
        raise FileNotFoundError(f"missing generated scene: {path}")
    for required in ("static_obstacles.csv", "dynamic_obstacles.yaml", "astar_path.csv"):
        if not (path / required).is_file():
            raise FileNotFoundError(f"scene {scene_id} missing {required}")
    metadata_path = path / "scene_metadata.yaml"
    metadata = load(metadata_path) if metadata_path.is_file() else {}
    return scene_id, path, int(metadata.get("tree_count", 0)), hash_scene(path)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def build(cfg):
    scenario = cfg["scenario"]
    scene_rows = []
    scene_lookup = {}
    all_maps = sorted(set(scenario["map_ids"]["validation"] + scenario["map_ids"]["test"]))
    for map_id in all_maps:
        for density, tree_count in scenario["density_counts"].items():
            for profile, multiplier in PROFILES.items():
                scene_id, path, actual_count, digest = scene_path(cfg, map_id, density, profile)
                if actual_count and actual_count != int(tree_count):
                    raise ValueError(f"{scene_id} has {actual_count} trees, expected {tree_count}")
                relative = path.relative_to(VISION_ROOT).as_posix()
                scene_rows.append({
                    "scene_id": scene_id,
                    "map_id": map_id,
                    "forest_density": density,
                    "tree_count": actual_count or tree_count,
                    "dynamic_profile": profile,
                    "scene_path": relative,
                    "scene_hash": digest,
                })
                scene_lookup[(map_id, density, profile)] = (scene_id, relative, digest, actual_count or tree_count)

    def cases_for(split, maps, phases, entries):
        rows = []
        for scenario_id, speed, density, profile in entries:
            for map_id in maps:
                for phase_seed in phases:
                    scene_id, relative, digest, tree_count = scene_lookup[(map_id, density, profile)]
                    case_id = f"{split}__{scenario_id}__m{map_id:03d}__p{phase_seed}"
                    rows.append({
                        "case_id": case_id,
                        "split": split,
                        "scenario_id": scenario_id,
                        "scene_id": scene_id,
                        "map_id": map_id,
                        "phase_seed": phase_seed,
                        "desired_speed": speed,
                        "forest_density": density,
                        "tree_count": tree_count,
                        "dynamic_profile": profile,
                        "dynamic_speed_multiplier": PROFILES[profile],
                        "scene_path": relative,
                        "scene_hash": digest,
                        "evaluation_profile": "strict",
                    })
        return rows

    validation = scenario["map_ids"]["validation"]
    test = scenario["map_ids"]["test"]
    val_phases = scenario["phase_seeds"]["validation"]
    test_phases = scenario["phase_seeds"]["test"]
    ablation_entries = [
        ("dynamic_off", 5.0, "medium", "dynamic_off"),
        ("dynamic_collection", 5.0, "medium", "dynamic_collection"),
        ("dynamic_high", 5.0, "medium", "dynamic_high"),
    ]
    comparison_entries = [
        ("baseline", 5.0, "medium", "dynamic_collection"),
        ("flight_speed_3", 3.0, "medium", "dynamic_collection"),
        ("flight_speed_7", 7.0, "medium", "dynamic_collection"),
        ("forest_density_low", 5.0, "low", "dynamic_collection"),
        ("forest_density_high", 5.0, "high", "dynamic_collection"),
        ("dynamic_off", 5.0, "medium", "dynamic_off"),
        ("dynamic_high", 5.0, "medium", "dynamic_high"),
    ]
    return scene_rows, cases_for("validation", validation, val_phases, ablation_entries), cases_for("test", test, test_phases, comparison_entries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load(args.config)
    output = Path(args.output)
    scene_rows, ablation, comparison = build(cfg)
    write_csv(output / "scene_manifest.csv", SCENE_FIELDS, scene_rows)
    write_csv(output / "ablation_validation_cases.csv", CASE_FIELDS, ablation)
    write_csv(output / "comparison_test_cases.csv", CASE_FIELDS, comparison)
    print(f"[MANIFEST] scenes={len(scene_rows)} ablation_cases={len(ablation)} comparison_cases={len(comparison)}")


if __name__ == "__main__":
    main()
