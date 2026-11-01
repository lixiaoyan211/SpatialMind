"""Export paired VSR error cases for human inspection without copying images."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from analyze_relations import aligned_records  # noqa: E402
from relation_mapping import category_for_relation  # noqa: E402

GROUPS = ("improved", "regressed", "persistent_error")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path, help="Base evaluation directory")
    parser.add_argument("--lora-dir", required=True, type=Path, help="LoRA evaluation directory")
    parser.add_argument("--output", required=True, type=Path, help="New JSON case-list path")
    parser.add_argument("--per-group", type=int, default=10, help="Maximum cases from each outcome group")
    parser.add_argument("--seed", type=int, default=42, help="Seed for repeatable category-diverse selection")
    args = parser.parse_args()
    if args.per_group <= 0:
        parser.error("--per-group must be positive")
    return args


def outcome_group(base_row, lora_row):
    base_correct = base_row["prediction"] == base_row["label"]
    lora_correct = lora_row["prediction"] == lora_row["label"]
    if not base_correct and lora_correct:
        return "improved"
    if base_correct and not lora_correct:
        return "regressed"
    if not base_correct and not lora_correct:
        return "persistent_error"
    return None


def select_diverse(candidates, limit, seed):
    """Draw reproducibly across categories before repeating a category."""
    buckets = defaultdict(list)
    for base_row, lora_row in candidates:
        buckets[category_for_relation(base_row["relation"])].append((base_row, lora_row))
    randomizer = random.Random(seed)
    for category in sorted(buckets):
        randomizer.shuffle(buckets[category])
    chosen = []
    while len(chosen) < limit and any(buckets.values()):
        for category in sorted(buckets):
            if buckets[category] and len(chosen) < limit:
                chosen.append(buckets[category].pop())
    return chosen


def case_record(group, base_row, lora_row, image_dir):
    image_path = (image_dir / base_row["image"]).resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"Selected case image does not exist: {image_path}")
    return {
        "index": base_row["index"],
        "outcome_group": group,
        "image_path": str(image_path),
        "caption": base_row["caption"],
        "relation": base_row["relation"],
        "category": category_for_relation(base_row["relation"]),
        "label": base_row["label"],
        "base_prediction": base_row["prediction"],
        "lora_prediction": lora_row["prediction"],
        "base_raw_response": base_row.get("raw_response"),
        "lora_raw_response": lora_row.get("raw_response"),
        "manual_error_type": None,
        "manual_notes": None,
    }


def build_report(base, lora, base_dir, lora_dir, image_dir, per_group, seed):
    candidates = {group: [] for group in GROUPS}
    for base_row, lora_row in zip(base, lora):
        group = outcome_group(base_row, lora_row)
        if group is not None:
            candidates[group].append((base_row, lora_row))
    selected = []
    for group in GROUPS:
        for base_row, lora_row in select_diverse(candidates[group], per_group, seed):
            selected.append(case_record(group, base_row, lora_row, image_dir))
    return {
        "base_dir": str(base_dir),
        "lora_dir": str(lora_dir),
        "dataset_sample_count": len(base),
        "selection": "Repeatable round-robin across project-defined relation categories",
        "seed": seed,
        "per_group_limit": per_group,
        "total_group_counts": {group: len(candidates[group]) for group in GROUPS},
        "selected_group_counts": {
            group: sum(case["outcome_group"] == group for case in selected)
            for group in GROUPS
        },
        "cases": selected,
    }


def main():
    args = parse_args()
    base_dir = args.base_dir.resolve()
    lora_dir = args.lora_dir.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Case list already exists: {output}")
    base, lora, _ = aligned_records(base_dir, lora_dir)
    config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    image_dir = Path(config["image_dir"]).resolve()
    report = build_report(base, lora, base_dir, lora_dir, image_dir, args.per_group, args.seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Exported {len(report['cases'])} cases: {output}")


if __name__ == "__main__":
    main()
