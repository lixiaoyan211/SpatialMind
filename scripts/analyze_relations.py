"""Compare aligned Base and LoRA predictions by VSR relation and analysis group."""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from metrics import calculate_metrics  # noqa: E402
from relation_mapping import CATEGORY_LABELS, category_for_relation  # noqa: E402

IDENTITY_FIELDS = ("index", "image", "caption", "relation", "label")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", required=True, type=Path, help="Base evaluation directory")
    parser.add_argument("--lora-dir", required=True, type=Path, help="LoRA evaluation directory")
    parser.add_argument("--output", required=True, type=Path, help="New JSON report path")
    parser.add_argument("--min-count", type=int, default=20, help="Flag groups or relations below this count")
    args = parser.parse_args()
    if args.min_count <= 0:
        parser.error("--min-count must be positive")
    return args


def read_json(path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_predictions(path):
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError(f"Blank prediction line: {path}:{line_number}")
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(f"Prediction must be an object: {path}:{line_number}")
            if type(record.get("index")) is not int or record["index"] != line_number - 1:
                raise ValueError(f"Unexpected prediction index: {path}:{line_number}")
            if type(record.get("label")) is not bool:
                raise ValueError(f"Label must be Boolean: {path}:{line_number}")
            prediction = record.get("prediction")
            if prediction is not None and type(prediction) is not bool:
                raise ValueError(f"Prediction must be Boolean or null: {path}:{line_number}")
            relation = record.get("relation")
            if not isinstance(relation, str):
                raise ValueError(f"Relation must be text: {path}:{line_number}")
            category_for_relation(relation)
            records.append(record)
    if not records:
        raise ValueError(f"No predictions found: {path}")
    return records


def aligned_records(base_dir, lora_dir):
    base_config = read_json(base_dir / "config.json")
    lora_config = read_json(lora_dir / "config.json")
    without_adapter = lambda config: {key: value for key, value in config.items() if key != "adapter_path"}
    if without_adapter(base_config) != without_adapter(lora_config):
        raise ValueError("Base and LoRA evaluation configurations differ beyond adapter_path")
    if base_config.get("adapter_path") is not None or not lora_config.get("adapter_path"):
        raise ValueError("Expected a Base run without an adapter and a LoRA run with one")

    base = read_predictions(base_dir / "predictions.jsonl")
    lora = read_predictions(lora_dir / "predictions.jsonl")
    if len(base) != len(lora):
        raise ValueError("Base and LoRA prediction counts differ")
    for index, (base_row, lora_row) in enumerate(zip(base, lora)):
        if any(base_row.get(field) != lora_row.get(field) for field in IDENTITY_FIELDS):
            raise ValueError(f"Base and LoRA sample {index} do not match")
    return base, lora, lora_config["adapter_path"]


def comparison(base_count, lora_count, sample_count, min_count):
    return {
        "sample_count": sample_count,
        "base_correct_count": base_count,
        "lora_correct_count": lora_count,
        "base_accuracy": base_count / sample_count,
        "lora_accuracy": lora_count / sample_count,
        "delta_accuracy": (lora_count - base_count) / sample_count,
        "low_sample": sample_count < min_count,
    }


def build_report(base, lora, base_dir, lora_dir, adapter_path, min_count):
    base_metrics = calculate_metrics(base)
    lora_metrics = calculate_metrics(lora)
    categories = defaultdict(lambda: {"sample_count": 0, "base_correct": 0, "lora_correct": 0})
    relations = {}
    for relation, base_result in base_metrics["by_relation"].items():
        lora_result = lora_metrics["by_relation"][relation]
        count = base_result["sample_count"]
        if count != lora_result["sample_count"]:
            raise ValueError(f"Relation count differs: {relation}")
        category = category_for_relation(relation)
        totals = categories[category]
        totals["sample_count"] += count
        totals["base_correct"] += base_result["correct_count"]
        totals["lora_correct"] += lora_result["correct_count"]
        relations[relation] = {
            "category": category,
            **comparison(base_result["correct_count"], lora_result["correct_count"], count, min_count),
        }

    base_only = sum(a["prediction"] == a["label"] and b["prediction"] != b["label"] for a, b in zip(base, lora))
    lora_only = sum(a["prediction"] != a["label"] and b["prediction"] == b["label"] for a, b in zip(base, lora))
    return {
        "base_dir": str(base_dir),
        "lora_dir": str(lora_dir),
        "adapter_path": adapter_path,
        "min_count": min_count,
        "overall": {
            "base": {key: value for key, value in base_metrics.items() if key != "by_relation"},
            "lora": {key: value for key, value in lora_metrics.items() if key != "by_relation"},
            "delta_accuracy": lora_metrics["accuracy"] - base_metrics["accuracy"],
            "base_only_correct": base_only,
            "lora_only_correct": lora_only,
        },
        "by_category": {
            category: {
                "label": CATEGORY_LABELS[category],
                **comparison(totals["base_correct"], totals["lora_correct"], totals["sample_count"], min_count),
            }
            for category, totals in sorted(categories.items())
        },
        "by_relation": relations,
    }


def main():
    args = parse_args()
    base_dir = args.base_dir.resolve()
    lora_dir = args.lora_dir.resolve()
    output = args.output.resolve()
    if base_dir == lora_dir:
        raise ValueError("Base and LoRA directories must differ")
    if output.exists():
        raise FileExistsError(f"Report already exists: {output}")
    base, lora, adapter_path = aligned_records(base_dir, lora_dir)
    report = build_report(base, lora, base_dir, lora_dir, adapter_path, args.min_count)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(f"Compared {len(base)} aligned samples; report: {output}")


if __name__ == "__main__":
    main()
