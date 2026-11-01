"""Evaluate a VSR split with the same inference and metric rules for Base and LoRA."""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference import PROMPT, SpatialPredictor  # noqa: E402
from metrics import calculate_metrics  # noqa: E402

SPLITS = ("validation", "test")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, type=Path, help="VSR Random or Zero-shot saved with datasets.save_to_disk")
    parser.add_argument("--image-dir", required=True, type=Path, help="Directory of VSR images")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B")
    parser.add_argument("--model-type", choices=("qwen3_5", "qwen3_vl"), default="qwen3_5")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path, help="One directory per experiment")
    parser.add_argument("--limit", type=int, help="Evaluate only the first N samples for a smoke test")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


def load_split(dataset_dir, split):
    from datasets import load_from_disk

    dataset = load_from_disk(str(dataset_dir))
    source_split = "dev" if split == "validation" and "validation" not in dataset and "dev" in dataset else split
    if source_split not in dataset:
        raise ValueError(f"Dataset has no {split!r} split")
    return dataset[source_split]


def run_config(args):
    return {
        "dataset_dir": str(args.dataset_dir.resolve()),
        "image_dir": str(args.image_dir.resolve()),
        "split": args.split,
        "model": args.model,
        "model_type": args.model_type,
        "adapter_path": str(args.adapter_path.resolve()) if args.adapter_path else None,
        "limit": args.limit,
        "prompt": PROMPT,
        "image_max_token_num": os.environ.get("IMAGE_MAX_TOKEN_NUM"),
        "max_tokens": 8,
        "temperature": 0,
    }


def prepare_output(output_dir, config):
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path = output_dir / "config.json"
    if config_path.exists():
        old_config = json.loads(config_path.read_text(encoding="utf-8"))
        if old_config != config:
            raise ValueError("Output directory contains a different experiment configuration")
    elif any(output_dir.iterdir()):
        raise ValueError("Output directory is not empty and has no config.json")
    else:
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_dir / "predictions.jsonl"


def read_predictions(path, rows, image_dir):
    predictions = []
    if not path.exists():
        return predictions
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            record = json.loads(line)
            if index >= len(rows):
                raise ValueError("Saved predictions exceed the requested split or limit")
            row = rows[index]
            expected = {
                "index": index,
                "image": row["image"],
                "caption": row["caption"],
                "relation": row["relation"],
                "label": bool(row["label"]),
            }
            if any(record.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Saved prediction {index} does not match source data")
            if record.get("prediction") is not None and type(record["prediction"]) is not bool:
                raise ValueError(f"Saved prediction {index} has an invalid parsed value")
            predictions.append(record)
    return predictions


def evaluate(args):
    dataset_dir = args.dataset_dir.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_dir.is_dir() or not image_dir.is_dir():
        raise FileNotFoundError("Dataset and image directories must both exist")
    if output_dir == dataset_dir or output_dir == image_dir or dataset_dir in output_dir.parents or image_dir in output_dir.parents:
        raise ValueError("Output directory must be separate from source data")
    rows = load_split(dataset_dir, args.split)
    if args.limit is not None:
        rows = rows.select(range(min(args.limit, len(rows))))
    config = run_config(args)
    predictions_path = prepare_output(output_dir, config)
    predictions = read_predictions(predictions_path, rows, image_dir)

    if len(predictions) < len(rows):
        predictor = SpatialPredictor(model=args.model, adapter_path=args.adapter_path, model_type=args.model_type)
        with predictions_path.open("a", encoding="utf-8", newline="\n") as stream:
            for index in range(len(predictions), len(rows)):
                row = rows[index]
                image = image_dir / row["image"]
                result = predictor.predict(image, row["caption"])
                record = {
                    "index": index,
                    "image": row["image"],
                    "caption": row["caption"],
                    "relation": row["relation"],
                    "label": bool(row["label"]),
                    "raw_response": result["raw_response"],
                    "prediction": result["prediction"],
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                stream.flush()
                predictions.append(record)
                if (index + 1) % 100 == 0 or index + 1 == len(rows):
                    print(f"Completed {index + 1}/{len(rows)}")

    metrics = calculate_metrics(predictions)
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.4f}; results: {output_dir}")


if __name__ == "__main__":
    evaluate(parse_args())
