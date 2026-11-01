"""Convert official VSR Random or Zero-shot splits to ms-swift multimodal JSONL."""

import argparse
import json
from collections import Counter
from pathlib import Path

EXPECTED_COUNTS = {
    "random": {"train": 7680, "validation": 1097, "test": 2195},
    "zeroshot": {"train": 3489, "dev": 340, "test": 1222},
}
PROMPT = (
    "<image>\nDoes the caption correctly describe the spatial relation in the image?\n"
    "Caption: {caption}\nAnswer True or False."
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=EXPECTED_COUNTS, default="random", help="VSR split protocol (default: random)")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="VSR dataset saved with datasets.save_to_disk")
    parser.add_argument("--image-dir", type=Path, required=True, help="Directory containing VSR image files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for JSONL and validation reports")
    return parser.parse_args()


def load_dataset(path):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install the 'datasets' package before conversion.") from exc
    return load_from_disk(str(path))


def validate_image(name, image_dir, cache):
    if not isinstance(name, str) or not name or Path(name).name != name:
        return "invalid image filename"
    if name in cache:
        return cache[name]
    path = image_dir / name
    if not path.is_file():
        cache[name] = "missing image"
        return cache[name]
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError, ImportError) as exc:
        cache[name] = f"unreadable image: {type(exc).__name__}"
        return cache[name]
    cache[name] = None
    return None


def convert(dataset, image_dir, protocol):
    converted = {}
    statistics = {}
    problems = []
    image_cache = {}
    for split, expected in EXPECTED_COUNTS[protocol].items():
        if split not in dataset:
            problems.append({"split": split, "reason": "missing split"})
            continue
        rows = dataset[split]
        labels = Counter()
        relations = Counter()
        output = []
        if len(rows) != expected:
            problems.append({"split": split, "reason": f"expected {expected} rows, found {len(rows)}"})
        for index, row in enumerate(rows):
            name = row.get("image")
            caption = row.get("caption")
            relation = row.get("relation")
            label = row.get("label")
            reason = validate_image(name, image_dir, image_cache)
            if reason:
                problems.append({"split": split, "index": index, "image": name, "reason": reason})
            if not isinstance(caption, str) or not caption.strip():
                problems.append({"split": split, "index": index, "reason": "empty or invalid caption"})
            if not isinstance(relation, str) or not relation.strip():
                problems.append({"split": split, "index": index, "reason": "empty or invalid relation"})
            if type(label) is not int or label not in (0, 1):
                problems.append({"split": split, "index": index, "reason": "label must be 0 or 1"})
            if reason or not isinstance(caption, str) or not caption.strip() or not isinstance(relation, str) or not relation.strip() or type(label) is not int or label not in (0, 1):
                continue
            labels[str(label)] += 1
            relations[relation] += 1
            output.append({
                "messages": [
                    {"role": "user", "content": PROMPT.format(caption=caption.strip())},
                    {"role": "assistant", "content": "True" if label == 1 else "False"},
                ],
                "images": [str((image_dir / name).resolve())],
            })
        output_split = "validation" if split == "dev" else split
        converted[output_split] = output
        statistics[output_split] = {
            "source_count": len(rows),
            "valid_count": len(output),
            "label_counts": dict(sorted(labels.items())),
            "relation_counts": dict(sorted(relations.items())),
        }
    return converted, statistics, problems


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    image_dir = args.image_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {dataset_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"Image directory does not exist: {image_dir}")
    if output_dir == dataset_dir or output_dir == image_dir or dataset_dir in output_dir.parents or image_dir in output_dir.parents:
        raise SystemExit("Output directory must be separate from the source dataset and image directories.")
    dataset = load_dataset(dataset_dir)
    converted, statistics, problems = convert(dataset, image_dir, args.protocol)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "statistics.json", statistics)
    write_json(output_dir / "validation_problems.json", problems)
    if problems:
        raise SystemExit(
            f"Validation failed with {len(problems)} problem(s); no training JSONL written. "
            f"See {output_dir / 'validation_problems.json'}"
        )
    for split, rows in converted.items():
        path = output_dir / f"{split}.jsonl"
        with path.open("w", encoding="utf-8", newline="\n") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{split}: {len(rows)} rows -> {path}")
    print(f"Statistics: {output_dir / 'statistics.json'}")


if __name__ == "__main__":
    main()
