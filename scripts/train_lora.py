"""Run Qwen3.5 or Qwen3-VL LoRA training with ms-swift.

Use --smoke for a 200-row training check. Use --dry-run to inspect the
exact command without starting training or creating the output directory.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="Local model directory or model ID")
    parser.add_argument("--model-type", choices=("qwen3_5", "qwen3_vl"), default="qwen3_5")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--val-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true", help="Use 200 training and 32 validation rows for one epoch")
    parser.add_argument("--dry-run", action="store_true", help="Print command without running it")
    parser.add_argument("--epochs", type=int, default=1, help="Formal-training epochs; ignored in smoke mode")
    parser.add_argument("--image-tokens", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    args = parser.parse_args()
    if args.epochs <= 0 or args.image_tokens <= 0 or args.gradient_accumulation <= 0:
        parser.error("--epochs, --image-tokens, and --gradient-accumulation must be positive")
    return args


def swift_executable():
    python_dir = Path(sys.executable).resolve().parent
    executable = python_dir / ("Scripts/swift.exe" if os.name == "nt" else "swift")
    if not executable.is_file():
        raise FileNotFoundError(f"ms-swift executable not found beside this Python: {executable}")
    return executable


def build_command(args):
    train = args.train_jsonl.resolve()
    validation = args.val_jsonl.resolve()
    output = args.output_dir.resolve()
    if not train.is_file() or not validation.is_file():
        raise FileNotFoundError("Training and validation JSONL files must exist")
    if train == validation:
        raise ValueError("Training and validation files must be different")
    if output in (train, validation) or train.parent == output or validation.parent == output:
        raise ValueError("Output directory must be separate from input files")
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty for a new run: {output}")

    train_arg = f"{train}#200" if args.smoke else str(train)
    val_arg = f"{validation}#32" if args.smoke else str(validation)
    steps = "25" if args.smoke else "100"
    command = [
        str(swift_executable()), "sft",
        "--model", args.model,
        "--model_type", args.model_type,
        "--dataset", train_arg,
        "--val_dataset", val_arg,
        "--split_dataset_ratio", "0",
        "--dataset_shuffle", "false" if args.smoke else "true",
        "--tuner_type", "lora",
        "--torch_dtype", "bfloat16",
        "--num_train_epochs", "1" if args.smoke else str(args.epochs),
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", str(args.gradient_accumulation),
        "--learning_rate", "1e-4",
        "--lora_rank", "8",
        "--lora_alpha", "32",
        "--target_modules", "all-linear",
        "--freeze_vit", "true",
        "--freeze_aligner", "true",
        "--gradient_checkpointing", "true",
        "--max_length", "2048",
        "--save_strategy", "steps",
        "--eval_strategy", "steps",
        "--save_steps", steps,
        "--eval_steps", steps,
        "--save_total_limit", "2",
        "--logging_steps", "5",
        "--dataset_num_proc", "1",
        "--dataloader_num_workers", "0",
        "--seed", "42",
        "--output_dir", str(output),
    ]
    if args.model_type == "qwen3_5":
        command.extend([
            "--template", "qwen3_5",
            "--add_non_thinking_prefix", "true",
            "--loss_scale", "ignore_empty_think",
        ])
    environment = os.environ.copy()
    environment["PYTHONNOUSERSITE"] = "1"
    environment["IMAGE_MAX_TOKEN_NUM"] = str(args.image_tokens)
    return command, environment


def main():
    args = parse_args()
    command, environment = build_command(args)
    print(f"PYTHONNOUSERSITE={environment['PYTHONNOUSERSITE']}")
    print(f"IMAGE_MAX_TOKEN_NUM={environment['IMAGE_MAX_TOKEN_NUM']}")
    print(subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command))
    if args.dry_run:
        return
    completed = subprocess.run(command, env=environment, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
