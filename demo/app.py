"""Local Gradio demo for Qwen3.5 spatial-relation judgments."""

import argparse
import gc
import os
import sys
from pathlib import Path
from threading import Lock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from inference import SpatialPredictor  # noqa: E402

MODES = ("Base", "LoRA")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path, help="Local Qwen3.5-2B model directory")
    parser.add_argument("--adapter-path", required=True, type=Path, help="LoRA checkpoint directory")
    parser.add_argument("--image-tokens", type=int, default=512)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args(argv)
    args.model = args.model.expanduser().resolve()
    args.adapter_path = args.adapter_path.expanduser().resolve()
    if not args.model.is_dir():
        parser.error(f"Model directory does not exist: {args.model}")
    if not args.adapter_path.is_dir():
        parser.error(f"Adapter directory does not exist: {args.adapter_path}")
    if args.image_tokens <= 0:
        parser.error("--image-tokens must be positive")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    return args


class ModelManager:
    """Keep only one Base or LoRA predictor resident on the 8 GB GPU."""

    def __init__(self, model: Path, adapter_path: Path):
        self.model = str(model)
        self.adapter_path = adapter_path
        self._predictor = None
        self._mode = None
        self._lock = Lock()

    def _release(self):
        self._predictor = None
        self._mode = None
        gc.collect()
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def predict(self, image_path: str, caption: str, mode: str):
        if mode not in MODES:
            raise ValueError(f"Unknown mode: {mode!r}")
        if not image_path or not Path(image_path).is_file():
            raise ValueError("Please upload an image.")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("Please enter a spatial-relation description.")
        with self._lock:
            if mode != self._mode:
                self._release()
                adapter = self.adapter_path if mode == "LoRA" else None
                self._predictor = SpatialPredictor(
                    model=self.model,
                    adapter_path=adapter,
                    model_type="qwen3_5",
                )
                self._mode = mode
            return self._predictor.predict(image_path, caption)


def build_demo(manager: ModelManager):
    import gradio as gr

    def judge(image_path, caption, mode):
        try:
            result = manager.predict(image_path, caption, mode)
        except (FileNotFoundError, ValueError) as exc:
            raise gr.Error(str(exc)) from exc
        prediction = result["prediction"]
        if prediction is True:
            decision = "True：描述符合图片"
        elif prediction is False:
            decision = "False：描述不符合图片"
        else:
            decision = "无法解析模型回答"
        return decision, result["raw_response"] or ""

    with gr.Blocks(title="SpatialMind", analytics_enabled=False) as demo:
        gr.Markdown("# SpatialMind\n上传图片并输入英文空间关系描述，查看模型判断。")
        image = gr.Image(type="filepath", sources=["upload"], label="图片")
        caption = gr.Textbox(
            label="空间关系描述",
            placeholder="The cat is in front of the refrigerator.",
        )
        mode = gr.Radio(choices=list(MODES), value="LoRA", label="模型")
        button = gr.Button("判断")
        decision = gr.Textbox(label="判断结果", interactive=False)
        raw_response = gr.Textbox(label="模型原始回答", interactive=False, lines=4)
        button.click(judge, inputs=[image, caption, mode], outputs=[decision, raw_response], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1)
    return demo


def main():
    args = parse_args()
    os.environ["IMAGE_MAX_TOKEN_NUM"] = str(args.image_tokens)
    manager = ModelManager(args.model, args.adapter_path)
    demo = build_demo(manager)
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=False)


if __name__ == "__main__":
    main()
