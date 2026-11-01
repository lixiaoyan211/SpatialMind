"""Single-image spatial-relation inference with ms-swift.

Use the same model revision, image-token setting, and decoding configuration
for Base and LoRA evaluation. IMAGE_MAX_TOKEN_NUM must be set before loading
this module's model if an image limit is needed.
"""

import re
from pathlib import Path

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
PROMPT = (
    "<image>\nDoes the caption correctly describe the spatial relation in the image?\n"
    "Caption: {caption}\nAnswer True or False."
)
_ANSWER = re.compile(r"(true|false)\.?\Z", re.IGNORECASE)
_QWEN35_ANSWER = re.compile(r"\A\s*(?:<think>.*?</think>\s*)?(true|false)\b", re.IGNORECASE | re.DOTALL)


def parse_answer(raw_response: str | None) -> bool | None:
    """Accept only an unambiguous True or False answer."""
    if not isinstance(raw_response, str):
        return None
    match = _ANSWER.fullmatch(raw_response.strip())
    if match is None:
        return None
    return match.group(1).lower() == "true"


def parse_qwen35_answer(raw_response: str | None) -> bool | None:
    """Read Qwen3.5's first Boolean answer after its optional thinking block."""
    if not isinstance(raw_response, str):
        return None
    match = _QWEN35_ANSWER.match(raw_response)
    if match is None:
        return None
    return match.group(1).lower() == "true"


class SpatialPredictor:
    """Load the model once, then predict one image-caption pair at a time."""

    def __init__(
        self,
        model: str = MODEL_ID,
        adapter_path: str | Path | None = None,
        model_type: str = "qwen3_vl",
    ):
        if not isinstance(model, str) or not model.strip():
            raise ValueError("model must be a nonempty model ID or path")
        if model_type not in ("qwen3_vl", "qwen3_5"):
            raise ValueError("model_type must be 'qwen3_vl' or 'qwen3_5'")
        from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine

        self._model_type = model_type
        self._request_type = InferRequest
        self._request_config = RequestConfig(max_tokens=8, temperature=0)
        adapters = []
        if adapter_path is not None:
            adapter = Path(adapter_path).expanduser().resolve()
            if not adapter.is_dir():
                raise FileNotFoundError(f"Adapter directory does not exist: {adapter}")
            adapters = [str(adapter)]

        if model_type == "qwen3_vl":
            kwargs = {"model_type": model_type}
            if adapters:
                kwargs["adapters"] = adapters
            self._engine = TransformersEngine(model, **kwargs)
        else:
            from swift import get_model_processor, get_template

            loaded_model, processor = get_model_processor(model, model_type=model_type)
            template = get_template(processor, template_type=model_type, enable_thinking=False)
            kwargs = {"template": template}
            if adapters:
                kwargs["adapters"] = adapters
            self._engine = TransformersEngine(loaded_model, **kwargs)

    def predict(self, image_path: str | Path, caption: str) -> dict[str, str | bool | None]:
        """Return the raw response and parsed prediction for one sample."""
        image = Path(image_path).expanduser().resolve()
        if not image.is_file():
            raise FileNotFoundError(f"Image does not exist: {image}")
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError("caption must be a nonempty string")
        request = self._request_type(
            messages=[{"role": "user", "content": PROMPT.format(caption=caption.strip())}],
            images=[str(image)],
        )
        responses = self._engine.infer([request], request_config=self._request_config)
        raw_response = responses[0].choices[0].message.content
        parser = parse_qwen35_answer if self._model_type == "qwen3_5" else parse_answer
        return {"raw_response": raw_response, "prediction": parser(raw_response)}
