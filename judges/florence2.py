"""Independent <CAPTION> judge for the original Microsoft Florence-2 checkpoint."""

import re
import unicodedata
from pathlib import Path
from typing import Sequence

from .revisions import is_commit_sha, resolve_model_revision
from .schema import JudgeOutput

DEFAULT_MODEL = "microsoft/Florence-2-large-ft"
TASK_PROMPT = "<CAPTION>"
SUPPORTED_TRANSFORMERS = "4.49.0"


def normalize_caption(caption):
    """Separate text cleanup only; no word matching or semantic interpretation."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", caption)).strip().lower()


def load_florence(model_id, revision, *, cache_dir, local_files_only, dtype):
    import transformers
    if transformers.__version__ != SUPPORTED_TRANSFORMERS:
        raise RuntimeError(
            f"Microsoft Florence-2 custom code is supported here with Transformers "
            f"{SUPPORTED_TRANSFORMERS}; found {transformers.__version__}. Use a separate "
            "Python 3.12 environment with judges/requirements-florence2.txt. "
            "Do not downgrade the working SigLIP2 environment. See judges/README.md."
        )
    from transformers import AutoModelForCausalLM, AutoProcessor

    resolved = resolve_model_revision(model_id, revision, cache_dir=cache_dir,
                                      local_files_only=local_files_only)
    options = dict(revision=resolved, code_revision=resolved, trust_remote_code=True,
                   cache_dir=cache_dir, local_files_only=local_files_only)
    try:
        processor = AutoProcessor.from_pretrained(model_id, **options)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="eager", **options)
        reported = getattr(model.config, "_commit_hash", None)
        if reported is not None and reported != resolved:
            raise ValueError(f"Model commit {reported!r} conflicts with pinned commit {resolved!r}.")
        return model, processor, resolved
    except Exception as exc:
        raise RuntimeError(
            f"Cannot load {model_id}@{resolved}: {exc}. Use the isolated Florence "
            "requirements/environment documented in judges/README.md; for offline loading "
            "cache both processor files and custom Python code as well as model weights."
        ) from exc


class Florence2Judge:
    def __init__(self, *, model_id=DEFAULT_MODEL, revision="main", device="auto",
                 dtype="auto", max_new_tokens=128, num_beams=3, cache_dir=None,
                 local_files_only=False, model=None, processor=None):
        import torch
        import transformers

        self._torch = torch
        if type(max_new_tokens) is not int or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if type(num_beams) is not int or num_beams < 1:
            raise ValueError("num_beams must be a positive integer.")
        if (model is None) != (processor is None):
            raise ValueError("Inject both model and processor or neither.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Only cpu and cuda devices are supported.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --device cpu.")
        if dtype == "auto":
            dtype = "float16" if self.device.type == "cuda" else "float32"
        if dtype not in {"float16", "float32"} or (self.device.type == "cpu" and dtype != "float32"):
            raise ValueError("Use float32 on CPU, or float16/float32 on CUDA.")
        self.dtype = getattr(torch, dtype)
        if model is None:
            model, processor, resolved = load_florence(
                model_id, revision, cache_dir=cache_dir, local_files_only=local_files_only,
                dtype=self.dtype)
        else:
            resolved = getattr(model.config, "_commit_hash", None)
        if not is_commit_sha(resolved):
            raise ValueError("Florence-2 requires a verified immutable 40-character model commit SHA.")
        self._model = model.to(device=self.device, dtype=self.dtype).eval()
        self._model.requires_grad_(False)
        self._processor = processor
        self.model_name = f"{model_id}@{resolved}"
        self.generation_parameters = dict(max_new_tokens=max_new_tokens, num_beams=num_beams,
                                          do_sample=False, num_return_sequences=1,
                                          use_cache=True, early_stopping=False,
                                          length_penalty=1.0, repetition_penalty=1.0)
        self.configuration = {
            "model_id": model_id, "requested_revision": revision,
            "resolved_revision": resolved, "code_revision": resolved,
            "task_prompt": TASK_PROMPT, "generation_parameters": dict(self.generation_parameters),
            "device": str(self.device), "dtype": dtype, "eval_mode": True,
            "gradients_enabled": False, "attention_implementation": "eager",
            "trust_remote_code": True, "adapter_version": 1,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "image_mode": "RGB", "processor_class": type(processor).__name__,
            "image_processor_config": processor.image_processor.to_dict(),
            "tokenizer_class": type(processor.tokenizer).__name__,
            "normalization": "NFKC, collapse whitespace, strip, lowercase",
            "generation_config": model.language_model.generation_config.to_dict(),
        }

    def checkpoint_configuration(self):
        torch = self._torch
        return {**self.configuration, "float32_matmul_precision": torch.get_float32_matmul_precision(),
                "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "cuda_version": torch.version.cuda,
                "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None}

    def predict(self, image_path: Path, candidates: Sequence[str]) -> JudgeOutput:
        return self.predict_batch([image_path], candidates)[0]

    def predict_batch(self, image_paths: Sequence[Path], candidates: Sequence[str]) -> list[JudgeOutput]:
        # candidates exists only for interface compatibility. Never inspect or pass
        # its contents, intended words, filenames or other judges' outputs to model.
        from PIL import Image

        if not image_paths:
            return []
        images = []
        try:
            for path in image_paths:
                try:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                except Exception as exc:
                    raise RuntimeError(f"Cannot read image {path}: {exc}") from exc
            torch = self._torch
            with torch.inference_mode():
                inputs = self._processor(text=[TASK_PROMPT] * len(images), images=images,
                                         return_tensors="pt", padding=True)
                # Preserve integer token IDs; only cast floating-point image tensors.
                inputs = {name: tensor.to(device=self.device,
                          dtype=self.dtype if tensor.is_floating_point() else tensor.dtype)
                          for name, tensor in inputs.items()}
                # Original Microsoft's generate builds its own joint image/text
                # embeddings. A tokenizer-only attention_mask has the wrong
                # sequence length for those embeddings; follow its documented API.
                ids = self._model.generate(input_ids=inputs["input_ids"],
                                           pixel_values=inputs["pixel_values"],
                                           **self.generation_parameters)
                decoded = self._processor.batch_decode(ids, skip_special_tokens=False,
                                                       clean_up_tokenization_spaces=False)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            if len(decoded) != len(images):
                raise ValueError("Generation returned the wrong number of captions.")
            outputs = []
            for text, image in zip(decoded, images):
                if not isinstance(text, str):
                    raise ValueError("Decoded generation must be a string.")
                parsed = self._processor.post_process_generation(
                    text, task=TASK_PROMPT, image_size=image.size)
                caption = parsed.get(TASK_PROMPT)
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError("Florence-2 returned an empty or invalid caption.")
                outputs.append(JudgeOutput(
                    {"caption": caption, "generated_text": text,
                     "configuration": {**self.configuration, "image_batch_size": len(images)}},
                    normalize_caption(caption),  # No ranking/score is fabricated.
                ))
            return outputs
        except Exception as exc:
            raise RuntimeError(f"Florence-2 captioning failed for {[str(p) for p in image_paths]}: {exc}") from exc
        finally:
            for image in images:
                image.close()
