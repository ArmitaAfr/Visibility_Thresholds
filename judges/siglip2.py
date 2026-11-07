"""Reusable SigLIP2 adapter. Importing this module does not load model weights."""

import hashlib
import json
from pathlib import Path
from string import Formatter
from typing import Sequence

from .schema import JudgeOutput
from .revisions import is_commit_sha, resolve_model_revision

DEFAULT_MODEL = "google/siglip2-base-patch16-224"
DEFAULT_PROMPT = "A drawing of a {}."


class SigLIP2Judge:
    """One model and one immutable candidate vocabulary per adapter instance.

    model/processor injection is intended for offline tests. Real loading uses
    AutoModel and its matching AutoProcessor (including checkpoint tokenizer).
    """

    def __init__(self, candidates: Sequence[str], *, model_id=DEFAULT_MODEL,
                 revision="main", device="auto", prompt_template=DEFAULT_PROMPT,
                 text_batch_size=32, model=None, processor=None,
                 cache_dir=None, local_files_only=False):
        import torch

        self._torch = torch
        self._candidates = tuple(candidates)
        keys = [w.strip().lower() for w in self._candidates]
        if not keys or any(not k for k in keys) or len(set(keys)) != len(keys):
            raise ValueError("Vocabulary must be nonempty and unique ignoring case.")
        if any(w != w.strip() for w in self._candidates):
            raise ValueError("Candidate words must already be stripped.")
        fields = [name for _, name, _, _ in Formatter().parse(prompt_template) if name is not None]
        if fields != [""]:
            raise ValueError("prompt_template must contain exactly one '{}' placeholder.")
        if type(text_batch_size) is not int or text_batch_size < 1:
            raise ValueError("text_batch_size must be positive.")
        if (model is None) != (processor is None):
            raise ValueError("Supply both model and processor, or neither.")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Supported devices are cpu and cuda.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; use --device cpu.")
        if model is None:
            from transformers import AutoModel, AutoProcessor

            resolved = resolve_model_revision(model_id, revision, cache_dir=cache_dir,
                                              local_files_only=local_files_only)
            load_options = dict(revision=resolved, cache_dir=cache_dir,
                                local_files_only=local_files_only)
            model = AutoModel.from_pretrained(model_id, attn_implementation="eager", **load_options)
            reported = getattr(model.config, "_commit_hash", None)
            if reported is not None and reported != resolved:
                raise ValueError(f"Model reports commit {reported!r}, expected pinned commit {resolved!r}.")
            # AutoProcessor loads its tokenizer from this same immutable revision.
            processor = AutoProcessor.from_pretrained(model_id, **load_options)
        else:
            # Injected models are for offline tests; never claim an unverified alias
            # as a commit. Checkpointing still requires a full commit identity.
            reported = getattr(model.config, "_commit_hash", None)
            resolved = reported if is_commit_sha(reported) else None
        self._model = model.float().to(self.device).eval()
        self._model.requires_grad_(False)
        self._processor = processor
        self.model_name = f"{model_id}@{resolved or revision}"
        import transformers
        self.configuration = {
            "model_id": model_id, "requested_revision": revision,
            "resolved_revision": resolved, "prompt_template": prompt_template,
            "candidate_vocabulary_size": len(keys),
            "candidate_vocabulary_sha256": hashlib.sha256(json.dumps(
                self._candidates, ensure_ascii=False).encode()).hexdigest(),
            "device": str(self.device), "dtype": "float32", "eval_mode": True,
            "gradients_enabled": False, "attention_implementation": "eager",
            "text_batch_size": text_batch_size, "text_lowercase": True,
            "padding": "max_length", "max_length": 64, "truncation": True,
            "score_type": "scaled_cosine_logit_with_bias",
            "tie_policy": "original_candidate_order", "image_mode": "RGB",
            "processor_class": type(processor).__name__,
            "torch_version": torch.__version__, "transformers_version": transformers.__version__,
            "cuda_version": torch.version.cuda,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "image_processor_config": processor.image_processor.to_dict(),
            "tokenizer_class": type(processor.tokenizer).__name__,
        }
        # Keep only normalized text features, not intermediate encoder outputs.
        features = []
        with torch.inference_mode():
            for offset in range(0, len(keys), text_batch_size):
                prompts = [prompt_template.format(w).lower()
                           for w in self._candidates[offset:offset + text_batch_size]]
                inputs = processor(text=prompts, padding="max_length", max_length=64,
                                   truncation=True, return_tensors="pt")
                encoded = self._model.get_text_features(**self._to_device(inputs))
                features.append(self._normalize(encoded))
            self._text_features = torch.cat(features, dim=0)
            if self._text_features.shape[0] != len(keys):
                raise ValueError("Text encoder returned the wrong number of features.")
        self._sync()

    def _to_device(self, inputs):
        return {key: value.to(self.device) for key, value in inputs.items()}

    def checkpoint_configuration(self):
        """Conservative resume identity, including implementation and runtime flags."""
        if not is_commit_sha(self.configuration["resolved_revision"]):
            raise ValueError("Checkpointing requires a verified 40-character model commit SHA; "
                             "load the model through the adapter with a cached or online Hub revision.")
        torch = self._torch
        return {
            **self.configuration, "adapter_version": 1,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
        }

    def _normalize(self, features):
        # Transformers 4.x returns a Tensor; newer versions may return a pooled output.
        if not isinstance(features, self._torch.Tensor):
            features = features.pooler_output
        features = features.float()
        norms = features.norm(dim=-1, keepdim=True)
        if (features.ndim != 2 or not self._torch.isfinite(features).all()
                or not self._torch.isfinite(norms).all() or (norms == 0).any()):
            raise ValueError("Encoder produced invalid feature vectors.")
        return features / norms

    def _sync(self):
        if self.device.type == "cuda":
            self._torch.cuda.synchronize(self.device)

    def predict(self, image_path: Path, candidates: Sequence[str]) -> JudgeOutput:
        return self.predict_batch([image_path], candidates)[0]

    def predict_batch(self, image_paths: Sequence[Path], candidates: Sequence[str]) -> list[JudgeOutput]:
        from PIL import Image

        if tuple(candidates) != self._candidates:
            raise ValueError("Candidate vocabulary/order differs from the adapter's fixed vocabulary.")
        if not image_paths:
            return []
        torch = self._torch
        images = []
        try:
            for path in image_paths:
                try:
                    with Image.open(path) as image:
                        images.append(image.convert("RGB"))
                except Exception as exc:
                    raise RuntimeError(f"Cannot read image {path}: {exc}") from exc
            with torch.inference_mode():
                inputs = self._processor(images=images, return_tensors="pt")
                features = self._normalize(self._model.get_image_features(**self._to_device(inputs)))
                # Same compatibility logit formula as Hugging Face SigLIP forward.
                logits = (features @ self._text_features.T) * self._model.logit_scale.exp() + self._model.logit_bias
                if logits.shape != (len(images), len(self._candidates)) or not torch.isfinite(logits).all():
                    raise ValueError("Model produced invalid compatibility scores.")
                rows = logits.cpu().tolist()
            self._sync()
        except Exception as exc:
            raise RuntimeError(f"SigLIP2 batch failed for {[str(p) for p in image_paths]}: {exc}") from exc
        finally:
            for image in images:
                image.close()
        outputs = []
        for scores in rows:
            order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
            ranked = [self._candidates[i] for i in order]
            ranked_scores = [scores[i] for i in order]
            predictions = [{"word": w, "score": s} for w, s in zip(ranked, ranked_scores)]
            outputs.append(JudgeOutput(
                {"top5": predictions[:5], "candidate_scores": predictions,
                 "configuration": {**self.configuration, "image_batch_size": len(images)}},
                ranked[0].lower(), ranked, ranked_scores,
            ))
        return outputs
