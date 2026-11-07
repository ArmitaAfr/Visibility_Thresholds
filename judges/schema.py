"""Model-independent records. No model libraries are imported here."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Sequence


@dataclass(frozen=True)
class Experiment:
    image_id: str
    target_word: str
    image_path: Path
    metadata_path: Path


@dataclass(frozen=True)
class JudgeOutput:
    """Adapter output: JSON-compatible raw data and a label or caption.

    normalized_prediction is adapter-normalized text, not a correctness score.
    target_rank is computed by the pipeline from ranked_candidates, if supplied.
    """

    raw_output: Any
    normalized_prediction: str
    ranked_candidates: Sequence[str] | None = None
    candidate_scores: Sequence[float] | None = None  # Aligned with ranked_candidates.


class Judge(Protocol):
    model_name: str

    def predict(self, image_path: Path, candidates: Sequence[str]) -> JudgeOutput:
        """Evaluate one image, without access to its intended target word."""
        ...


class BatchJudge(Judge, Protocol):
    def predict_batch(self, image_paths: Sequence[Path], candidates: Sequence[str]) -> Sequence[JudgeOutput]:
        """Optional batched interface; outputs must preserve input image order."""
        ...


@dataclass(frozen=True)
class EvaluationResult:
    image_id: str
    target_word: str
    image_path: str
    metadata_path: str
    model_name: str
    raw_output: Any
    normalized_prediction: str
    target_rank: int | None
    inference_time_seconds: float
    target_score: float | None = None
