"""Model-independent adapter orchestration and non-overwriting exports."""

import csv
import json
import math
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Iterable, Sequence

from .schema import EvaluationResult, Experiment, Judge
from .checkpoints import CheckpointStore, digest


class EvaluationFailure(RuntimeError):
    """Successful rows are checkpointed; failures remain eligible for retry."""

    def __init__(self, results, failures):
        self.results, self.failures = results, failures
        super().__init__("Evaluation failed for: " + "; ".join(f"{path}: {error}" for path, error in failures))


def evaluate(experiments: Iterable[Experiment], candidates: Sequence[str], judge: Judge,
             *, batch_size: int = 1, checkpoint_dir=None, resume=False,
             checkpoint_config=None) -> list[EvaluationResult]:
    """Time adapter calls only (not loading/export); propagate adapter failures.

    Rankings must be a complete permutation of the candidate vocabulary. Rank
    is one-based in adapter order; adapters must document their tie policy.
    Caption adapters omit rankings, leaving target_rank null. batch_size > 1
    requires an adapter predict_batch(paths, candidates) method. Timing is the
    batch wall time divided by its image count (includes preprocessing).
    Optional checkpoint_dir saves validated rows immediately. resume reuses only
    matching identities/configuration. In checkpoint mode, failed batches retry
    images individually and EvaluationFailure reports remaining failed images.
    """
    candidates = tuple(candidates)
    if resume and checkpoint_dir is None:
        raise ValueError("resume requires checkpoint_dir.")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    keys = [word.strip().lower() for word in candidates]
    if not keys or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("Candidates must be nonempty, nonblank and unique ignoring case.")
    if not isinstance(judge.model_name, str) or not judge.model_name.strip():
        raise ValueError("Adapter needs a nonblank model_name.")
    experiments = list(experiments)
    if len({e.image_id for e in experiments}) != len(experiments):
        raise ValueError("Duplicate experiment IDs supplied to evaluation.")
    for experiment in experiments:
        if not experiment.image_path.is_file():
            raise FileNotFoundError(experiment.image_path)
        if experiment.target_word.lower() not in keys:
            raise ValueError(f"Target {experiment.target_word!r} is absent from candidate vocabulary.")
    store = None
    if checkpoint_dir is not None:
        destination = Path(checkpoint_dir).resolve()
        for experiment in experiments:
            if destination == experiment.image_path.resolve().parent or destination == experiment.metadata_path.resolve().parent:
                raise ValueError("Checkpoint directory must be separate from source experiment directories.")
        if checkpoint_config is None:
            provider = getattr(judge, "checkpoint_configuration", None)
            if not callable(provider):
                raise ValueError("Checkpointing requires checkpoint_config or an adapter checkpoint_configuration().")
            checkpoint_config = provider()
        store = CheckpointStore(checkpoint_dir, {
            "model_name": judge.model_name, "adapter_configuration": checkpoint_config,
            "candidate_vocabulary_sha256": digest(candidates), "batch_size": batch_size,
            "pipeline_version": 1,
        }, resume=resume)
    results, pending, failures = {}, [], []
    for experiment in experiments:
        identity = store.identity(experiment) if store else None
        saved = store.load(identity, experiment) if store else None
        if saved is not None:
            results[experiment.image_id] = saved
        else:
            pending.append((experiment, identity))

    def predict(batch, use_batch):
        started = time.perf_counter()
        if not use_batch:
            outputs = [judge.predict(batch[0][0].image_path, candidates)]
        else:
            outputs = list(judge.predict_batch([e.image_path for e, _ in batch], candidates))
        elapsed = (time.perf_counter() - started) / len(batch)
        if len(outputs) != len(batch):
            raise ValueError("Adapter returned the wrong number of batch outputs.")
        return outputs, elapsed

    for offset in range(0, len(pending), batch_size):
        batch = pending[offset:offset + batch_size]
        try:
            outputs, elapsed = predict(batch, batch_size > 1)
        except Exception:
            if not store:
                raise
            # A failed batched call has no trusted outputs. Retry independently.
            for item in batch:
                try:
                    outputs, elapsed = predict([item], False)
                    row = _result(item[0], outputs[0], keys, judge.model_name, elapsed)
                except Exception as exc:
                    failures.append((str(item[0].image_path), str(exc)))
                    continue
                store.save(item[1], item[0], row)  # Storage errors must stop immediately.
                results[item[0].image_id] = row
            continue
        for item, output in zip(batch, outputs):
            # Persist each valid returned output even if another output is malformed.
            try:
                row = _result(item[0], output, keys, judge.model_name, elapsed)
            except Exception as exc:
                if not store:
                    raise
                failures.append((str(item[0].image_path), str(exc)))
                continue
            if store:
                store.save(item[1], item[0], row)
            results[item[0].image_id] = row
    ordered = [results[e.image_id] for e in experiments if e.image_id in results]
    if failures:
        raise EvaluationFailure(ordered, failures)
    return ordered


def _result(experiment, output, keys, model_name, elapsed):
    json.dumps(output.raw_output, allow_nan=False)
    if not isinstance(output.normalized_prediction, str):
        raise ValueError("normalized_prediction must be a string.")
    rank, target_score = None, None
    if output.ranked_candidates is not None:
        ranking = [word.strip().lower() for word in output.ranked_candidates]
        if len(ranking) != len(keys) or set(ranking) != set(keys):
            raise ValueError("Ranking must contain each candidate exactly once.")
        rank = ranking.index(experiment.target_word.lower()) + 1
    if output.candidate_scores is not None:
        scores = list(output.candidate_scores)
        if rank is None or len(scores) != len(keys) or not all(
                isinstance(s, (float, int)) and math.isfinite(s) for s in scores):
            raise ValueError("Candidate scores must be finite and aligned with the full ranking.")
        target_score = float(scores[rank - 1])
    return EvaluationResult(
        experiment.image_id, experiment.target_word, str(experiment.image_path),
        str(experiment.metadata_path), model_name, output.raw_output,
        output.normalized_prediction, rank, elapsed, target_score,
    )


def export_results(results: Iterable[EvaluationResult], destination: str | Path) -> Path:
    """Write a new .csv or .json file; refuse to overwrite any existing file.

    CSV raw_output cells contain JSON, so structured outputs remain recoverable.
    JSON null rank becomes an empty CSV field. Paths remain provenance strings.
    """
    destination = Path(destination)
    rows = [asdict(result) for result in results]
    # Validate/serialize before opening a destination, including CSV raw outputs.
    serialized = json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False)
    suffix = destination.suffix.lower()
    if suffix not in {".csv", ".json"}:
        raise ValueError("Export destination must end in .csv or .json.")
    if suffix == ".csv":
        for row in rows:
            row["raw_output"] = json.dumps(row["raw_output"], ensure_ascii=False, allow_nan=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8", newline="") as handle:
        if suffix == ".json":
            handle.write(serialized + "\n")
        else:
            writer = csv.DictWriter(handle, fieldnames=[f.name for f in fields(EvaluationResult)])
            writer.writeheader()
            writer.writerows(rows)
    return destination
