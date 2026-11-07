# Judge evaluation pipeline (Phases 1-5)

For independent Google Drive evaluation, benchmarks and 2,000+ image job
preparation, use [the separate Colab workflow](COLAB.md) and
[`judges_colab.ipynb`](../judges_colab.ipynb). It isolates both judge environments
from the original CLIPDraw notebook and preserves Drive checkpoints across sessions.

For reproducible comparison of **existing JSON exports**, see
[Phase 4 analysis](analysis/README.md). Run `python -m judges.analyze` to generate
a combined CSV, JSON summary and Markdown report without rerunning inference.

This package evaluates existing CLIPDraw outputs independently of generation.
The Phase 1 foundation uses only the standard library. Phase 2 adds an optional
SigLIP2 adapter using Transformers, PyTorch and Pillow. Phase 3 adds independent
Florence-2 captioning in a separate compatible environment. Importing `judges` does
not load those dependencies or any weights.
No notebook globals, CLIP, DiffVG or pandas are required.

## Architecture

| File | Responsibility |
| --- | --- |
| `discovery.py` | Read-only recursive discovery, validation reports, stable IDs and duplicate detection. |
| `vocabulary.py` | Candidate words from the official local CSV, in original order. |
| `schema.py` | Shared experiment, adapter output, evaluation result and `Judge` protocol. |
| `pipeline.py` | Invoke an injected adapter, time its calls, associate targets, compute ranks and export. |
| `checkpoints.py` | Versioned run manifest and atomic, checksummed per-evaluation records. |
| `siglip2.py` | Fixed-vocabulary SigLIP2 recognition with cached text features and batched images. |
| `run_siglip2.py` | Standalone command for existing completed output folders. |
| `requirements-siglip2.txt` | Optional inference dependencies. |
| `florence2.py`, `run_florence2.py` | Independent Microsoft Florence-2 caption adapter and CLI. |
| `requirements-florence2.txt` | Isolated Florence dependency set; do not install over SigLIP2. |
| `tests/test_foundation.py` | Temporary fixtures and mocked adapters; no real inference. |

SigLIP2 is implemented in `siglip2.py` and Florence-2 in `florence2.py`. Each adapter
will expose `model_name` (ideally including checkpoint/revision) and
`predict(image_path, candidates) -> JudgeOutput`. The target word and generation
prompt are deliberately not passed to the adapter. SigLIP2 can return raw scores,
a normalized top label and the full ordered candidate ranking. Florence-2 can
return its raw generated output and normalized caption, omitting the ranking.
Caption normalization is text cleanup, not a guessed target word or correctness
assessment. Adapters own normalization and must eventually document score and
tie semantics; the foundation does not choose model prompts or preprocessing.

## Input contract and validation

Supply one part directory, a list of part directories, or their shared parent.
Discovery recursively looks for `experiment.json` and its sibling
`clipdraw_final.png`, as written by `batch/batch_runner.py`. ZIP archives are not
read: extract them separately first. No batch summary CSV is needed.

- The runner writes metadata after image export; its presence is the completion
  marker. A JSON object with a nonblank string `word` is required. The label is
  taken only from that field, never guessed from a filename or prompt.
- The sibling PNG must exist and be nonempty. Phase 1 does not decode images or
  verify their visual integrity; a later adapter must handle corrupt image bytes.
  SVG files are not needed for image evaluation.
- Invalid roots, malformed JSON/metadata, missing/empty PNGs, PNGs without
  metadata and duplicate records produce explicit issues. Directories with no
  image or metadata artifacts are ignored. An entirely empty scan is an error.
- Default `strict=True` raises `DiscoveryError`, whose `.report` contains all
  discovered issues and valid experiments. `strict=False` returns that report,
  excludes invalid/duplicate records and retains the first valid copy. Inspect
  issues before using this mode; it must not silently become a partial study.
- `image_id` is SHA-256 over canonical metadata plus image bytes. It survives
  relocating files and JSON whitespace/key-order changes. Overlapping input
  roots and identical copied records are reported as duplicates. Different
  metadata/settings remain separate experiments even for the same target word.
  Existing metadata has no run UUID, so intentionally identical reruns cannot
  be distinguished from copied records. Any metadata or pixel change changes ID.
- Discovery does not filter smoke/full modes. Supply only the intended `partN`
  folders to avoid mixing smoke tests and research runs.

The candidate loader defaults to `dataset/processed/sampled_words.csv` relative
to the package, independent of the current working directory. It preserves
order and spelling, strips blanks, deduplicates using lowercase, and retains
literal `NA` and `null`, matching batch word cleanup. It accepts local CSVs
only. Every evaluation target must occur in the supplied vocabulary, ignoring
case; otherwise evaluation fails explicitly.

## Common results and exports

Each image/model result has these fields:

| Field | Meaning |
| --- | --- |
| `image_id` | Stable experiment/image content identifier. |
| `target_word` | Intended word read from metadata. |
| `image_path`, `metadata_path` | Resolved source paths for provenance. |
| `model_name` | Adapter-specified model/checkpoint identifier. |
| `raw_output` | JSON-compatible scores, generated text or structured output. |
| `normalized_prediction` | Adapter-normalized label or caption. |
| `target_rank` | One-based index in the complete candidate ranking, or null for captioning. |
| `target_score` | Raw compatibility score for the target, or null for adapters without scores. |
| `inference_time_seconds` | Wall time per image, excluding loading/text cache/export; batched wall time is divided by image count. |

Ranking adapters must return each candidate exactly once; ordered ties follow
the adapter's documented policy. Optional `JudgeOutput.candidate_scores` align
with that ranking; the shared pipeline calculates `target_score` without
revealing the target to the adapter. Existing adapters can omit scores and
continue to use `predict`. `evaluate(..., batch_size=N)` with N > 1 uses the
optional `BatchJudge.predict_batch` interface. Without checkpointing, failures
propagate immediately as before. Optional checkpointing adds successful-result
reuse and individual retries after failed batch calls; see below.

`export_results(results, destination)` accepts `.csv` or `.json`. JSON preserves
structured raw outputs; CSV serializes `raw_output` as a JSON cell and uses an
empty field for a null rank. Empty results still produce headers/an empty list.
Exports refuse to overwrite existing files, including source metadata. Use a
separate evaluation directory and new filenames. Discovery/evaluation never
write source images or metadata; future adapters must honor this read-only contract.

## Google Colab preparation

Phase 1 can run in a CPU Colab runtime. Open a separate notebook; the CLIPDraw
notebook and Sections 1-4 are not required. Clone/check out the feature branch
containing `judges/` (or reuse that checkout), then:

```python
from google.colab import drive
drive.mount("/content/drive")

import sys
sys.path.insert(0, "/content/Visibility_Thresholds")  # Your feature-branch checkout.
from judges import discover_experiments, load_candidates

roots = [
    "/content/drive/MyDrive/clipdraw_results/part1",
    "/content/drive/MyDrive/clipdraw_results/part2",
]  # Add the full-run parts being evaluated; omit smoke_partN folders.
report = discover_experiments(roots)
candidates = load_candidates()
print(len(report.experiments), "images;", len(candidates), "candidate words")
```

To audit broken inputs without starting evaluation:

```python
report = discover_experiments(roots, strict=False)
for issue in report.issues:
    print(issue.code, issue.path, issue.message)
```

The model-independent API remains:

```python
from judges import evaluate, export_results
# judge = ...  # An independently initialized adapter.
# results = evaluate(report.experiments, candidates, judge)
# export_results(results, "/content/drive/MyDrive/judge_results/run1.json")
# export_results(results, "/content/drive/MyDrive/judge_results/run1.csv")
```

This older example prepares discovery only. For judge execution in Colab,
use the separate [Phase 5 notebook/workflow](COLAB.md), leaving `clipdraw.ipynb`
unchanged. The standalone SigLIP2 command remains available for local runs.
Do not treat captions as candidate rankings.

## SigLIP2: standalone small evaluation

From the repository root, install the optional dependencies in your Python
environment (Python 3.10+; choose a supported CUDA PyTorch build if using a GPU):

```sh
python -m pip install -r judges/requirements-siglip2.txt
python -m judges.run_siglip2 "/path/to/clipdraw_results/part1" --limit 3 --batch-size 2 --device cpu --output "judge_results/siglip2_first3.json"
```

Replace the input path with an existing completed part folder. No example
images are generated. The command uses the complete official 150-word vocabulary
even when `--limit` selects only three images. `--limit` takes the first records
in discovery order (supplied root order, then sorted metadata paths). Discovery
strictly validates all supplied roots before applying the limit. Pass multiple
part folders as positional arguments; omit smoke folders unless testing them
intentionally. Use a new output filename each time. Choose `.csv` instead to
export CSV; the Python API can export the same results to both formats.

The first real run downloads pretrained weights and processor/tokenizer files
from Hugging Face; subsequent adapter instances use the Hugging Face disk cache.
No training or fine-tuning occurs. The default model is
`google/siglip2-base-patch16-224`. One `AutoModel` and matching `AutoProcessor`
(including its tokenizer) are loaded per adapter. Before loading weights, the
adapter resolves `config.json` through the Hub cache/download API and reads the
exact 40-character commit from its `snapshots/COMMIT/` path. The model and
processor/tokenizer are then all loaded using that immutable commit. This does
not depend on Transformers' private `model.config._commit_hash`, which may be
absent after a successful download. `--revision COMMIT_HASH` pins a run
explicitly; otherwise the resolved commit is still recorded in each result.
Conflicting model-reported commits or unverifiable snapshots stop loading.

Already downloaded files are reused from the normal Hugging Face cache. For
strictly offline loading, add `--local-files-only`; optionally pass
`--cache-dir "/path/to/huggingface/hub"` for a nondefault cache. A cached branch
reference or explicit SHA must identify a snapshot containing all required
model, tokenizer and processor files. Errors explain how to cache the revision
or select a full commit SHA. Cache directory and offline mode affect file
lookup only, not evaluation checkpoint identity.

If an earlier run failed with `Checkpointing requires a resolved model commit`,
rerun the same command after updating this code. That failure occurred before
checkpoint creation, so it does not require `--resume`. Preserve existing valid
checkpoints; use `--resume` only when a valid manifest already exists, with the
same revision argument and inference configuration.

Inference configuration:

- Fixed template: **`A drawing of a {}.`**. Text is explicitly lowercased before
  the matching processor, with padding/truncation to 64 tokens. Override with
  `--prompt-template` only for a new evaluation; keep it fixed across all images.
- Normalized text features are computed once, in chunks (`--text-batch-size`,
  default 32), and cached for the adapter lifetime. Changing vocabulary spelling,
  order, model or template requires a new adapter. No text encoder runs per image.
- Images are decoded and converted to RGB, then processed by the checkpoint's
  matching image processor. Images are encoded in batches (`--batch-size`, default
  4). Reduce to 1 for limited memory; out-of-memory errors propagate rather than
  generating placeholder predictions. The adapter's direct `predict_batch`
  method expects callers to provide an appropriately sized batch.
- `--device auto` selects CUDA when available, otherwise CPU. Explicit `cpu`
  and `cuda` are supported. Float32, eager attention, evaluation mode, frozen
  parameters and `torch.inference_mode()` are used. There are no random image
  augmentations. CUDA is synchronized before returning results. Exact equality
  across hardware/library versions is not guaranteed.
- Raw scores follow the model forward formula:
  `exp(logit_scale) * cosine(image_features, text_features) + logit_bias`.
  They are **model-specific compatibility logits, not calibrated probabilities**.
  No sigmoid or softmax is applied. Equal scores rank in original vocabulary
  order. Every candidate is ranked, including words outside the top five.
- Unreadable images, invalid encoder outputs, model failures and vocabulary
  changes raise explicit exceptions. Successful evaluations are checkpointed
  incrementally. The final CSV/JSON export is written only after all selected
  images succeed; failures are reported explicitly and remain eligible for retry.

Each result preserves Phase 1 fields and adds nullable `target_score`. For
SigLIP2, `normalized_prediction` is the lowercase top-ranked candidate and
`target_rank` is its intended word's one-based rank. `raw_output` contains:

```json
{
  "top5": [{"word": "crocodile", "score": 2.1}],
  "candidate_scores": [{"word": "crocodile", "score": 2.1}],
  "configuration": {"model_id": "google/siglip2-base-patch16-224"}
}
```

The abbreviated example shows one entry; real results contain five top entries
(or fewer for a smaller custom vocabulary) and all ranked candidates.
Configuration includes requested/resolved revisions, prompt template, vocabulary
size and hash, device, dtype, processor settings/class, tokenizer class, text
chunk size, actual image batch size and PyTorch/Transformers versions. Source
images and experiment metadata are never changed; exports refuse overwrites.

Implementation references: [Hugging Face model card](https://huggingface.co/google/siglip2-base-patch16-224),
[SigLIP2 documentation](https://huggingface.co/docs/transformers/model_doc/siglip2),
and [Transformers compatibility-logit implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/siglip/modeling_siglip.py).

## Incremental checkpoints and resume

The SigLIP2 command now checkpoints by default to `OUTPUT.checkpoint`, or to an
explicit `--checkpoint` directory. Use a directory separate from source image
folders, with only one evaluation process writing to it at a time.

Fresh run (from the repository root, after installing optional dependencies):

```sh
python -m judges.run_siglip2 "/path/to/clipdraw_results/part1" --limit 3 --batch-size 2 --device cpu --checkpoint "judge_results/siglip2_first3.checkpoint" --output "judge_results/siglip2_first3.json"
```

Resume after interruption or reported image failures:

```sh
python -m judges.run_siglip2 "/path/to/clipdraw_results/part1" --limit 3 --batch-size 2 --device cpu --checkpoint "judge_results/siglip2_first3.checkpoint" --resume --output "judge_results/siglip2_first3.json"
```

Replace the input path with your completed output folder. Keep the model revision,
vocabulary, prompt, device, batch size and other inference settings unchanged.
For exact checkpoint selection across sessions, pass the same explicit
`--revision COMMIT_HASH` on both commands. When `main` resolves to a different
model commit, resume rejects the old configuration rather than reusing it.
You can increase `--limit` or add new input folders under the same configuration;
existing matching records are reused and only missing records are evaluated.

Final exports still refuse overwrites. An interrupted run normally has no final
export, so the same `--output` works. If that export already exists, select a new
output filename while keeping the explicit checkpoint directory and `--resume`.
Choose `.csv` for CSV; both exports include cached and newly computed results in
the requested discovery order. Model loading and text caching still occur on
resume, but cached images are not passed through inference again.

Checkpoint format and identity:

- `manifest.json` stores the versioned run configuration and its hash. This
  includes model identifier/resolved checkpoint, the ordered candidate vocabulary
  hash, prompt and processor settings, dtype/device/library versions, runtime
  inference flags, adapter version and requested batch size. SigLIP2 refuses
  checkpointing if its model commit cannot be resolved. Other adapters can opt
  in using `checkpoint_configuration()` or an explicit `checkpoint_config` dict
  describing their immutable checkpoint and all prediction-affecting settings.
- Each completed evaluation has a hash-named JSON record containing its result,
  identity and checksum. Identity includes the discovered image ID, a fresh
  SHA-256 of current PNG bytes, metadata bytes when present, target word, and run
  configuration hash. Moving files preserves eligibility when content/identity
  is unchanged; returned source paths are updated to the current location.
- Changed vocabulary/order, model checkpoint, prompt or inference configuration
  rejects the checkpoint manifest: use a new directory for a new configuration.
  Changed image/metadata bytes produce a new record key and trigger reevaluation
  of that image. Old records remain available but are never silently substituted.
- Every accepted batch is saved immediately, one atomic record at a time. Writes
  use a temporary file in the same directory, flush/fsync, then `os.replace`.
  A failure between record writes leaves the already written records reusable.
  Incomplete temporary files are ignored. Atomic replacement depends on the
  backing filesystem; use reliable storage for checkpoints.
- Malformed manifests, damaged records or checksum mismatches stop resume with
  a clear error. They are never treated as valid predictions. Keep the damaged
  directory for inspection and use a new directory to recompute. An interrupted
  initial manifest write may likewise require a new directory.

Failure handling:

- A normal exception from a batch triggers individual-image retries. Successful
  retries are saved; individual failures are collected while subsequent batches
  continue. Malformed individual returned outputs do not discard valid peers.
- At the end, `EvaluationFailure` identifies failed paths/errors and exposes
  `.results` for successful rows and `.failures` for diagnostics. Failed images
  have no completion record, so `--resume` retries them. Repairing an unreadable
  image changes its content identity and causes a fresh evaluation.
- Keyboard interruption/process termination does not claim any still-running
  batch as complete. Outputs not yet returned or atomically saved are recomputed
  on resume. Model-internal partial computation cannot be recovered.
- Checkpoint write errors stop immediately. Source changes detected during
  inference also stop rather than saving a result under the wrong content hash.

Shared API (existing adapters remain usable without checkpointing):

```python
results = evaluate(experiments, candidates, judge, batch_size=2,
                   checkpoint_dir="judge_results/run1.checkpoint", resume=False)
# Later, same configuration:
results = evaluate(experiments, candidates, judge, batch_size=2,
                   checkpoint_dir="judge_results/run1.checkpoint", resume=True)
export_results(results, "judge_results/run1.json")
```

## CPU-only validation

From the repository root, with Python 3.10+:

```sh
python -m unittest discover -s judges/tests -v
```

Tests use temporary output trees and mocked ranking/caption outputs, checking
metadata associations, duplicate detection, malformed/missing inputs, vocabulary
cleanup, ranks, exports and preservation of original input bytes.
With optional SigLIP2 dependencies installed, adapter tests also run on CPU,
mock model loading, verify cached text/rankings/batching/error handling, and
compare logits with a tiny randomly initialized Transformers model. No pretrained
weights or network access are needed by tests. Adapter tests are explicitly
skipped if the optional dependencies are absent.
Checkpoint tests use only the standard library and cover interruption before
and between saves, resumed inference skipping, changed inputs/configurations,
partial failures, atomic replacement failure, integrity checks and source safety.

## Florence-2: independent caption judge

This adapter uses exactly `microsoft/Florence-2-large-ft`, never an automatically
substituted community model. The only text supplied to its matching processor
is **`<CAPTION>`** for every image. The adapter never inspects the vocabulary
argument, reads experiment metadata, or supplies target words, image filenames
or SigLIP2 predictions as model context. The shared evaluator uses metadata and
the dataset only to associate outputs/validate targets outside inference.

### Compatibility and isolated dependencies

The existing SigLIP2 environment was inspected: Python 3.14.2, PyTorch 2.10.0,
Transformers 5.3.0. Microsoft's original checkpoint uses custom Python code
through `AutoModelForCausalLM`; it is not the converted community checkpoint
used in newer native Florence-2 examples. Rather than patching Transformers or
changing the working SigLIP2 environment, this adapter supports an isolated
Python 3.12 environment with **Transformers 4.49.0**, PyTorch 2.6.0,
torchvision 0.21.0, timm and einops. Real model loading in a different
Transformers version stops with an actionable environment message.

References: [Microsoft's checkpoint and task API](https://huggingface.co/microsoft/Florence-2-large-ft),
[native Florence-2 documentation](https://huggingface.co/docs/transformers/model_doc/florence2).

Exact PowerShell setup from the repository root (uses the installed `uv` tool;
the environment is outside the repository):

```powershell
Set-Location "C:\Users\orjue\Documents\Visibility_Thresholds"
uv venv --python 3.12 "$env:LOCALAPPDATA\venvs\visibility-florence"
uv pip install --python "$env:LOCALAPPDATA\venvs\visibility-florence\Scripts\python.exe" "torch==2.6.0" "torchvision==0.21.0" --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$env:LOCALAPPDATA\venvs\visibility-florence\Scripts\python.exe" -r judges/requirements-florence2.txt
```

This setup uses CPU wheels. For CUDA, install a matching PyTorch 2.6.0 and
torchvision 0.21.0 CUDA build in this separate environment instead, then install
the remaining requirements. The adapter supports `--device auto`, `cpu` or
`cuda`; `auto` chooses CUDA when available. Dtype defaults to float16 on CUDA
and float32 on CPU. Explicit `--dtype float32` is also supported on CUDA;
CPU float16 is rejected. No quantization, Flash Attention, training, fine-tuning
or monkey-patching is used.

The loader resolves the immutable SHA via `judges/revisions.py` before loading
anything heavy. Model weights, custom code (`code_revision`), matching processor
and tokenizer all use the same SHA. `trust_remote_code=True` executes the
Microsoft repository's code at that pinned commit. Resolved revision, code
revision and loading/inference settings are recorded. Missing files or custom
code, incompatible dependencies and model-loading failures produce explicit
errors. `--cache-dir` and `--local-files-only` work as with SigLIP2; offline use
also requires the custom Python files and tokenizer/processor to be cached.

### Three-image test and resume

Run this command after the setup above. It selects the first three completed
images in the existing discovery order, from the requested folder:

```powershell
& "$env:LOCALAPPDATA\venvs\visibility-florence\Scripts\python.exe" -m judges.run_florence2 "C:\Users\orjue\Documents\part3" --limit 3 --batch-size 1 --device cpu --revision 4a12a2b54b7016a48a22037fbd62da90cd566f2a --checkpoint "judge_results/florence2_test.checkpoint" --output "judge_results/florence2_test.json"
```

That SHA was resolved from the Microsoft repository during implementation.
The first full test downloads large pretrained weights; automated tests do not.
CPU captioning may be slow. To resume an interrupted run, repeat the exact
command with `--resume` added. An existing final export is never overwritten;
choose another output filename while keeping the explicit checkpoint directory.
Use a new checkpoint directory for changed revisions, generation settings,
devices/dtypes or batch sizes. CSV output is available by choosing a `.csv`
destination. SigLIP2 checkpoints and exports remain separate.

Generation is deterministic by default: `do_sample=False`, `num_beams=3`,
`max_new_tokens=128`, `num_return_sequences=1`, `use_cache=True`,
`early_stopping=False`, `length_penalty=1.0`, `repetition_penalty=1.0`.
`--max-new-tokens` and `--num-beams` are configurable. Generation config defaults
inherited from the checkpoint's language model are also recorded. The model is
loaded once, frozen, and kept in evaluation mode; calls run under inference
mode. No random image augmentations are applied. Exact cross-device/library
equality is not guaranteed. Batch size defaults to **1**; larger batches are
supported but may consume much more memory. Integer token tensors remain
integer when image tensors are cast. CUDA is synchronized before returning;
the shared pipeline records time, including preprocessing, per image (batch
wall time divided by count for batches).

Output uses the existing schema:

- `raw_output.caption`: the exact caption returned by the standard processor's
  `<CAPTION>` extraction, preserving case, punctuation and whitespace.
- `raw_output.generated_text`: exact decoded generation including special
  tokens (`skip_special_tokens=False`, tokenization cleanup disabled), retained
  separately so the unparsed output is never lost.
- `normalized_prediction`: a separate NFKC-normalized, whitespace-collapsed,
  stripped, lowercase caption. It does not alter either raw string.
- `raw_output.configuration`: model/code SHA, task, generation parameters,
  effective dtype/device, processor configuration and library versions.
- `target_rank` and `target_score`: **null**, never guessed from a caption.

No caption-to-word scoring or accuracy calculations are implemented. Empty or
malformed captions, unreadable images and generation errors raise explicit
exceptions. The shared checkpoint system atomically saves successes, retries
failed batches individually and reports remaining failures. Only matching
image/configuration identities are skipped on resume. Original images and
metadata are read only. Colab execution is provided by the separate
[Phase 5 workflow](COLAB.md), not the generation notebook.

Validation: the pinned original config, custom model class and matching
processor were loaded in the isolated environment without model weights; a
tiny randomly initialized model using that custom class successfully generated
tokens. CPU unit tests use mocks and temporary fixtures, with no network or
pretrained-weight downloads. Run the Florence tests in its own environment:

```powershell
& "$env:LOCALAPPDATA\venvs\visibility-florence\Scripts\python.exe" -m unittest discover -s judges/tests -p test_florence2.py -v
```

Run the full suite in the existing SigLIP2 environment to verify compatibility
with the shared pipeline; injected Florence mock models do not require its
legacy runtime. The full real pretrained Florence checkpoint still requires
the three-image inference smoke test above.
