# Phase 5: independent judge Colab workflow

Use [`judges_colab.ipynb`](../judges_colab.ipynb) in a **separate runtime** from
CLIPDraw. It reads completed team images; it does not generate or change them.
The original CLIPDraw and CLIPAG notebooks, optimization and batch runner are
unchanged. No remote upload, commit or push is automated.

## Compatibility decision, before implementation

- Colab's runtime Python changes. Its release notes currently describe a move
  to Python 3.13; selecting a past runtime is temporary, not a durable dependency
  strategy. Never assume its Python version supports legacy Florence wheels.
- Microsoft's original `Florence-2-large-ft` uses custom code, supported here
  with Transformers **4.49.0**. The newer native examples use converted community
  checkpoints, which this project does not silently substitute.
- The legacy Torch **2.6.0** / torchvision **0.21.0** pair supports Python through
  3.12. Installing it into a Python 3.13+ Colab kernel may fail, and downgrading
  only one package can break compiled torchvision operators.
- SigLIP2 and Florence must not share a Transformers installation. The notebook
  uses two isolated **managed Python 3.12.10** environments, with Transformers
  **5.3.0** for SigLIP2 and **4.49.0** for Florence. Existing kernel packages and
  CLIPDraw's runtime remain untouched. Only `uv==0.13.5`, the small environment
  launcher, is installed into the notebook kernel.
- Both environments use the matched Torch/vision **CUDA 12.4** wheels, installed
  from the official PyTorch index. These carry their CUDA user-space libraries;
  do not install a toolkit or replace Colab's NVIDIA driver. The allocated driver
  and GPU still have to support the wheels. A real allocation/matrix multiply/
  synchronization preflight runs **before model weights load**. A T4/L4/A100-class
  allocation is the conservative target. New GPU architectures or incompatible
  drivers must stop rather than trigger an unreviewed package upgrade.

References: [Colab release notes](https://colab.research.google.com/notebooks/relnotes.ipynb),
[Colab runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html),
[torchvision Python/Torch compatibility](https://github.com/pytorch/vision#installation),
[official Torch 2.6 CUDA wheel commands](https://docs.pytorch.org/get-started/previous-versions/),
[Microsoft Florence checkpoint](https://huggingface.co/microsoft/Florence-2-large-ft).

The CUDA path, Google Drive mount semantics and actual performance still need
validation in Colab. Paid access does not promise a GPU type, session duration
or a particular throughput. No GPU timing estimates have been invented locally.

## 1. Prepare the uncommitted feature-branch source locally

From PowerShell in this repository:

```powershell
Set-Location "C:\Users\orjue\Documents\Visibility_Thresholds"
python -m judges.prepare_colab --output "judge_results/visibility_judges_source.zip"
```

The exclusive-create ZIP contains the current `judges/` Python, requirements and
documentation, the new notebook, official word CSV and a SHA-256 source manifest.
It excludes generated images/results, model weights, environment directories,
git credentials and the generation notebooks. This works before committing or
pushing `feature/integrated-batch-setup`. Use a new ZIP filename for a new source
version. Do not mix changed source into an existing checkpointed production job.

Manually upload the ZIP to My Drive and open `judges_colab.ipynb` in Colab via
Upload notebook. The default archive path is
`/content/drive/MyDrive/visibility_judges_source.zip`. A normal GitHub clone of
the remote branch would currently omit unpushed work; the ZIP avoids that issue.

## 2. Team Drive layout and configuration

Each teammate needs read access to completed source folders and write access to
the judge output folder. Add shared folders as agreed, consistently named
shortcuts in My Drive so their paths are visible after `drive.mount`.

Configure all settings in notebook Section 1:

- `INPUT_ROOTS`: a list of complete part directories from one or several people.
  Include final PNG and `experiment.json` together. Exclude smoke-test folders,
  duplicate copies and unextracted ZIPs. Do not write into these folders.
- `OUTPUT_ROOT`: a Drive folder for locks, checkpoints, manifests, benchmarks,
  exports and analysis. The notebook rejects a temporary `/content` output root.
- `MODEL`: one of `siglip2` and `florence2`. Only the selected judge is loaded.
- `JOB_NAME`: a stable experiment/cohort name, e.g. `pilot35_v1`. Choose a new
  name for changed source, images, vocabulary, prompts or generation conditions.
- `SHARD_INDEX`, `NUM_SHARDS`: default 1/1. For multiple writers, everyone uses
  the **same complete frozen input cohort**, same shard count and different
  1-based indices. Assignment sorts stable image IDs and interleaves them.
- `SIGLIP2_BATCH_SIZE=2`, `FLORENCE2_BATCH_SIZE=1`: conservative CUDA defaults.
  Increase only after measuring GPU memory and performance. Changing production
  batch size is a new condition/checkpoint, not a silent resume adjustment.
- Explicit model SHAs, SigLIP2 prompt/text batch size, Florence dtype/token limit/
  beam count and official vocabulary are recorded. Florence remains `<CAPTION>`,
  `do_sample=False`, and never receives target words as model context.
- `MODEL_CACHE`: defaults to temporary `/content/hf_judge_cache` for local model
  reads. Set it to a Drive folder if persistent model caching is worth the IO
  tradeoff. Checkpoints/environment locks must stay on Drive regardless.

Run Sections 2-4: mount Drive, verify/unpack the ZIP, then build the selected
environment and test CUDA. The interpreter and installed packages live in
`/content/judge_envs/MODEL`, not the Drive mount or notebook kernel. On runtime
deletion those environments must be rebuilt.

The first setup saves a checksummed package freeze plus setup specification in
`OUTPUT_ROOT/environment_locks/MODEL.json`. Future sessions reinstall exactly
that freeze and reject differing packages/specs. Let one person create each
initial shared model lock before teammates set up concurrently. Preserve these
locks with the source ZIP. If a pinned wheel becomes unavailable, stop and
review a new environment; do not bypass the lock. Expect first-time setup and
model downloads to take time and consume several GB of local disk.

## 3. Measure a small benchmark first

Keep `MODE="benchmark"`, `RESUME=False`. Configure:

```python
BENCHMARK_TAG = "bench_t4_01"
BENCHMARK_IMAGES = 8
WARMUP_IMAGES = 1
ESTIMATE_SIZES = [150, 2000, 2500]
```

Run Section 5. It loads the selected model once, warms it, then measures a fresh
sample from your shard. Its own results/checkpoints cannot be mistaken for full
production output. Run each judge separately; the subprocess exits and releases
GPU memory before the other judge starts.

Outputs are under:

```text
OUTPUT_ROOT/MODEL/JOB_NAME/shard1of1/bench_t4_01/
    job_manifest.json
    checkpoint/manifest.json
    checkpoint/<evaluation-hash>.json
    results.json
    benchmark.json
```

`benchmark.json` records model load/text caching, warmup, sample count, inference
time, evaluation wall time including source hashing/checkpoint IO, images/second
and estimated seconds for each requested workload. Inference time is summed
from the shared evaluator's per-image timing, including image preprocessing.
Estimates are linear projections of this sample. Installation, discovery before
evaluation, exports, session interruptions and an unrepresentative mix of
caption lengths/Drive IO are not included in the projected evaluation rate.
Model-loading time is separately available for a cold-start estimate. Download
time can affect the measured model-loading value. No confidence bound or paid
Colab guarantee is implied.

Use a fresh benchmark tag for every repeat; benchmark resume is prohibited so
cached predictions cannot inflate throughput. Measure several representative
samples/repeats before budgeting 2,000+ images. This does not alter production
settings or imply that 150 images equals the full benchmark image count; the
word vocabulary currently contains 150 entries while image counts may be larger.

## 4. Production and interruption recovery

Set `MODE="run"` and `RESUME=False`, then rerun Section 1 and Section 5.
Production outputs go to the corresponding `shardNofM/full/` directory.

Every valid result is atomically checkpointed by the existing shared pipeline.
The job manifest also records ordered selected image IDs, source file hashes,
vocabulary file/content hashes and installed package versions. The existing
checkpoint identity additionally binds current image/metadata bytes, model
commit, prompts, inference flags and GPU type. Config/source/cohort/environment
changes stop resume instead of silently reusing stale predictions.

After a disconnect or runtime deletion:

1. Confirm the old runtime is no longer processing this checkpoint directory.
   A browser disconnect alone may leave it running. Never start two writers on
   the same job/shard/checkpoint, even with resume enabled.
2. Reopen the separate judge notebook, select a compatible GPU and restore the
   same archive, paths, model, job name, shard and generation settings.
3. Set `MODE="run"`, `RESUME=True`. Remount Drive and rerun verification/setup.
   Saved package locks rebuild the environment; select the same GPU type to
   match strict checkpoint validation. A different allocation requires the
   original GPU type or a reviewed new job, not relaxed validation.
4. Run Section 5. Completed images are skipped; unfinished/failed images retry.
   Model initialization/text caching still occurs, but cached images do not
   undergo inference again. If interruption happened before checkpoint creation,
   the validated job can initialize its checkpoint on resume.

Failed batched calls retry images independently and report remaining errors.
Checkpoint/storage errors stop immediately. Existing matching final exports are
retained; conflicting exports are rejected. No source image/metadata is written.
Do not remove a damaged checkpoint to force reuse: preserve it and investigate
or start a new reviewed job. File-level fsync/atomic replacement is used by the
existing system, but cloud-mounted Drive durability/rename semantics require the
real Colab test; keep archival copies of completed results.

For 2,000+ images, prefer several independent fixed shards, all with identical
conditions. They can run sequentially or on separate accounts without shared
checkpoint writers. Do not add images to an active cohort: create a new job for
new input selection. Benchmark each GPU/condition rather than assuming all
paid accounts receive the same allocation.

## 5. Merge outputs produced in separate sessions

Run notebook Section 6 after populating `SIGLIP2_EXPORTS` and `FLORENCE2_EXPORTS`
with the completed `full/results.json` paths for nonoverlapping shards. Supply
all Drive input roots for content verification. Analysis runs in the kernel
with the standard library only; it does not load either judge or require CUDA.
The two judges may be evaluated on different days, but within-model profiles
must match (model revision, vocabulary, prompt, generation/device/library config).

Locally the equivalent is:

```powershell
python -m judges.analyze --siglip2 "results/siglip_shard1.json" "results/siglip_shard2.json" --florence2 "results/florence_shard1.json" "results/florence_shard2.json" --source-root "C:\Users\orjue\Documents\part3" --output-dir "judge_results/scaled_comparison_v1"
```

Never include both a cumulative export and its resumed replacement, or mix
benchmark sample exports into production cohorts. Duplicate IDs/mixed conditions
are explicit errors; no latest-file-wins deduplication occurs. Mismatched ID
sets fail unless the existing explicit `--allow-unmatched` option is chosen,
which records exclusions and metric denominators. Captions remain free form;
text mentions remain descriptive cues, not OCR or human recognition outcomes.

## Files and validation boundaries

- `judges_colab.ipynb`: separate Drive/config/setup/benchmark/run/analysis UI.
- `colab_setup.py`: managed environments, matched CUDA wheel commands and locks.
- `requirements-siglip2-colab.txt`: critical SigLIP2 Colab version pins.
- `scaling.py`: frozen-cohort jobs, fixed-ID sharding and measured estimates.
- `prepare_colab.py`: local, checksummed upload bundle without remote writes.
- `tests/test_scaling.py` and the separate-session analysis test cover selection,
  estimates, checkpoint reuse, changed job/source rejection, lock integrity,
  CUDA command construction, source packaging and notebook syntax on CPU.

Before production, validate in an actual Colab GPU runtime: managed interpreter
and CUDA wheel setup on Linux, pinned Florence custom-code loading, both real
model outputs, memory/throughput at each batch size, Drive interruption/resume
including runtime deletion, and final cross-session/shard analysis. Local CPU
tests do not establish GPU performance or mounted-Drive durability.
