# CLIPAG cohort experiment

Re-draws the 150 prod150 nouns with **CLIPAG** guidance instead of standard CLIP, keeping every other setting fixed, then compares the two runs word by word.

## Question

In prod150 (standard CLIP guidance), the judge recognized 81% of drawings, imageability made no detectable difference, and drawings that spelled the word were recognized more often. CLIPAG is CLIP ViT-B/32 fine-tuned to be adversarially robust (Ganz & Elad, WACV 2024). Its gradients are reported to produce more image-like results, so it may lean less on writing the word. This experiment asks three things:

1. Does recognition change when CLIPAG guides the drawing?
2. Does CLIPAG write the target word less often?
3. Does an imageability effect appear, or change, under CLIPAG?

All three are tested two-sided: no direction is assumed.

## Design

Paired, within-word. Each word gets one CLIPAG drawing, compared with its existing prod150 CLIP drawing.

| Setting | prod150 (CLIP) | This run (CLIPAG) |
|---|---|---|
| Words | `sampled_words.csv` at `4717920` (SHA-256 checked) | same |
| Prompt | `A drawing of {word}.` | same |
| Seed | 0 | same |
| Paths / iterations / augmentations | 256 / 1,000 / 4 | same |
| Max stroke width | 50 | same |
| Negative prompts / input normalization | off / off | same |
| Guidance model | OpenAI CLIP ViT-B/32 | CLIPAG ViT-B/32 (official checkpoint, size and MD5 verified) |
| Judges | SigLIP2, Florence-2, text score | same, same versions |

**Why the pairs share a starting point:** with seed 0, the initial strokes depend only on Python's `random` module. They are the same for every word, in `batch_runner.py` (prod150) and in `clipdraw_clipag.ipynb` (this run). The notebook hashes each word's initial scene and checks it against the run's first word (`initial_scene.json`).

**What isn't identical:** GPU operations are not bit-for-bit reproducible, and the CLIP arm ran through `batch_runner.py`, not this notebook. The **CLIP control** (`SUBSET="control15"`: 5 words per imageability group, fixed seed) re-draws 15 words with standard CLIP through this code path, to measure how much drawings change from that alone.

## Pre-registered outcomes

Fixed before any CLIPAG results exist. `compare.py` implements exactly these.

**Primary** (Holm-corrected across the three):

| # | Outcome | Test |
|---|---|---|
| P1 | Top-1 recognition (SigLIP2 ranks the target first) | Exact McNemar; paired bootstrap 95% CI on the difference |
| P2 | Legible target word (text level ≥ 3: OCR reads at least half the word) | Exact McNemar; paired bootstrap CI |
| P3 | Imageability × guidance | Spearman ρ between Glasgow imageability and the change in SigLIP2's target score |

**Secondary** (descriptive, uncorrected): change in the judge's target score and rank (Wilcoxon), text score (Wilcoxon), Florence-2 text mentions (McNemar), top-1 by imageability group within each run, and the words that flip.

**Control rule:** if 3 or more of the 15 control words change top-1 outcome against prod150, the code path alone moves results enough to confound P1. In that case, re-draw all 150 words with CLIP through this notebook and compare CLIPAG against that instead of prod150.

**Power:** with 150 pairs and Holm's strictest threshold (α = .05/3), there is about 80% power to detect a net change in top-1 of 12 points if about 10% of words flip in both directions, or 15 points if 20% do. Smaller changes may go undetected. CLIP is already at 81%, so an increase can be at most 19 points.

## Running it

### 1. Generate (Colab, GPU)

Open `clipag_cohort.ipynb` in Colab from this branch. The notebook clones the repo at `SOURCE_REF`, so these files must be pushed first.

| Step | Settings | What it checks |
|---|---|---|
| Smoke test | `RUN_ID="clipag_smoke_1"`, `RUN_MODE="smoke_test"`, `WORD_LIMIT=2` | Setup, DiffVG build, checkpoint download, Drive output |
| CLIPAG cohort | `RUN_ID="clipag150_v1"`, `GUIDANCE_MODEL="clipag"`, `SUBSET="all"`, `RUN_MODE="full"` | The experiment |
| CLIP control | `RUN_ID="clip150ctl_v1"`, `GUIDANCE_MODEL="clip"`, `SUBSET="control15"`, `RUN_MODE="full"` | Code-path reproducibility |

To split a run across teammates, everyone uses the same `RUN_ID` and `NUM_PEOPLE`, each with their own `PERSON`. Each person's words are every `NUM_PEOPLE`-th word starting at `PERSON`, the same rule prod150 used (it used 15 people). If Colab disconnects, rerun all cells with the same settings: finished words are skipped, and unfinished ones are retried. The first session downloads the 1.31 GB CLIPAG checkpoint to `ExperimentResults/checkpoints/`; later sessions reuse it.

The settings of a run are locked in `runs/<RUN_ID>/cohort_settings.json`. Changing any of them under the same `RUN_ID` stops the notebook. Per-word timing and errors go to `runs/<RUN_ID>/logs/part<P>.csv`. A part is complete when `runs/<RUN_ID>/generation/part<P>.json` exists.

### 2. Judge (Colab, team judge notebook)

Use `visibility_thresholds_colab.ipynb` from `feature/unified-colab-integration`, with a new evaluation `RUN_ID` (for example `clipag150_eval_v1`):

```python
EXISTING_ROOTS = [str(DRIVE_ROOT / "runs/clipag150_v1/images")]
SOURCE_LINEAGE = None
```

Then run `MANIFEST` → `GPU_EVALUATE` → `CPU_EVALUATE`, with the same SigLIP2, Florence-2 and text-score revisions as `prod150_eval9f_v1`. Do the same for `clip150ctl_v1`.

### 3. Compare (anywhere)

```bash
python3 clipag_cohort/compare.py \
    <prod150_eval9f_v1>/stages/balanced150/analysis/combined.csv \
    <clipag150_eval_v1>/stages/balanced150/analysis/combined.csv \
    --labels CLIP CLIPAG --json clipag_vs_clip.json

# Control: pairs only the 15 shared words
python3 clipag_cohort/compare.py <prod150 combined.csv> <clip150ctl combined.csv> --labels prod150 CLIP-rerun
```

## Limits

- **The judge is still a CLIP-family model.** SigLIP2 may favour whichever guidance model is closer to it. Human judges remain the test that matters.
- **One seed per word**, as in prod150, so seed-to-seed variation is not measured.
- **The text measures are proxies** (text score ρ = 0.57 with human ratings on an earlier set).

## Files

| File | Purpose |
|---|---|
| `../clipag_cohort.ipynb` | Colab driver; reuses the setup, guidance, optimizer and export cells of `clipdraw_clipag.ipynb` |
| `cohort.py` | Word list, team split, resume checks, prod150-compatible `experiment.json`, atomic publish |
| `compare.py` | Paired analysis of two `combined.csv` files |
| `tests/test_cohort.py` | Local tests, no GPU: `PROD150_COMBINED=<combined.csv> python3 -m unittest discover -s clipag_cohort/tests -t .` |
