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

## Pixel runs (`clipag_pixels.ipynb`)

Makes one image per word by optimizing pixels directly against the guidance model, with no strokes. Together with the stroke runs this gives a 2×2 design:

| | Strokes | Pixels |
|---|---|---|
| **CLIP** | `prod150_v1` | `clip150px_v1` |
| **CLIPAG** | `clipag150_v1` | `clipag150px_v1` |

**Recipe:** from the CLIPAG authors' `CLIPAG_T2I.ipynb` (`generate()`). Raw 224×224 pixels, AdamW at lr 0.0175, 100 steps. Each step scores 32 noisy, DiffAugmented whole-image views and 32 random 128 px crops resized to 224. Loss = mean(1 − cos) on views + 5 × mean(1 − cos) on crops, with CLIP normalization, gradient-norm clip 0.001, and dynamic thresholding at p = 0.95. The image with the lowest whole-view loss is kept. Every value is in `pixels.RECIPE` and is saved in each `experiment.json`.

**Where it differs from the authors**, so the runs stay paired:
- **Start image:** uniform noise × 0.05 with seed 0, the same for every word and both models. This is the authors' commented-out option. Their default picks a different start per prompt from a Tiny-ImageNet mixture model.
- **Prompt:** `A drawing of {word}.`, not a raw caption.
- **DiffAugment** is re-implemented from the notebook's parameters, not copied, and the resize step uses one scale per batch.

The authors' paper says 1,000 steps but their notebook uses 100. This uses the notebook.

**Expectations:** standard CLIP is expected to give noise-like images it scores highly but people can't read (the reason CLIPDraw uses strokes). If SigLIP2 still "recognizes" them, the judge is fooled by the same kind of shortcut as the generator. `CHECKPOINT_LOCATION="colab"` keeps the 1.3 GB checkpoint off Drive (it re-downloads each session). No DiffVG is needed.

**Loss values are not comparable across substrates.** Strokes record the summed negative cosine over 4 views; pixels record mean(1 − cos).

## Synthesis (`clipag_synthesis.ipynb`)

Collects every condition into `ExperimentResults/synthesis/<OUTPUT_NAME>/` and runs on CPU. It writes:
- `images/<condition>/<word>.png` and `metadata/`
- `index.csv`, with one row per image and judge results matched by PNG SHA-256
- `wide.csv`, with one row per word
- per-word and per-group image grids
- `judge_input/`, to judge every condition in one judge run
- `SUMMARY.md` and `summary.json`, with completeness, top-1 by group, and the paired comparisons from `compare.py`

Conditions that don't exist yet are skipped. Run it once to collect, judge the new runs, then run it again with their `combined.csv` files listed in `JUDGE_CSVS`.

## Limits

- **The judge is still a CLIP-family model.** SigLIP2 may favour whichever guidance model is closer to it. Human judges remain the test that matters.
- **One seed per word**, as in prod150, so seed-to-seed variation is not measured.
- **The text measures are proxies** (text score ρ = 0.57 with human ratings on an earlier set).

## Files

| File | Purpose |
|---|---|
| `../clipag_cohort.ipynb` | Stroke runs; reuses the setup, guidance, optimizer and export cells of `clipdraw_clipag.ipynb` |
| `../clipag_pixels.ipynb` | Pixel runs; reuses only the guidance-loading cell (no DiffVG) |
| `../clipag_synthesis.ipynb` | Collects all runs into one folder (CPU) |
| `cohort.py` | Word list, team split, resume checks, `experiment.json` records, atomic publish |
| `pixels.py` | Pixel optimizer (CLIPAG authors' recipe) |
| `compare.py` | Paired analysis of two judged runs |
| `synthesize.py` | Index, judge merge, grids, summary |
| `tests/` | Local tests, no GPU: `PROD150_COMBINED=<combined.csv> python3 -m unittest discover -s clipag_cohort/tests -t .` |
