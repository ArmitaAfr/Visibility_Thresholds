# Visibility threshold demo

A page that replays a CLIPDraw drawing one stroke at a time, most important stroke first, and shows when the judges start recognizing the concept.

At 8, 16, 32, 64, 128 and 256 strokes the partial drawing is judged by the **same pipeline as the paper evaluation**: SigLIP2 (`google/siglip2-base-patch16-224`, ranking all 150 words), Florence-2 (`microsoft/Florence-2-large-ft` captions) and the text score. A word's **visibility threshold** is the first stroke count where SigLIP2 ranks it **#1 of 150** and keeps it #1 at every larger count. Florence's caption and the text score are shown next to it but do not decide the threshold.

## Files

| File | What it does |
|---|---|
| `index.html` | The demo page. Starts with hand-made **example** data (labelled). **Load results JSON** replays real runs. |
| `export_budgets.py` | Colab, CLIPDraw runtime. Ranks each finished drawing's strokes (leave-one-out CLIP score drop), renders the top 8…256, and saves them as ordinary image folders `b008/part1/001_word/…` plus the stroke order. |
| `build_demo_json.py` | Joins the evaluation notebook's `combined.csv` with the stroke order into the page's JSON. No ML libraries. |
| `export_for_demo.py` | Older one-step version with its own SigLIP judge. Superseded: its numbers would not match the paper's. |

## Making real data

**1. Export the stroke budgets** (GPU runtime, `clipdraw.ipynb` Sections 1–4 run, so `model` is CLIP ViT-B/32):

```python
%cd /content/Visibility_Thresholds
%run -i demo/export_budgets.py
export_budgets(["/content/drive/MyDrive/ExperimentResults/runs/prod150_v1/images/part1"], model,
               out_root="/content/drive/MyDrive/ExperimentResults/demo_inputs/prune_v1",
               limit=1)          # one word first; then limit=None
```

It prints the `EXISTING_ROOTS` setting for step 2. Rerunning skips finished words.

**2. Judge them** with `visibility_thresholds_evaluation_9F.ipynb` and `visibility-evaluation-9f.zip`, the same notebook and source that produced `prod150_eval9f_v1`. Change only these settings in section 4:

```python
RUN_ID = "prune_demo_v1"          # new ID; a new set of exported words needs another new ID
EXISTING_ROOTS = [str(DRIVE_ROOT / "demo_inputs/prune_v1")]
SOURCE_LINEAGE = None             # these are new images, not the prod150 cohort
```

Then run the stages in order, each in its own runtime: MANIFEST → GPU_EVALUATE (GPU; SigLIP2 then Florence-2) → CPU_EVALUATE (CPU; text score, then the analysis that writes `combined.csv`).

**3. Build the page data:**

```bash
python demo/build_demo_json.py \
  --combined ".../runs/prune_demo_v1/stages/<COHORT_ID>/analysis/combined.csv" \
  --strokes  "/content/drive/MyDrive/ExperimentResults/demo_inputs/prune_v1/strokes" \
  --out      "/content/drive/MyDrive/ExperimentResults/demo_inputs/prune_v1/demo_results.json"
```

Load `demo_results.json` on the page.

## Checks and caveats

- **Sanity check:** the `b256` images are the full drawings re-rendered from the SVG with the same render seed. Their SigLIP2 ranks should match `prod150_eval9f_v1` for the same words, apart from the odd borderline word (the SVG rounds colours slightly). Large or frequent differences mean the rendering differs and the thresholds can't be trusted.
- **Pruning is not the stroke sweep.** Keeping the 16 most important strokes of a 256-stroke drawing is not the same as CLIPDraw drawing with 16 strokes. Say "pruned" when presenting; the sweep (separate CLIPDraw runs per budget) is the paper measurement.
- **Writing.** Many final drawings contain the word written out. Watch the text score and Florence caption: a word that becomes "recognized" when its letters appear has been read, not seen.
- Not yet run on a GPU. Start with `limit=1`.
