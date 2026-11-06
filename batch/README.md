# Batch runner

Runs the CLIPDraw notebook (`clipdraw.ipynb`) on every word in a CSV, such as the official stimulus list `dataset/processed/sampled_words.csv`, and saves a PNG, an editable SVG and an `experiment.json` for each word.

The drawing code is the same as notebook Sections 5–7: same optimizer, learning-rate schedule, clamps and seed handling. Each word is re-seeded with the same seed, so results don't depend on word order or on how the list is split.

## Files

| File | What it does |
|---|---|
| `load_words.py` | Reads words from a CSV (path or link). Finds the `word`/`words` column, skips blanks, removes duplicates, keeps words like "null". |
| `batch_runner.py` | `batch_run(...)`: draws every word, saves outputs, skips finished words, records errors, writes `batch_summary.csv`. |
| `results_cell.py` | After a run: joins the batch summary with the word list's scores and saves `results_with_scores.csv`. Combines all team parts automatically. |

## Time and usage limits

- A full-length word takes about **6 minutes on a Colab T4** (256 strokes, 1,000 iterations, 4 augmentations). The 150-word list is about **15 GPU-hours**.
- Colab limits are **per Google account**: free-tier time is limited and not guaranteed, and paid compute units are used at about **1.15 units/hour on a T4 while a runtime is connected, even when idle**.
- So the list is split across the team: with 4 people, each runs **~38 words (~4 hours)** on their own account, saving to one shared Drive folder.

## One-time setup: shared results folder

1. **One person** creates a Google Drive folder named `clipdraw_results` and shares it with the team as **Editor**.
2. **Everyone else**: in Drive, open **Shared with me**, right-click `clipdraw_results` → **Organize → Add shortcut → My Drive**. (Colab only sees shared folders through a shortcut in My Drive.)
3. In Colab the folder is then `/content/drive/MyDrive/clipdraw_results/`. Each person writes to their own subfolder (`part1`, `part2`, …), so nothing clashes.

Before anyone starts a full run, **agree on the settings** (prompt template, negative prompts, iterations, strokes). Every part must use the same settings or the results aren't comparable.

## Every session: integrated notebook workflow

No setup cells need to be pasted. Use the **Batch Experiment Setup** section in the original [`clipdraw.ipynb`](../clipdraw.ipynb).

1. Open the notebook in Colab and select a GPU runtime. Run original **Sections 1-4** in order, leaving generation settings at the agreed team values. Sections 5-8 are for single images only.
2. In **Batch Experiment Setup**, set `WORKFLOW = "batch"`. In that same configuration cell, set `PERSON` to your agreed 1-based number, `NUM_PEOPLE` to the team size, `RUN_MODE = "smoke_test"`, `WORDS_SOURCE` to the ordered CSV path/URL, and `OUTPUT_ROOT` to your part's destination. The default relative source is the official stimulus list; the default destination uses a generic `clipdraw_results` folder/shortcut in My Drive with separate `smoke_partN` and `partN` folders. There is no preassigned person.
3. For Drive output, set `MOUNT_DRIVE = True` and run the optional mounting cell (or mount Drive beforehand). Use the shared folder setup above. Local `/content` storage will be lost if the runtime is deleted.
4. Run **Load existing batch scripts and assign words**. It reuses a complete checkout or clones the team repository automatically, executes the existing scripts in the notebook namespace, loads the CSV, and prints `words[PERSON - 1::NUM_PEOPLE]`. Review the assignment and output path.
5. Run **Run or resume the batch**. It calls the existing `batch_run()` with `skip_existing=True` and prediction disabled. The original single-image cells are guarded and do nothing in batch mode.
6. After checking the smoke outputs, change `RUN_MODE` to `"full"` in the batch configuration and rerun the configuration, mount, loader, and execution cells. The default output expression changes to `partN`; if you customized it, use a separate destination for the full experiment.
7. Disconnect/delete the GPU runtime when finished. To use the single-image workflow instead, run Sections 3-4 again, select `WORKFLOW = "single"`, and continue with Sections 5-8. Avoid **Run all** until you have reviewed the workflow selection.

The batch configuration overrides Section 3's `RUN_MODE` only for batches; all other drawing settings still come from Sections 3-4. Keep the prompt template and generation settings consistent across teammates. No additional judge models are loaded.

## If you get disconnected

- Reconnect to a GPU runtime and rerun Sections 1-4, then the batch configuration, optional Drive mount, loader, and execution cells.
- Restore exactly the same `PERSON`, `NUM_PEOPLE`, CSV contents/order, `RUN_MODE`, `OUTPUT_ROOT`, prompt template, and generation settings. Changing the split/order changes indexed word folder names and can cause words to be regenerated.
- `skip_existing=True` skips a word when its indexed output folder contains `experiment.json`, the existing runner's completion marker. Completed PNG/SVG files in those folders are left untouched. Failed or interrupted words without that marker are retried and their partial files may be replaced.
- A fresh experiment or changed settings require a new output root. Smoke and full outputs must use separate folders.
- Drive output requires mounted Drive; the runner stops before drawing if it is not mounted. Confirm that another runtime is not still writing to your part before resuming.

## Combining everyone's results

When all parts are done, one person runs this after the integrated batch loader cell (which defines `REPO_ROOT`):

```python
from google.colab import drive
drive.mount("/content/drive")

WORDS_SOURCE = str(REPO_ROOT / "dataset/processed/sampled_words.csv")
OUTPUT_ROOT = "/content/drive/MyDrive/clipdraw_results"
results_script = REPO_ROOT / "batch/results_cell.py"
exec(compile(results_script.read_text(encoding="utf-8"), str(results_script), "exec"), globals())
```

The results cell finds every `part*` folder, combines their summaries into `batch_summary_all.csv`, joins each word with its scores from the word list (concreteness, imageability, `score_bin`, …) and saves **`clipdraw_results/results_with_scores.csv`**. Pointing `OUTPUT_ROOT` at a single `partN` folder gives results for just that part.

## Outputs

```
clipdraw_results/                     (shared Drive folder)
├── part1/
│   ├── batch_summary.csv             one row per word: status, final loss, time, prediction, error
│   ├── 001_alimony/
│   │   ├── clipdraw_final.png
│   │   ├── clipdraw_final.svg
│   │   └── experiment.json           settings, environment, full loss history, prediction (if on)
│   └── ...
├── part1.zip                         made when part 1 finishes
├── part2/ … part4/
├── batch_summary_all.csv             written by the results cell
└── results_with_scores.csv           written by the results cell
```

Folder numbers (`001_`, `002_`, …) restart in each part; the word name is in every folder name and in the summaries.

## `batch_run` options

| Option | Default | Notes |
|---|---|---|
| `words` | — | A Python list of words, or a CSV path or link. |
| `prompt_template` | `"{}"` | How each word becomes a prompt. Keep it the same for every word and every part. |
| `output_root` | `/content/clipdraw_batch` | Use Google Drive so results survive disconnects. |
| `run_mode` | Section 3 `RUN_MODE` | `"smoke_test"` or `"full"`. |
| `seed`, `num_paths`, `num_iterations`, `num_augmentations`, `max_stroke_width`, `use_negative_prompts` | Section 3 values | Override per call if needed; keep identical across parts. |
| `enable_prediction` | Section 3 `ENABLE_PREDICTION` | Adds top-5 CLIP labels and the target word's rank. Needs Section 7's noun list: run Section 7 once with `ENABLE_PREDICTION=True`. |
| `skip_existing` | `True` | Skips words that already have an `experiment.json`. |
| `show_preview` | `True` | Live drawing preview in the notebook. |
| `zip_results` | `True` | Zips the output folder at the end. |

## Notes for experiments

- Settings that change the drawings (prompt template, iterations, augmentations, strokes, negative prompts, seed) must be fixed before the real run and kept the same for every part.
- CLIPDraw sometimes writes the word itself as letters in the drawing. For recognisability studies this is a possible confound; one option is a negative prompt such as `"Written text, letters, words."`.
- Smoke-test outputs go to `smoke_partN` folders and can be deleted once setups are confirmed.
- The batch runner targets Colab. Running locally through Colab's Docker runtime under Windows/WSL2 works but needs a DiffVG workaround and was slower than a Colab T4 in testing.
