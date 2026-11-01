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

## Running your part (Google Colab)

1. Open `clipdraw.ipynb` in Colab and select **Runtime → Change runtime type → GPU**.
2. Run notebook **Sections 1–4** (setup, runtime check, settings, CLIP and helpers). Skip Sections 5–8.
3. Add a cell to get the repo and load the batch code:

   ```python
   ![ -d /content/Visibility_Thresholds ] || git clone https://github.com/ArmitaAfr/Visibility_Thresholds.git /content/Visibility_Thresholds
   %cd /content/Visibility_Thresholds
   %run -i batch/load_words.py
   %run -i batch/batch_runner.py
   ```

   `%run -i` lets the batch code use the model and helpers from Sections 1–4. (Pasting each file into its own cell also works.)

4. Add the run cell. Set `PERSON` to your number. Run it once with `RUN_MODE = "smoke_test"` (under a second per word) to check your setup, then change to `"full"` and run it again.

   ```python
   from google.colab import drive
   drive.mount("/content/drive")

   PERSON = 1            # 1, 2, 3 or 4: each teammate uses a different number
   NUM_PEOPLE = 4

   WORDS_SOURCE = "/content/Visibility_Thresholds/dataset/processed/sampled_words.csv"
   SHARED_ROOT = "/content/drive/MyDrive/clipdraw_results"
   PROMPT_TEMPLATE = "A drawing of {}."
   RUN_MODE = "smoke_test"                                  # then "full"

   OUTPUT_ROOT = f"{SHARED_ROOT}/part{PERSON}" if RUN_MODE == "full" else f"{SHARED_ROOT}/smoke_part{PERSON}"
   words = load_words(WORDS_SOURCE)
   my_words = words[PERSON - 1::NUM_PEOPLE]                 # interleaved: everyone gets a mix of the list
   print(f"Part {PERSON}: {len(my_words)} words")

   results = batch_run(my_words, prompt_template=PROMPT_TEMPLATE,
                       run_mode=RUN_MODE, output_root=OUTPUT_ROOT)
   ```

5. When your part finishes, **remove or disconnect the runtime** so it stops using your allowance.

## If you get disconnected

- Everything finished is already saved in Drive. If one word fails, it's logged as `error` in the summary and the batch continues.
- To resume: reconnect, rerun Sections 1–4, the batch code cell and the run cell (with the same `PERSON` and `RUN_MODE`). Finished words are skipped and the run continues from the next one. Words that failed are retried.
- If `OUTPUT_ROOT` points into Drive but Drive isn't mounted, `batch_run` stops before drawing anything, so results can't end up in temporary storage.
- Before resuming, check your `partN` folder in Drive: if new word folders are still appearing, the run is still going, so don't start a second copy.

## Combining everyone's results

When all parts are done, one person runs (after steps 2–3 above, so the batch code is loaded):

```python
from google.colab import drive
drive.mount("/content/drive")

WORDS_SOURCE = "/content/Visibility_Thresholds/dataset/processed/sampled_words.csv"
OUTPUT_ROOT = "/content/drive/MyDrive/clipdraw_results"
%run -i batch/results_cell.py
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
