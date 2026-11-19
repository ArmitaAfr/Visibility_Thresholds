# Visibility threshold demo

A page for the midpoint showcase that replays a CLIPDraw drawing one stroke at a time, most important stroke first, and shows when the judges start recognizing the concept.

At 8, 16, 32, 64, 128 and 256 strokes, SigLIP (and optionally Florence) judge the partial drawing. A word's **visibility threshold** is the first stroke count where it is recognized and stays recognized at every larger count. Recognized means SigLIP ranks the target word in the top 5 of 100 words (the target plus 99 fixed distractors) and, if Florence is used, its match score is 0.50 or more.

## Files

| File | What it does |
|---|---|
| `index.html` | The demo page. Open it in a browser. It starts with hand-made **example** drawings and scores (labelled on the page); use **Load results JSON** to replay real runs. |
| `export_for_demo.py` | Colab script. Reads the `clipdraw_final.svg` files from a batch part folder, ranks each drawing's strokes by importance (leave-one-out CLIP score drop), renders the top 8…256 strokes, judges each with SigLIP, and writes the JSON the page loads. No new CLIPDraw runs needed. |

## Making real data (Colab)

1. Run notebook Sections 1–4, then:

   ```python
   %cd /content/Visibility_Thresholds
   %run -i batch/load_words.py
   %run -i demo/export_for_demo.py

   words = load_words("/content/Visibility_Thresholds/dataset/processed/sampled_words.csv")
   export_part("/content/drive/MyDrive/clipdraw_results/part1", model, words,
               out_path="/content/drive/MyDrive/clipdraw_results/demo_part1.json",
               limit=10)
   ```

2. Download the JSON and load it on the page.

## Notes

- Not yet tested on a GPU runtime. Try one word first.
- Florence is optional: pass `florence_fn(pil_image, word) -> float` to add its score.
- This replays existing drawings. Pruning strokes from a finished 256-stroke drawing measures how much of that drawing is needed; a full stroke-budget sweep (separate CLIPDraw runs at each budget) is the stronger test, planned for a subset of words.
