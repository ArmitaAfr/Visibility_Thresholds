# Methodology

In choosing the best model for the text score, 50 images were manually scored by a human based on how much text was in each image. Then, several models were judged by their corelation between their scores and human scores, as well as cross-correlation between models. The current agreement is ρ = 0.57 (Spearman corraletion) with the SigLIP model. The human scores are avaliable in ```./human_scores.csv```. The file ```./compare.py``` runs the ```./human_scores.csv``` against the model and outputs the Spearman correlation.

# Text score

CLIPDraw often writes the prompt word into its drawings ("Boost", "PAUSE", "Tact"). A judge model that can read could then "recognize" an image by reading the word, not by seeing what's drawn. This feature measures how much text each image contains, so the benchmark can account for it.

## What it does

For each `clipdraw_final.png`, `text_score.py` does three things:

1. **Reads the image with OCR.** It uses EasyOCR on a 3× upscale, because the strokes are tiny at 224 px. Then it checks how closely the text matches the target word, both the whole word and part of it.
2. **Asks SigLIP whether the image looks like it has text.** It compares the image with "an image with written words or letters" and "an image with no text or letters". It also checks whether the image looks more like the word written out or like a drawing of it.
3. **Combines the two.** The result is a 0–4 level (none, scribbles, letters, part of the word, full word) and a 0–1 `text_score`.

It runs after the batch, on the saved images, and needs no GPU.

## Usage

```bash
python3 text_score/text_score.py --root <part folder> [--root <another part>] --out text_scores.csv
```

This writes one row per image to `text_scores.csv`. Running `python3 text_score/compare.py` checks the scores against the hand ratings in `human_scores.csv`. 
