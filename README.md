# Visibility Thresholds: The Shape a Thought Needs

## Project overview

[QMIND](https://qmind.ca/) is Queen's University's student-run artificial intelligence organization. **Visibility Thresholds: The Shape a Thought Needs** investigates whether high CLIP text-image similarity corresponds to visual concepts that people can genuinely recognize, and how that relationship changes across image substrates.

[`clipdraw.ipynb`](clipdraw.ipynb) covers the project's **vector-stroke substrate**. It uses CLIPDraw to optimize coloured Bézier curves against a text prompt, producing a simple vector drawing without training a new model. This notebook is one experimental component, not the project's complete research or human-evaluation pipeline.

## Requirements

- A Google account with access to [Google Colab](https://colab.research.google.com/).
- A Colab CUDA-compatible GPU runtime, such as a T4 or equivalent.
- Internet access during setup to install dependencies and download pinned CLIP and DiffVG sources and CLIP model weights.
- Enough uninterrupted runtime for DiffVG compilation and optimization.

When using Colab, teammates do **not** need to install CUDA or DiffVG manually on their own computers. The notebook performs its own validated DiffVG build while retaining Colab's preinstalled PyTorch/torchvision pair.

## Quick start

1. Open the notebook using [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArmitaAfr/Visibility_Thresholds/blob/feature/clipdraw-cleanup/clipdraw.ipynb). If the repository is not yet available on its default branch, open Colab, choose **File → Upload notebook**, and upload `clipdraw.ipynb`.
2. In Colab, select **Runtime → Change runtime type**, choose a **GPU** hardware accelerator, and reconnect.
3. Run the environment setup and runtime verification cells. The first setup compiles DiffVG and may take several minutes.
4. In **Experiment configuration**, start with:

   ```python
   RUN_MODE = "smoke_test"
   PROMPT = "A drawing of a banana"
   ENABLE_PREDICTION = False
   ```

5. Run the remaining cells from top to bottom. Confirm that optimization finishes and displays the final drawing.
6. For a baseline research run, change `RUN_MODE` to `"full"` and rerun from the configuration cell through export. Full mode uses 256 paths, 1,000 iterations, and four augmentations.
7. To inspect the optional CLIP labels, set `ENABLE_PREDICTION = True`; otherwise leave it off.
8. In the final cell, use the generated links to download the PNG, SVG, and JSON files from `/content/clipdraw_output` before disconnecting.

## Configuration reference

These are the user-facing settings in the notebook. Smoke-test mode overrides only the effective path, iteration, and augmentation counts.

| Parameter | Default | Meaning | Practical notes |
| --- | --- | --- | --- |
| `RUN_MODE` | `full` | Chooses the original workload or a quick validation run. | Use `smoke_test` first; use `full` for baseline experiments. |
| `ENABLE_PREDICTION` | `False` | Runs the optional final-image noun predictor. | Does not affect optimization or the exported drawing. |
| `PROMPT` | `Watercolor painting of an underwater submarine.` | Positive text target for CLIP. | Replace with the concept/style being tested. |
| `USE_NEGATIVE_PROMPTS` | `False` | Enables both negative-prompt terms. | Off preserves the original default. |
| `NEGATIVE_PROMPT_1` | `A badly drawn sketch.` | First discouraged text concept. | Used only when negative prompts are enabled. |
| `NEGATIVE_PROMPT_2` | `Many ugly, messy drawings.` | Second discouraged concept. | Used only when negative prompts are enabled. |
| `NORMALIZE_CLIP_INPUTS` | `False` | Applies CLIP channel normalization after augmentation. | `False` is the original CLIPDraw baseline; enabling it creates a different preprocessing condition. |
| `SEED` | `0` | Seeds Python, NumPy, and PyTorch. | Improves comparability but does not guarantee bitwise-identical results across runtimes. |
| `NUM_PATHS` | `256` | Number of trainable vector strokes. | Higher values increase capacity, memory use, and render time. |
| `NUM_ITERATIONS` | `1000` | Number of optimization steps. | Higher values take longer and may further refine a result. |
| `NUM_AUGMENTATIONS` | `4` | Random image views scored per step. | More views increase compute; the objective sums their losses. |
| `MAX_STROKE_WIDTH` | `50.0` | Maximum allowed trainable stroke width. | Keep fixed unless width is an experimental variable. |
| `DISPLAY_EVERY` | `10` | Number of iterations between previews. | Affects display frequency only. |

## How the algorithm works

1. The notebook randomly initializes open cubic Bézier paths with trainable control points, widths, and RGBA colours.
2. DiffVG differentiably rasterizes the vector scene onto a white 224 × 224 canvas.
3. The rendered image is transformed into several random perspective/crop views.
4. CLIP ViT-B/32 compares those views with the text prompt using cosine similarity.
5. Separate Adam optimizers update path geometry, stroke width, and colour through the renderer and CLIP image encoder.
6. After the configured iterations, the final scene is rendered and exported as raster, vector, and metadata files.

The notebook preserves the original CLIPDraw learning rates, schedule, parameter clamps, augmentation pipeline, summed similarity objective, and unnormalized-input default.

## Optional image predictor

Set `ENABLE_PREDICTION = True` in the experiment configuration cell to run the original CLIPDraw diagnostic after optimization. It compares one augmented view of the completed drawing with the original **2,343 candidate nouns**, each prompted as `a drawing of a <noun>`, and displays the five highest relative scores.

The displayed percentages are a softmax over this fixed candidate vocabulary. They are **not calibrated confidence probabilities**, do not measure whether a person will recognize the drawing, and should not be compared as absolute probabilities across different candidate sets. Prediction runs after optimization and cannot change the generated paths or loss.

This predictor is a temporary exploratory testing tool, not the project's final research evaluation system.

## Outputs

The final cell writes three files to `/content/clipdraw_output`:

| File | Purpose |
| --- | --- |
| `clipdraw_final.png` | Raster image for quick viewing, presentations, and image-based analysis. |
| `clipdraw_final.svg` | Editable vector scene for inspecting or modifying the generated paths. |
| `experiment.json` | Prompt, requested/effective settings, optimizer constants, source revisions, runtime environment, elapsed time, and final loss. |

Colab displays download links after export. Download the files before deleting or disconnecting the runtime because `/content` is temporary storage.

## Reproducibility

- `SEED` initializes Python, NumPy, and PyTorch randomness before scene creation.
- The notebook pins OpenAI CLIP commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6`, DiffVG commit `85802a71fbcc72d79cb75716eb4da4392fd09532`, and pybind11 tag `v2.13.6`.
- The setup validates its source patches and records a build signature containing the Python, PyTorch, and CUDA versions.
- Experiment metadata records the effective workload and relevant runtime versions.

Exact bitwise reproduction is not guaranteed. GPU models, CUDA/compiler versions, Colab base images, floating-point behavior, DiffVG rendering, and some CUDA operations can produce differences even with the same seed. The Python dependencies installed from package indexes are not fully version-pinned, so environment metadata should accompany reported results.

## Troubleshooting

### CUDA or GPU unavailable

Select **Runtime → Change runtime type → GPU**, save the setting, reconnect, and rerun from environment setup. The notebook intentionally rejects CPU-only execution.

### First setup takes a long time

The first run clones and compiles DiffVG. Keep the cell running and inspect its live compiler output. Rerunning setup in the same matching runtime should print that DiffVG already matches and skip recompilation.

### DiffVG compilation fails

Confirm that the runtime has a GPU and restart with **Runtime → Disconnect and delete runtime** before retrying. Keep the complete setup output: on failure, the notebook repeats the final 200 build lines so the underlying compiler/CMake error can be diagnosed.

### A Python dependency or import is missing

Rerun the setup cell. It installs and explicitly verifies CLIP/DiffVG import dependencies and reports the missing module. Avoid manually installing a different `clip` package; setup verifies that `clip` resolves to the pinned `/content/CLIP` checkout.

### Colab disconnects

Reconnect and rerun from the top. Colab runtime files, compiled artifacts, model state, and `/content/clipdraw_output` may be lost when the runtime is deleted.

### Output files are missing

Ensure optimization reached its completion message, then run the **Results and export** cell. The cell creates `/content/clipdraw_output` automatically and displays links to its files.

## Contributing

Create a focused feature branch rather than working directly on the shared default branch. Before opening a pull request:

1. Run the smoke test in a fresh Colab GPU runtime.
2. Clear notebook outputs and execution counts.
3. Describe the runtime/GPU used and whether the change affects preprocessing, optimization, rendering, or result comparability.
4. Treat changes to normalization, augmentations, similarity, learning rates, path initialization, or DiffVG rendering as research-method changes rather than cleanup.

## Attribution and licensing

This notebook is a QMIND-maintained adaptation of Kevin Frans' original [CLIPDraw](https://github.com/kvfrans/clipdraw) implementation and paper by Kevin Frans, L. B. Soros, and Olaf Witkowski. It uses [DiffVG](https://github.com/BachiLi/diffvg) and [OpenAI CLIP](https://github.com/openai/CLIP).

QMIND does not claim authorship of the original CLIPDraw algorithm. See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for upstream attribution and license terms. These files do not assign a license to the QMIND repository as a whole.
