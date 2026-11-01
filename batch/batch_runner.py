import csv
import gc
import os
import re
import shutil
import traceback

from IPython.display import display


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _init_scene(num_paths, canvas_width=224, canvas_height=224):
    shapes, shape_groups = [], []
    for _ in range(num_paths):
        num_segments = random.randint(1, 3)
        num_control_points = torch.zeros(num_segments, dtype=torch.int32) + 2
        p0 = (random.random(), random.random())
        points = [p0]
        for _ in range(num_segments):
            radius = 0.1
            p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
            p2 = (p1[0] + radius * (random.random() - 0.5), p1[1] + radius * (random.random() - 0.5))
            p3 = (p2[0] + radius * (random.random() - 0.5), p2[1] + radius * (random.random() - 0.5))
            points.extend([p1, p2, p3])
            p0 = p3
        points = torch.tensor(points)
        points[:, 0] *= canvas_width
        points[:, 1] *= canvas_height
        path = pydiffvg.Path(num_control_points=num_control_points, points=points,
                             stroke_width=torch.tensor(1.0), is_closed=False)
        shapes.append(path)
        shape_groups.append(pydiffvg.ShapeGroup(
            shape_ids=torch.tensor([len(shapes) - 1]), fill_color=None,
            stroke_color=torch.tensor([random.random(), random.random(), random.random(), random.random()]),
        ))
    return shapes, shape_groups


def run_clipdraw(prompt, *, seed, num_paths, num_iterations, num_augmentations,
                 max_stroke_width, use_negative_prompts, display_every=0, preview_handle=None):
    pydiffvg.set_print_timing(False)
    pydiffvg.set_use_gpu(True)
    pydiffvg.set_device(device)

    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize(prompt).to(device))

    _seed_everything(seed)
    shapes, shape_groups = _init_scene(num_paths)

    points_vars, stroke_width_vars, color_vars = [], [], []
    for path in shapes:
        path.points.requires_grad = True
        points_vars.append(path.points)
        path.stroke_width.requires_grad = True
        stroke_width_vars.append(path.stroke_width)
    for group in shape_groups:
        group.stroke_color.requires_grad = True
        color_vars.append(group.stroke_color)

    points_optimizer = torch.optim.Adam(points_vars, lr=1.0)
    width_optimizer = torch.optim.Adam(stroke_width_vars, lr=0.1)
    color_optimizer = torch.optim.Adam(color_vars, lr=0.01)

    loss_history = []
    started = time.perf_counter()
    for iteration in range(num_iterations):
        if iteration == int(num_iterations * 0.5):
            for group in points_optimizer.param_groups:
                group["lr"] = 0.4
        if iteration == int(num_iterations * 0.75):
            for group in points_optimizer.param_groups:
                group["lr"] = 0.1

        points_optimizer.zero_grad()
        width_optimizer.zero_grad()
        color_optimizer.zero_grad()
        image_hwc = render_scene(shapes, shape_groups, iteration)
        image_nchw = image_hwc.unsqueeze(0).permute(0, 3, 1, 2)
        image_features = model.encode_image(
            torch.cat([augment(image_nchw) for _ in range(num_augmentations)]))

        loss = 0
        for index in range(num_augmentations):
            current = image_features[index:index + 1]
            loss -= F.cosine_similarity(text_features, current, dim=1)
            if use_negative_prompts:
                loss += 0.3 * F.cosine_similarity(negative_features_1, current, dim=1)
                loss += 0.3 * F.cosine_similarity(negative_features_2, current, dim=1)

        loss.backward()
        points_optimizer.step()
        width_optimizer.step()
        color_optimizer.step()
        with torch.no_grad():
            for path in shapes:
                path.stroke_width.clamp_(1.0, max_stroke_width)
            for group in shape_groups:
                group.stroke_color.clamp_(0.0, 1.0)

        loss_history.append(float(loss.detach().item()))
        if preview_handle is not None and display_every and (
                iteration % display_every == 0 or iteration == num_iterations - 1):
            preview_handle.update(to_pil(image_hwc))

    elapsed_seconds = time.perf_counter() - started
    with torch.no_grad():
        final_image = render_scene(shapes, shape_groups, num_iterations)
    return {
        "final_image": final_image,
        "shapes": shapes,
        "shape_groups": shape_groups,
        "loss_history": loss_history,
        "elapsed_seconds": elapsed_seconds,
    }


def _prediction_scores(final_image, nouns_norm):
    with torch.no_grad():
        prediction_input = final_image.detach().unsqueeze(0).permute(0, 3, 1, 2)
        prediction_features = model.encode_image(augment(prediction_input))
        image_norm = prediction_features / prediction_features.norm(dim=-1, keepdim=True)
        return (100.0 * image_norm @ nouns_norm.T).softmax(dim=-1)[0]


# ============================== New in the batch runner ==============================

# Safe folder names
def _slugify(text, max_len=60):
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text.strip()).strip("_").lower()
    return (slug or "prompt")[:max_len]


# Top-5 labels and target-word rank
def _top5_and_rank(scores, labels, target_word=None):
    values, indices = scores.topk(5)
    top5 = [{"label": labels[i], "relative_score": float(v)} for v, i in zip(values.tolist(), indices.tolist())]
    target_rank = None
    if target_word is not None and target_word.lower() in labels:
        target_index = labels.index(target_word.lower())
        target_rank = int((scores > scores[target_index]).sum().item()) + 1
    return top5, target_rank


def batch_run(words, *, prompt_template="{}", output_root="/content/clipdraw_batch",
              run_mode=None, seed=None, num_paths=None, num_iterations=None,
              num_augmentations=None, max_stroke_width=None, use_negative_prompts=None,
              enable_prediction=None, prediction_vocab=None,
              skip_existing=True, show_preview=True, display_every=None, zip_results=True):

    # Settings: arguments, else Section 3
    run_mode = RUN_MODE if run_mode is None else run_mode
    seed = SEED if seed is None else seed
    max_stroke_width = MAX_STROKE_WIDTH if max_stroke_width is None else max_stroke_width
    use_negative_prompts = USE_NEGATIVE_PROMPTS if use_negative_prompts is None else use_negative_prompts
    enable_prediction = ENABLE_PREDICTION if enable_prediction is None else enable_prediction
    display_every = DISPLAY_EVERY if display_every is None else display_every
    requested = {
        "num_paths": NUM_PATHS if num_paths is None else num_paths,
        "num_iterations": NUM_ITERATIONS if num_iterations is None else num_iterations,
        "num_augmentations": NUM_AUGMENTATIONS if num_augmentations is None else num_augmentations,
        "max_stroke_width": max_stroke_width,
    }

    # Smoke-test overrides
    effective_cfg = {
        "num_paths": 16 if run_mode == "smoke_test" else requested["num_paths"],
        "num_iterations": 5 if run_mode == "smoke_test" else requested["num_iterations"],
        "num_augmentations": 1 if run_mode == "smoke_test" else requested["num_augmentations"],
    }

    # Load words, drop blanks and duplicates
    if isinstance(words, (str, Path)):
        loader = globals().get("load_words")
        words = loader(words) if loader else Path(words).read_text().splitlines()
    seen, word_list = set(), []
    for w in words:
        w = w.strip()
        if w and w.lower() not in seen:
            seen.add(w.lower())
            word_list.append(w)
    if not word_list:
        raise ValueError("No words to run.")

    # Prediction vocabulary, encoded once
    nouns_norm, labels = None, None
    if enable_prediction:
        labels = prediction_vocab if prediction_vocab is not None else globals().get("prediction_labels")
        if labels is None:
            raise RuntimeError("Prediction is on but no vocabulary found. Run Section 7 once with "
                               "ENABLE_PREDICTION=True, or pass prediction_vocab=[...].")
        labels = [label.lower() for label in labels]
        with torch.no_grad():
            nouns_features = model.encode_text(clip.tokenize(["a drawing of a " + l for l in labels]).to(device))
            nouns_norm = nouns_features / nouns_features.norm(dim=-1, keepdim=True)

    # Stop if saving to Drive while Drive isn't mounted
    if str(output_root).startswith("/content/drive") and not os.path.ismount("/content/drive"):
        raise RuntimeError("Google Drive isn't mounted, so results would be lost when the runtime ends. "
                           "Run:  from google.colab import drive; drive.mount('/content/drive')  "
                           "then call batch_run again.")

    # Output folder and summary CSV
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "batch_summary.csv"
    summary_fields = ["index", "word", "prompt", "folder", "status", "final_loss",
                      "elapsed_seconds", "top1", "top1_score", "target_rank", "error"]
    if not summary_path.exists():
        with summary_path.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=summary_fields).writeheader()

    print(f"Batch: {len(word_list)} words | mode={run_mode} | {effective_cfg} | output={output_root}")
    results = []
    batch_started = time.perf_counter()

    # Per-word loop
    for index, word in enumerate(word_list, start=1):
        prompt = prompt_template.format(word)
        folder = output_root / f"{index:03d}_{_slugify(word)}"
        metadata_path = folder / "experiment.json"
        row = {"index": index, "word": word, "prompt": prompt, "folder": str(folder)}

        # Skip finished words
        if skip_existing and metadata_path.exists():
            print(f"[{index}/{len(word_list)}] {word!r}: already done, skipping.")
            results.append({**row, "status": "skipped"})
            continue

        print(f"[{index}/{len(word_list)}] {word!r} -> {prompt!r}")
        handle = display(to_pil(torch.ones(224, 224, 3)), display_id=True) if show_preview else None
        run = None
        try:
            # Draw
            run = run_clipdraw(
                prompt, seed=seed, max_stroke_width=max_stroke_width,
                use_negative_prompts=use_negative_prompts,
                display_every=display_every if show_preview else 0, preview_handle=handle,
                **effective_cfg,
            )

            # Save PNG and SVG
            folder.mkdir(parents=True, exist_ok=True)
            png_path = folder / "clipdraw_final.png"
            svg_path = folder / "clipdraw_final.svg"
            pydiffvg.imwrite(run["final_image"].detach().cpu(), str(png_path), gamma=1.0)
            pydiffvg.save_svg(str(svg_path), 224, 224, run["shapes"], run["shape_groups"])
            if handle is not None:
                handle.update(to_pil(run["final_image"]))

            # Optional prediction
            top5, target_rank = (None, None)
            if enable_prediction:
                scores = _prediction_scores(run["final_image"], nouns_norm)
                top5, target_rank = _top5_and_rank(scores, labels, target_word=word)

            # Write experiment.json
            experiment = {
                "project": "Visibility Thresholds: The Shape a Thought Needs",
                "substrate": "vector strokes",
                "run_mode": run_mode,
                "word": word,
                "prompt": prompt,
                "prompt_template": prompt_template,
                "use_negative_prompts": use_negative_prompts,
                "negative_prompts": [NEGATIVE_PROMPT_1, NEGATIVE_PROMPT_2],
                "normalize_clip_inputs": NORMALIZE_CLIP_INPUTS,
                "seed": seed,
                "requested": requested,
                "effective": effective_cfg,
                "optimizer": {
                    "points_adam_lr": 1.0, "width_adam_lr": 0.1, "color_adam_lr": 0.01,
                    "negative_prompt_weight": 0.3,
                    "point_lr_schedule": {"50_percent": 0.4, "75_percent": 0.1},
                },
                "environment": environment,
                "elapsed_seconds": run["elapsed_seconds"],
                "final_loss": run["loss_history"][-1],
                "loss_history": run["loss_history"],
                "prediction_top5": top5,
                "target_word_rank": target_rank,
            }
            metadata_path.write_text(json.dumps(experiment, indent=2))

            # Record success
            row.update(status="ok", final_loss=round(run["loss_history"][-1], 6),
                       elapsed_seconds=round(run["elapsed_seconds"], 1),
                       top1=top5[0]["label"] if top5 else "",
                       top1_score=round(top5[0]["relative_score"], 4) if top5 else "",
                       target_rank=target_rank if target_rank is not None else "")
            msg = f"    done in {run['elapsed_seconds']:.1f}s | loss {run['loss_history'][-1]:.4f}"
            if top5:
                msg += f" | top1 {top5[0]['label']}" + (f" | target rank {target_rank}" if target_rank else "")
            print(msg)

        # Record failure, keep going
        except Exception as exc:
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            print(f"    FAILED: {row['error']}")
            traceback.print_exc()

        # Free GPU memory
        finally:
            del run
            gc.collect()
            torch.cuda.empty_cache()

        # Append summary row
        with summary_path.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=summary_fields).writerow({k: row.get(k, "") for k in summary_fields})
        results.append(row)

    # Final counts
    total = time.perf_counter() - batch_started
    n_ok = sum(r["status"] == "ok" for r in results)
    n_err = sum(r["status"] == "error" for r in results)
    n_skip = sum(r["status"] == "skipped" for r in results)
    print(f"\nBatch finished in {total / 60:.1f} min: {n_ok} ok, {n_err} failed, {n_skip} skipped.")
    print(f"Summary: {summary_path}")

    # Zip everything
    if zip_results:
        archive = shutil.make_archive(str(output_root), "zip", root_dir=output_root)
        print(f"Zipped results: {archive}")
    return results
