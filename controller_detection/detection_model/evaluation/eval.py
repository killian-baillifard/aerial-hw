"""
Multi-model YOLO Pose Evaluation Script
Run from within the detection_model folder.

Evaluates each model listed in MODELS against a single test dataset
and prints a side-by-side comparison table.
"""

import os
import json
import shutil
import tempfile
import time
from pathlib import Path

import yaml
import pandas as pd
from ultralytics import YOLO
from ultralytics.utils import SETTINGS

# ---------------------------------------------------------------------------
# Configuration – edit freely
# ---------------------------------------------------------------------------

MODELS: dict[str, str] = {
    "YOLOv8n v2rgb2": "../models/yolov8n_v2rgb_r2/weights/best.pt",
    "YOLOv8n v2bwr1": "../models/yolov8n_v2bw_r1/weights/best.pt",
    "YOLOv8n v3bwr1": "../models/yolov8n_v3bw_r1/weights/best.pt",
    "YOLOv8n v4bwr2": "../models/yolov8n_v4bw_r2/weights/best.pt",
    # Add / remove entries freely – dict key becomes the column header
}

TEST_DATASET = "../dataset/annotated_gates_v3bw_split/val/"

# Inference / val settings
IMGSZ    = 320
DEVICE   = "mps"   # 'mps' | 'cpu' | '0' (CUDA GPU index)
CONF     = 0.5    # confidence threshold
IOU      = 0.7    # NMS IoU threshold

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Metrics we want to surface in the comparison table.
# Keys must match what Ultralytics returns in results.results_dict.
METRIC_KEYS = [
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    # Pose-specific (present when the model has keypoints)
    "metrics/precision(P)",
    "metrics/recall(P)",
    "metrics/mAP50(P)",
    "metrics/mAP50-95(P)",
]

METRIC_LABELS = {
    "metrics/precision(B)":    "Box Precision",
    "metrics/recall(B)":       "Box Recall",
    "metrics/mAP50(B)":        "Box mAP@50",
    "metrics/mAP50-95(B)":     "Box mAP@50-95",
    "metrics/precision(P)":    "Pose Precision",
    "metrics/recall(P)":       "Pose Recall",
    "metrics/mAP50(P)":        "Pose mAP@50",
    "metrics/mAP50-95(P)":     "Pose mAP@50-95",
}


def find_data_yaml(test_dir: str) -> str:
    """Walk up from the test folder to find data.yaml."""
    test_path = Path(test_dir).resolve()
    for parent in [test_path, test_path.parent, test_path.parent.parent]:
        candidate = parent / "data.yaml"
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"Could not find data.yaml near '{test_dir}'. "
        "Pass the path explicitly or place data.yaml in the dataset root."
    )


def make_absolute_yaml(original_yaml: str, test_images_dir: str) -> str:
    """
    Ultralytics resolves relative paths in data.yaml against its global
    `datasets_dir` (often /opt/homebrew/datasets), not the yaml's own folder.

    Fix: write a patched copy to a temp directory where every path is absolute,
    and point `test` explicitly at the images folder we want to evaluate.
    Returns the path to the patched yaml file.
    """
    with open(original_yaml) as f:
        cfg = yaml.safe_load(f)

    yaml_dir = Path(original_yaml).parent.resolve()
    test_images = Path(test_images_dir).resolve()

    # Resolve train / val paths so the validator doesn't choke on those either
    for key in ("train", "val"):
        if key in cfg and cfg[key]:
            p = Path(cfg[key])
            if not p.is_absolute():
                cfg[key] = str(yaml_dir / p)

    # Point 'test' at our target folder (images sub-folder if it exists)
    images_subdir = test_images / "images"
    cfg["test"] = str(images_subdir if images_subdir.exists() else test_images)

    # Write to a temp file so we don't touch the original
    tmp_dir = Path(tempfile.mkdtemp())
    patched = tmp_dir / "data_eval.yaml"
    with open(patched, "w") as f:
        yaml.dump(cfg, f)

    return str(patched)


def warmup_device(patched_yaml: str, models_dict: dict) -> None:
    """
    Run one throwaway val() pass to force MPS/CUDA JIT compilation at the
    process level. This is the only reliable way — predict() and raw tensor
    passes don't exercise the same internal pipeline that val() uses, so the
    first timed val() always pays the JIT tax otherwise.

    We use the first available model for the warmup run, then discard it.
    """
    first_weights = next(
        (Path(__file__).parent / p for p in models_dict.values()),
        None
    )
    if first_weights is None or not Path(first_weights).resolve().exists():
        return

    print("  [Warmup] Running throwaway val() to prime MPS JIT...")
    _model = YOLO(str(Path(first_weights).resolve()))
    _model.val(
        data=patched_yaml,
        imgsz=IMGSZ,
        device=DEVICE,
        conf=CONF,
        iou=IOU,
        split="test",
        verbose=False,
    )
    del _model
    print("  [Warmup] Done.\n")


def evaluate_model(name: str, weights: str, patched_yaml: str) -> dict:
    """Load a YOLO model and run validation. Returns a flat metrics dict."""
    weights_path = Path(weights).resolve()
    if not weights_path.exists():
        print(f"  [SKIP] Weights not found: {weights_path}")
        return {}

    print(f"\n{'─' * 60}")
    print(f"  Evaluating : {name}")
    print(f"  Weights    : {weights_path}")
    print(f"  Data YAML  : {patched_yaml}")
    print(f"{'─' * 60}")

    original_datasets_dir = SETTINGS.get("datasets_dir")
    SETTINGS.update({"datasets_dir": str(Path(patched_yaml).parent)})

    try:
        model = YOLO(str(weights_path))

        t0 = time.perf_counter()
        results = model.val(
            data=patched_yaml,
            imgsz=IMGSZ,
            device=DEVICE,
            conf=CONF,
            iou=IOU,
            split="test",
            verbose=False,
        )
        elapsed = time.perf_counter() - t0
    finally:
        if original_datasets_dir is not None:
            SETTINGS.update({"datasets_dir": original_datasets_dir})

    raw: dict = results.results_dict

    row = {"Model": name, "Eval time (s)": round(elapsed, 1)}
    for key in METRIC_KEYS:
        label = METRIC_LABELS.get(key, key)
        row[label] = round(raw[key], 4) if key in raw else "—"

    if hasattr(results, "fitness"):
        row["Fitness"] = round(float(results.fitness), 4)

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    current_dir = Path(__file__).parent.resolve()
    test_dir = (current_dir / TEST_DATASET).resolve()

    print(f"\nWorking directory : {current_dir}")
    print(f"Test dataset      : {test_dir}")

    original_yaml = find_data_yaml(str(test_dir))
    print(f"Data YAML found   : {original_yaml}")

    # Patch the yaml so all paths are absolute – fixes Ultralytics datasets_dir bug
    patched_yaml = make_absolute_yaml(original_yaml, str(test_dir))
    print(f"Patched YAML      : {patched_yaml}\n")

    # Prime MPS JIT with a throwaway val() before any timed runs
    warmup_device(patched_yaml, MODELS)

    rows = []
    for model_name, weights_rel in MODELS.items():
        weights_abs = str((current_dir / weights_rel).resolve())
        row = evaluate_model(model_name, weights_abs, patched_yaml)
        if row:
            rows.append(row)

    if not rows:
        print("\nNo models were evaluated successfully. Check your weight paths.")
        return

    # Build comparison DataFrame
    df = pd.DataFrame(rows).set_index("Model")

    # Drop columns that are entirely "—" (e.g. pose metrics for a det-only model)
    df = df.loc[:, (df != "—").any(axis=0)]

    # Pretty-print
    print("\n" + "=" * 70)
    print("  MODEL COMPARISON RESULTS")
    print("=" * 70)
    print(df.to_string())
    print("=" * 70)

    # Highlight best per numeric column
    LOWER_IS_BETTER = {"Eval time (s)"}
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols):
        print("\n  Best per metric:")
        for col in numeric_cols:
            lower = col in LOWER_IS_BETTER
            best_val   = df[col].min() if lower else df[col].max()
            best_models = df.index[df[col] == best_val].tolist()
            winners = " & ".join(best_models)
            print(f"    {col:<25} → {winners}  ({best_val})")

    # Save results to CSV alongside the script
    out_csv = current_dir / "evaluation_results.csv"
    df.to_csv(out_csv)
    print(f"\n  Results saved to: {out_csv}\n")

    # Also save as JSON for programmatic use
    out_json = current_dir / "evaluation_results.json"
    df.reset_index().to_json(out_json, orient="records", indent=2)
    print(f"  JSON saved to   : {out_json}\n")


if __name__ == "__main__":
    main()
