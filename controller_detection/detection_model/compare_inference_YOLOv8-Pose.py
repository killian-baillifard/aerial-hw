import os
import cv2
import numpy as np
from ultralytics import YOLO

# ──────────────────────────────────────────────
#  CONFIGURATION  ← edit these
# ──────────────────────────────────────────────

MODELS = {
    "YOLOv8n v1r1": "models/yolov8n_v1_r1/weights/best.pt",
    "YOLOv8n v2r1": "models/yolov8n_v2_r1/weights/best.pt",
    "YOLOv8n v2r2": "models/yolov8n_v2_r2/weights/best.pt",
    "YOLOv8s v2r1": "models/yolov8s_v2_r1/weights/best.pt",
    # Add / remove entries freely – dict key becomes the column header
}

TEST_FOLDER = "dataset/annotated_gates_v2_split/test/"
OUTPUT_DIR  = "inference_comparison"
BATCH_SIZE  = 4

# Inference settings
CONF_THRESHOLD = 0.5
IOU_THRESHOLD  = 0.7

# Visual settings
GT_COLOR   = (0,  255,  0)   # green  – ground-truth keypoints
PRED_COLOR = (0,   0, 255)   # red    – predicted keypoints
GT_RADIUS  = 8
PRED_RADIUS = 4

HEADER_HEIGHT  = 48          # px for model-name header above each column
LEGEND_HEIGHT  = 54          # px for legend strip below each image
COLUMN_PAD     = 6           # px between columns
ROW_PAD        = 6           # px between rows
FONT           = cv2.FONT_HERSHEY_SIMPLEX
HEADER_FONT_SCALE = 0.55
LEGEND_FONT_SCALE = 0.42
FONT_THICKNESS = 1

# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def load_ground_truth(label_path, w, h):
    """Return list of (x, y) pixel coords for visible keypoints."""
    points = []
    if not os.path.exists(label_path):
        return points
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 17:
                for i in range(4):
                    gx = int(float(parts[5 + i*3])     * w)
                    gy = int(float(parts[5 + i*3 + 1]) * h)
                    vis = float(parts[5 + i*3 + 2])
                    if vis > 0:
                        points.append((gx, gy))
    return points


def draw_keypoints(img, gt_points, pred_points):
    """Draw GT circles and prediction dots onto a copy of img."""
    out = img.copy()
    for (x, y) in gt_points:
        cv2.circle(out, (x, y), GT_RADIUS, GT_COLOR, thickness=2)
    for (x, y) in pred_points:
        if x != 0 or y != 0:
            cv2.circle(out, (x, y), PRED_RADIUS, PRED_COLOR, thickness=-1)
    return out


def run_inference(model, image_path):
    """Return list of (x, y) pixel coords for all predicted keypoints."""
    results = model.predict(source=image_path, conf=CONF_THRESHOLD,
                            iou=IOU_THRESHOLD, verbose=False)
    pred_points = []
    for result in results:
        kp = result.keypoints
        if kp is not None and kp.xy.numel() > 0:
            for i in range(len(kp.xy)):
                coords = kp.xy[i].cpu().numpy()
                for j in range(len(coords)):
                    px, py = int(coords[j][0]), int(coords[j][1])
                    pred_points.append((px, py))
    return pred_points


def make_header_strip(text, width, height):
    """Dark banner with centred white model name."""
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (30, 30, 30)
    (tw, th), _ = cv2.getTextSize(text, FONT, HEADER_FONT_SCALE, FONT_THICKNESS)
    tx = max(4, (width - tw) // 2)
    ty = (height + th) // 2
    cv2.putText(strip, text, (tx, ty), FONT, HEADER_FONT_SCALE,
                (220, 220, 220), FONT_THICKNESS, cv2.LINE_AA)
    return strip


def make_legend_strip(width, height):
    """Small strip with GT / Pred legend."""
    strip = np.zeros((height, width, 3), dtype=np.uint8)
    strip[:] = (20, 20, 20)
    # GT entry
    cv2.circle(strip, (14, height // 2), 6, GT_COLOR, 2)
    cv2.putText(strip, "GT", (26, height // 2 + 5), FONT,
                LEGEND_FONT_SCALE, GT_COLOR, 1, cv2.LINE_AA)
    # Pred entry
    cv2.circle(strip, (width // 2 + 4, height // 2), 4, PRED_COLOR, -1)
    cv2.putText(strip, "Pred", (width // 2 + 14, height // 2 + 5), FONT,
                LEGEND_FONT_SCALE, PRED_COLOR, 1, cv2.LINE_AA)
    return strip


def make_image_label(text, width, label_h=20):
    """Thin strip showing image filename."""
    strip = np.zeros((label_h, width, 3), dtype=np.uint8)
    strip[:] = (45, 45, 45)
    cv2.putText(strip, text, (4, label_h - 5), FONT, 0.35,
                (180, 180, 180), 1, cv2.LINE_AA)
    return strip


def pad_strip(img, pad_px, color=(15, 15, 15)):
    """Add a thin horizontal padding row."""
    return np.full((pad_px, img.shape[1], 3), color, dtype=np.uint8)


# ──────────────────────────────────────────────
#  MAIN
# ──────────────────────────────────────────────

def main():
    # ---------- setup output directory ----------
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ---------- gather image paths ----------
    img_dir = os.path.join(TEST_FOLDER, "images")
    lbl_dir = os.path.join(TEST_FOLDER, "labels")
    all_images = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])

    if not all_images:
        print(f"No images found in {img_dir} – exiting.")
        return

    print(f"Found {len(all_images)} images.")
    print(f"Models to evaluate: {list(MODELS.keys())}")

    # ---------- load models ----------
    loaded_models = {}
    for name, path in MODELS.items():
        print(f"Loading: {name} ({path})")
        loaded_models[name] = YOLO(path)

    # ---------- process in batches ----------
    for i in range(0, len(all_images), BATCH_SIZE):
        batch_images = all_images[i : i + BATCH_SIZE]
        batch_num = (i // BATCH_SIZE) + 1
        print(f"\n[Batch {batch_num}] Processing {len(batch_images)} images...")

        # columns_strips[model_name] = list of image-strip arrays (one per row)
        columns_strips = {name: [] for name in MODELS}
        col_width = None

        for img_file in batch_images:
            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, os.path.splitext(img_file)[0] + ".txt")

            raw = cv2.imread(img_path)
            if raw is None:
                print(f"  WARNING: could not read {img_path}, skipping.")
                continue

            h, w = raw.shape[:2]
            if col_width is None:
                col_width = w

            # Resize if this image differs in width (keep aspect ratio)
            if w != col_width:
                scale = col_width / w
                raw = cv2.resize(raw, (col_width, int(h * scale)))
                h, w = raw.shape[:2]

            gt_points = load_ground_truth(lbl_path, w, h)
            img_label_strip = make_image_label(img_file, col_width)

            for model_name, model in loaded_models.items():
                pred_points = run_inference(model, img_path)
                annotated   = draw_keypoints(raw, gt_points, pred_points)

                legend = make_legend_strip(col_width, LEGEND_HEIGHT)
                # Stack: thin top separator + image label + annotated image + legend + row pad
                block = np.vstack([
                    img_label_strip,
                    annotated,
                    legend,
                    pad_strip(annotated, ROW_PAD),
                ])
                columns_strips[model_name].append(block)

        if col_width is None:
            print(f"  No valid images in batch {batch_num} – skipping render.")
            continue

        # ---------- assemble each column for this batch ----------
        model_names = list(MODELS.keys())
        column_arrays = []
        for name in model_names:
            strips = columns_strips[name]
            if not strips:
                continue
            header = make_header_strip(name, col_width, HEADER_HEIGHT)
            column = np.vstack([header] + strips)
            column_arrays.append(column)

        if not column_arrays:
            continue

        # Ensure all columns have the same height (pad bottom if needed)
        max_h = max(c.shape[0] for c in column_arrays)
        padded = []
        for col in column_arrays:
            shortage = max_h - col.shape[0]
            if shortage > 0:
                filler = np.full((shortage, col.shape[1], 3), 15, dtype=np.uint8)
                col = np.vstack([col, filler])
            padded.append(col)

        # Horizontal separator between columns
        sep = np.full((max_h, COLUMN_PAD, 3), 15, dtype=np.uint8)
        rows = [padded[0]]
        for col in padded[1:]:
            rows.extend([sep, col])

        final = np.hstack(rows)

        # Save the batch output
        output_filename = os.path.join(OUTPUT_DIR, f"comparison_batch_{batch_num:03d}.png")
        cv2.imwrite(output_filename, final)
        print(f"  Saved → {output_filename} ({final.shape[1]}×{final.shape[0]} px)")

    print("\nAll batches processed successfully!")


if __name__ == "__main__":
    main()
