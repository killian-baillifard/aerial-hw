import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# ──────────────────────────────────────────────
#  CONFIGURATION  ← edit these
# ──────────────────────────────────────────────

MODELS = {
    "YOLOv8n v2rgb2": "../models/yolov8n_v2rgb_r2/weights/best.pt",
    "YOLOv8n v2bwr1": "../models/yolov8n_v2bw_r1/weights/best.pt",
    "YOLOv8n v3bwr1": "../models/yolov8n_v3bw_r1/weights/best.pt",
    "YOLOv8n v4bwr2": "../models/yolov8n_v4bw_r2/weights/best.pt",
    # Add / remove entries freely – dict key becomes the column header
}

TEST_FOLDER = "../dataset/annotated_gates_v5d/test/images/"
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
    # Input folder: first CLI arg or fallback to TEST_FOLDER
    img_dir = sys.argv[1] if len(sys.argv) > 1 else TEST_FOLDER
    if not os.path.isdir(img_dir):
        print(f"Image folder '{img_dir}' not found.")
        return

    all_images = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ])
    if not all_images:
        print(f"No images found in {img_dir} – exiting.")
        return

    print(f"Found {len(all_images)} images in '{img_dir}'.")
    print(f"Models to evaluate: {list(MODELS.keys())}")

    # Load models once
    loaded_models = {}
    for name, path in MODELS.items():
        print(f"Loading: {name} ({path})")
        loaded_models[name] = YOLO(path)

    # Attempt to get screen width to scale final canvas to roughly full screen
    try:
        import tkinter as tk
        root = tk.Tk()
        screen_w = root.winfo_screenwidth()
        root.destroy()
    except Exception:
        # fallback if tkinter is not available
        screen_w = 1400
    # target_width = ~95% of screen width
    target_width = int(screen_w * 0.95)

    window_name = "Model Comparison"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    # cache final canvases so going back/forth doesn't rerun inference unnecessarily
    canvas_cache = {}

    def build_canvas(img_path, img_file):
        """Construct the horizontal comparison canvas for a single image path."""
        raw = cv2.imread(img_path)
        if raw is None:
            return None
        h, w = raw.shape[:2]
        col_width = w
        lbl_path = os.path.splitext(img_path)[0] + ".txt"
        gt_points = load_ground_truth(lbl_path, w, h)

        column_arrays = []
        for model_name, model in loaded_models.items():
            pred_points = run_inference(model, img_path)
            annotated = draw_keypoints(raw, gt_points, pred_points)
            header = make_header_strip(model_name, col_width, HEADER_HEIGHT)
            img_label_strip = make_image_label(img_file, col_width)
            legend = make_legend_strip(col_width, LEGEND_HEIGHT)

            block = np.vstack([
                header,
                img_label_strip,
                annotated,
                legend,
            ])
            column_arrays.append(block)

        if not column_arrays:
            return None

        # pad to equal heights
        max_h = max(c.shape[0] for c in column_arrays)
        padded = []
        for col in column_arrays:
            shortage = max_h - col.shape[0]
            if shortage > 0:
                filler = np.full((shortage, col.shape[1], 3), 15, dtype=np.uint8)
                col = np.vstack([col, filler])
            padded.append(col)

        sep = np.full((max_h, COLUMN_PAD, 3), 15, dtype=np.uint8)
        rows = [padded[0]]
        for col in padded[1:]:
            rows.extend([sep, col])
        final = np.hstack(rows)

        # scale final canvas to exactly (or very close to) target_width while preserving aspect ratio
        if final.shape[1] != target_width:
            scale = target_width / final.shape[1]
            new_w = max(1, int(final.shape[1] * scale))
            new_h = max(1, int(final.shape[0] * scale))
            final = cv2.resize(final, (new_w, new_h), interpolation=cv2.INTER_AREA)

        return final

    # navigation keys (cover common OpenCV/platform codes) plus fallbacks (a/d, h/l, j/k, n/p)
    LEFT_KEYS  = {81, 2424832, 65361, 63234}
    RIGHT_KEYS = {83, 2555904, 65363, 63235}
    FALLBACK_LEFT = {ord('a'), ord('h'), ord('j'), ord('p')}
    FALLBACK_RIGHT = {ord('d'), ord('l'), ord('k'), ord('n')}

    # merge sets for easier checking
    LEFT_ALL  = set.union(LEFT_KEYS, FALLBACK_LEFT)
    RIGHT_ALL = set.union(RIGHT_KEYS, FALLBACK_RIGHT)

    def key_in_set(key, keyset):
        """Robustly check if key matches any value in keyset under common masks/offsets."""
        if key in keyset:
            return True
        # low byte (ASCII)
        low8 = key & 0xFF
        if low8 in keyset:
            return True
        # low 16 bits
        low16 = key & 0xFFFF
        if low16 in keyset:
            return True
        # shifted bytes (some platforms place codes in higher bytes)
        mid8 = (key >> 8) & 0xFF
        if mid8 in keyset:
            return True
        high16 = (key >> 16) & 0xFFFF
        if high16 in keyset:
            return True
        return False

    idx = 0
    total = len(all_images)
    while True:
        img_file = all_images[idx]
        img_path = os.path.join(img_dir, img_file)

        if idx not in canvas_cache:
            canvas = build_canvas(img_path, img_file)
            if canvas is None:
                print(f"WARNING: could not build canvas for {img_file}, skipping.")
                # move forward automatically
                idx = (idx + 1) % total
                continue
            canvas_cache[idx] = canvas
        else:
            canvas = canvas_cache[idx]

        # display with index header in terminal
        print(f"[{idx+1}/{total}] {img_file} — use ← / → to navigate, 'q' to quit")

        # ensure the window size matches canvas so imshow refreshes properly
        try:
            cv2.resizeWindow(window_name, canvas.shape[1], canvas.shape[0])
        except Exception:
            # some backends ignore resizeWindow, ignore failures
            pass
        cv2.imshow(window_name, canvas)

        key = cv2.waitKeyEx(0)
        if key == -1:
            continue

        # Quit on 'q' or ESC (match via multiple masks)
        if key_in_set(key, {ord('q'), 27}):
            print("User requested exit.")
            break

        # Navigate left / right (robust matching)
        if key_in_set(key, LEFT_ALL):
            idx = (idx - 1) % total
            continue

        if key_in_set(key, RIGHT_ALL):
            idx = (idx + 1) % total
            continue

        # any other key: ignore and continue displaying same image

    cv2.destroyAllWindows()
    print("\nViewer exited.")

if __name__ == "__main__":
    main()
