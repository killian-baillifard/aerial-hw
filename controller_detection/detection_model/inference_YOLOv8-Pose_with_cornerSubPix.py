import os
from ultralytics import YOLO
import cv2
import numpy as np

# Load your custom trained model
model = YOLO('models/yolov8n_v2rgb_r2/weights/best.pt')

# Get test image folder path
test_folder = 'dataset/annotated_gates_v2bw_split/test/'

# Build image list (sorted) and derive label path from image basename to guarantee alignment
test_images = sorted([f for f in os.listdir(os.path.join(test_folder, 'images')) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

# Choose image index
image_idx = 12 # 3 # 12 # 9
test_image_path = os.path.join(test_folder, 'images', test_images[image_idx])
label_path = os.path.join(test_folder, 'labels', os.path.splitext(test_images[image_idx])[0] + '.txt')

print(f"Testing on image: {test_image_path}")
print(f"Using label: {label_path}")

# 1. Load the image with OpenCV to draw on it
img = cv2.imread(test_image_path)
h, w, _ = img.shape  # Get image height and width to scale annotations

# --- Convert to grayscale once — needed for cornerSubPix ---
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# cornerSubPix parameters
#   winSize:    half-size of the search window. (5,5) = 11×11 px window.
#               Larger = more robust to noise but slower and less precise near
#               closely-spaced corners. Try (3,3)–(7,7) for LED gates.
#   zeroZone:   half-size of the dead zone in the middle (-1,-1 = disabled).
#               Set to e.g. (2,2) if you get instability near the exact corner.
#   criteria:   stop when either max iterations reached OR accuracy < epsilon.
WIN_SIZE  = (5, 5)
ZERO_ZONE = (-1, -1)
CRITERIA  = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

def refine_corners(gray_img, raw_points):
    """
    Refine a list of (x, y) points to sub-pixel accuracy.

    Parameters
    ----------
    gray_img  : uint8 grayscale image
    raw_points: list/array of (x, y) float or int coordinates

    Returns
    -------
    numpy array of shape (N, 2) with refined float coordinates,
    or the original points if refinement fails.
    """
    if len(raw_points) == 0:
        return np.array(raw_points, dtype=np.float32)

    # cornerSubPix expects shape (N, 1, 2) float32
    corners = np.array(raw_points, dtype=np.float32).reshape(-1, 1, 2)

    refined = cv2.cornerSubPix(gray_img, corners, WIN_SIZE, ZERO_ZONE, CRITERIA)

    return refined.reshape(-1, 2)   # back to (N, 2)

# ── Ground truth ──────────────────────────────────────────────────────────────
# 2. Parse the ground truth label file and draw CIRCLES
if os.path.exists(label_path):
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            # Check if it's a pose label (Class + 4 Box coords + 12 Keypoint coords)
            if len(parts) >= 17: 
                # Keypoints start at index 5 and come in chunks of 3 (x, y, visibility)
                for i in range(4):
                    # YOLO saves normalized coordinates (0.0 to 1.0). 
                    # Multiply by width/height to get actual pixel coordinates.
                    gt_x = int(float(parts[5 + i*3]) * w)
                    gt_y = int(float(parts[5 + i*3 + 1]) * h)
                    visibility = float(parts[5 + i*3 + 2])
                    
                    # Only draw if the point is visible or occluded (visibility > 0)
                    if visibility > 0:
                        # Draw a hollow GREEN circle for Ground Truth
                        cv2.circle(img, (gt_x, gt_y), radius=8, color=(0, 255, 0), thickness=2)

# ── YOLO predictions ──────────────────────────────────────────────────────────
# Run inference on a new image
# Note: iou=0.7 allows bounding boxes to heavily overlap without being filtered out
results = model.predict(source=test_image_path, conf=0.5, iou=0.7)

corner_names = ["Bottom-Left", "Top-Left", "Top-Right", "Bottom-Right"]

for result in results:
    # Get the keypoints object
    keypoints = result.keypoints
    
    # Extract coordinates and confidences
    if keypoints is not None and keypoints.xy.numel() > 0:
        for i in range(len(keypoints.xy)):
            coords = keypoints.xy[i].cpu().numpy()  # Array of 4 (x,y) corners
            confs = keypoints.conf[i].cpu().numpy() # Array of 4 confidences
            
            # Collect valid raw predictions
            raw_pred = []
            valid_idx = []
            for j in range(4):
                px, py = float(coords[j][0]), float(coords[j][1])
                if px != 0 or py != 0:          # skip suppressed keypoints
                    raw_pred.append((px, py))
                    valid_idx.append(j)

            # ── Sub-pixel refinement ──────────────────────────────────────────────
            refined_pred = refine_corners(gray, raw_pred)

            print(f"\n--- Gate {i+1} ---")
            for k, j in enumerate(valid_idx):
                pred_x, pred_y = coords[j]
                rx, ry = refined_pred[k]
                print(f"{corner_names[j]}: raw=({pred_x:.2f}, {pred_y:.2f})  "
                    f"refined=({rx:.2f}, {ry:.2f})  conf={confs[j]:.2f}")

                cv2.circle(img, (int(round(pred_x)), int(round(pred_y))),
                        radius=2, color=(255, 0, 0), thickness=-1)
                cv2.circle(img, (int(round(rx)), int(round(ry))),
                        radius=1, color=(0, 0, 255), thickness=-1)

# Add a simple legend to the top-left of the image
cv2.putText(img, "Ground Truth: Green Circles", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(img, "Prediction: Red Points", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the final image
cv2.imshow('YOLOv8 Pose: Annotations vs Predictions', img)

# Wait for any key press, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()