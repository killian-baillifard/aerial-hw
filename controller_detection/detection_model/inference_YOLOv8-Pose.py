import os
from ultralytics import YOLO
import cv2

# Load your custom trained model
model = YOLO('models/yolov8n_v2_r2/weights/best.pt')

# Get test image folder path
test_folder = 'dataset/annotated_gates_v2_split/test/'

# Build image list (sorted) and derive label path from image basename to guarantee alignment
test_images = sorted([f for f in os.listdir(os.path.join(test_folder, 'images')) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])

# Choose image index
image_idx = 19 # 3 # 12 # 9
test_image_path = os.path.join(test_folder, 'images', test_images[image_idx])
label_path = os.path.join(test_folder, 'labels', os.path.splitext(test_images[image_idx])[0] + '.txt')

print(f"Testing on image: {test_image_path}")
print(f"Using label: {label_path}")

# 1. Load the image with OpenCV to draw on it
img = cv2.imread(test_image_path)
h, w, _ = img.shape  # Get image height and width to scale annotations

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

# Run inference on a new image
# Note: iou=0.7 allows bounding boxes to heavily overlap without being filtered out
results = model.predict(source=test_image_path, conf=0.5, iou=0.7)

for result in results:
    # Get the keypoints object
    keypoints = result.keypoints
    
    # Extract coordinates and confidences
    if keypoints is not None and keypoints.xy.numel() > 0:
        for i in range(len(keypoints.xy)):
            coords = keypoints.xy[i].cpu().numpy()  # Array of 4 (x,y) corners
            confs = keypoints.conf[i].cpu().numpy() # Array of 4 confidences
            
            print(f"\n--- Gate {i+1} ---")
            corner_names = ["Bottom-Left", "Top-Left", "Top-Right", "Bottom-Right"]
            
            for j in range(4):
                pred_x, pred_y = int(coords[j][0]), int(coords[j][1])
                print(f"{corner_names[j]}: ({pred_x}, {pred_y}), Conf: {confs[j]:.2f}")
                
                # 4. Draw the predicted POINTS
                # Only draw if the model actually predicted a location (x and y != 0)
                if pred_x != 0 and pred_y != 0:
                    # Draw a solid RED dot for Predictions
                    cv2.circle(img, (pred_x, pred_y), radius=3, color=(0, 0, 255), thickness=-1)


# Add a simple legend to the top-left of the image
cv2.putText(img, "Ground Truth: Green Circles", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
cv2.putText(img, "Prediction: Red Points", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Display the final image
cv2.imshow('YOLOv8 Pose: Annotations vs Predictions', img)

# Wait for any key press, then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()