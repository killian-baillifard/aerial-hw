"""run within detection_model folder"""
import os
from ultralytics import YOLO

# Get the absolute path of the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

def train_gate_pose_model():
    # Load a pre-trained YOLOv8 Nano pose model
    # The 'n' stands for nano (fastest, lightest). 
    # You can change to 'yolov8s-pose.pt' (small) if you need more accuracy
    model = YOLO('yolov8n-pose.pt')

    # Train the model
    # We pass the path to the data.yaml file we created
    results = model.train(
        data='dataset/annotated_gates_v1_split/data.yaml',
        
        # Training parameters
        epochs=100,               # 100 is a good starting point. It will early-stop if it plateaus.
        imgsz=320,                # Resize images to 320x320 (closest multiple of 32 to your 324 width)
        batch=16,                 # Adjust based on your GPU memory
        device='mps',                 # Use '0' for GPU, or 'cpu' if you don't have a dedicated GPU, 'mps' for Apple Silicon
        workers=4,                # Number of dataloader workers
        
        # --- Crucial Augmentations for your specific environment ---
        
        # Lighting augmentations (dim room handling)
        hsv_h=0.015,              # Image HSV-Hue augmentation
        hsv_s=0.7,                # Image HSV-Saturation augmentation
        hsv_v=0.4,                # Image HSV-Value augmentation (helps with dim lighting changes)
        
        # Spatial augmentations (overlapping and perspective)
        degrees=10.0,             # Image rotation (+/- deg)
        translate=0.1,            # Image translation (+/- fraction)
        scale=0.5,                # Image scale (+/- gain)
        perspective=0.0005,       # Perspective changes (helps if gates are viewed at angles)
        
        # Image composition augmentations (helps with overlap)
        mosaic=1.0,               # Combine 4 images into 1 (highly recommended for overlapping objects)
        mixup=0.1,                # Image mixup (layering images)
        
        # Save settings
        project=os.path.join(current_dir, 'models'),         # Folder name where results are saved
        name='yolov8n_v1_r1',     # Subfolder for this specific training run
        save=True                 # Save the weights
    )

if __name__ == '__main__':
    train_gate_pose_model()