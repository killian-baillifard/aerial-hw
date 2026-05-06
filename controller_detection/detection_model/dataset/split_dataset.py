import os
import random
import shutil
import yaml

# 1. Setup paths based on your folder info
current_file_path = os.path.dirname(os.path.abspath(__file__))
dataset_dir = 'annotated_gates_v1'
dataset_dir_path = os.path.join(current_file_path, dataset_dir)

# Define where the raw data is (CVAT export usually puts them in images/ labels/)
source_images_path = os.path.join(dataset_dir_path, 'images/train')
source_labels_path = os.path.join(dataset_dir_path, 'labels/train')
source_yaml_path = os.path.join(dataset_dir_path, 'data.yaml')

# Define where the final split dataset will go
output_root = os.path.join(current_file_path, f"{dataset_dir}_split")

# 2. Create the new folder structure
for folder in ['train', 'val']:
    os.makedirs(os.path.join(output_root, folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, folder, 'labels'), exist_ok=True)

# 3. Get list of all images
if not os.path.exists(source_images_path):
    print(f"Error: Could not find images folder at {source_images_path}")
else:
    images = [f for f in os.listdir(source_images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    print(f"Found {len(images)} image files in {source_images_path}")
    if len(images) == 0:
        print("No images to process. Check file extensions or image folder contents.")
    random.seed(42) # Keeps the split consistent every time you run it
    random.shuffle(images)

    # 4. Split point (80% train, 20% val)
    split_idx = int(len(images) * 0.8)
    train_images = images[:split_idx]
    val_images = images[split_idx:]

    def process_split(files, subset):
        count = 0
        for f in files:
            # Source paths
            img_src = os.path.join(source_images_path, f)
            # YOLO labels have the same name as images but .txt extension
            label_name = os.path.splitext(f)[0] + '.txt'
            label_src = os.path.join(source_labels_path, label_name)

            # Destination paths
            img_dest = os.path.join(output_root, subset, 'images', f)
            label_dest = os.path.join(output_root, subset, 'labels', label_name)

            # Copy Image
            shutil.copy(img_src, img_dest)
            
            # Copy Label (if it exists)
            if os.path.exists(label_src):
                shutil.copy(label_src, label_dest)
                count += 1
        return count

    # 5. Run the process
    train_count = process_split(train_images, 'train')
    val_count = process_split(val_images, 'val')

    # 6. Handle the data.yaml
    output_yaml_path = os.path.join(output_root, 'data.yaml')
    
    if os.path.exists(source_yaml_path):
        # Load the existing yaml to preserve 'names' and 'kpt_shape'
        with open(source_yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
        
        # Update the paths to the new absolute locations
        # Use a relative path starting with ./dataset/..._split
        data_config['path'] = os.path.join('.', os.path.basename(current_file_path), f"{dataset_dir}_split")
        data_config['train'] = 'train/images'
        data_config['val'] = 'val/images'
        # Optional: if you have a test set, set it too. Otherwise remove.
        if 'test' in data_config:
            data_config.pop('test')

        # Write the updated yaml to the new location
        with open(output_yaml_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False)
        print(f"data.yaml updated and copied to: {output_yaml_path}")
    else:
        print("Warning: source data.yaml not found. You will need to create one manually.")

    print(f"--- Split Complete ---")
    print(f"Total Images: {len(images)}")
    print(f"Train: {len(train_images)} images, {train_count} labels")
    print(f"Val:   {len(val_images)} images, {val_count} labels")
    print(f"New dataset location: {output_root}")