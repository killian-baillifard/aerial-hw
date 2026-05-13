# convert all the images in test/images train/images and val/images folder from rgb to bw and save them in the same location (overwrite the original images)
import os
from PIL import Image
from tqdm import tqdm # Great for progress bars

# pip install tqdm

def convert_rgb_to_bw(image_path):
    with Image.open(image_path) as img:
        # Check if it's already grayscale to save time
        if img.mode != "L":
            bw = img.convert("L")
            bw.save(image_path)

def process_directory(directory):
    print(f"Processing: {directory}")
    files = [f for f in os.listdir(directory) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    
    for filename in tqdm(files):
        img_path = os.path.join(directory, filename)
        convert_rgb_to_bw(img_path)

if __name__ == "__main__":
    process_directory("train/images")
    process_directory("val/images")
    process_directory("test/images")

