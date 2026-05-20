# Gate Detection via YOLOv8-Pose

## Image Annotation

### CVAT (Computer Vision Annotation Tool)

https://docs.cvat.ai/docs/administration/community/basics/installation/

```bash
cd cvat
docker-compose up -d

docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
```

Username: admin
Email address: admin@admin.admin
Password: admin

On Google Chrome: ```localhost:8080```

Create project "Gate Detection", add tasks for each image upload, and annotate the gates with bounding boxes and keypoints.

Export as YOLOv8-Pose format, e.g. ```Ultralytics YOLO Pose 1.0```.

### Labeling Instructions

- **Class**: gate
- **Keypoints**: 4 corners of the gate (bottom-left, top-left, top-right, bottom-right)
- **Bounding Box**: Enclose the entire gate

#### Edge Cases

* **When to Annotate:** If all corners are blurry, washed out, or distant, skip the gate. If at least one corner is crisp, you must annotate the full skeleton.
* **Visible vs. Occluded:** Mark crisp corners as **Visible**. Estimate the exact position of blocked or highly blurred corners (that are still within the camera view) and mark them as **Occluded**.
* **Out of Frame:** If a corner is cut off by the image edge, simply mark it as **Outside** (shortcut 'O'). It will disappear from the canvas - you do not need to estimate its position.
* **Strict Precision:** Never mark any point as **Visible** unless you can confidently pinpoint the exact pixel center of that specific LED corner.
* **Multiple Gates:** Annotate overlapping gates separately with unique IDs, double-checking that points do not accidentally snap to the wrong skeleton.

### Automatic Annotation - (do not use colima)

Architecture compatibility issues might arise when running CVAT on Apple Silicon (M4) due to the use of x86_64 images.

```bash
cd cvat

# stop CVAT
docker-compose down

# restart it with the serverless profile enabled
docker-compose -f docker-compose.yml -f components/serverless/docker-compose.serverless.yml up -d
```

Ship model to CVAT
```bash
# install nuclio client
curl -L https://github.com/nuclio/nuclio/releases/download/1.15.26/nuctl-1.15.26-darwin-arm64 -o nuctl

chmod +x nuctl
sudo mv nuctl /usr/local/bin/

nuctl version
```

Prepare CVAT
```bash
cd cvat
mkdir -p serverless/pytorch/ultralytics/yolov8
```
Inside there need to be three files:
- ```function.yaml```: Configuration (labels, hardware, etc.). Use Python 3.11 runtime for Apple Silicon (M4) - just match the correct version (look at at model training section) :/ - add ```--no-cache-dir``` flag to pip install in the Dockerfile to avoid caching issues.
- ```main.py```: The Python wrapper that runs the inference.
- ```best.pt```: Your custom weights from your training.

Deploy model to Nuclio (tell Nuclio to build a container for your model):
```bash
cd cvat
cd serverless/pytorch/ultralytics/yolov8/

# create the project in nuclio
nuctl create project cvat --platform local
```

```bash
conda activate gate_detection_env

# Download the correct wheel
pip download msgpack==1.1.0 \
  --platform manylinux2014_aarch64 \
  --python-version 311 \
  --only-binary=:all: \
  --no-deps \
  -d /tmp/msgpack-arm64

# Create a build context
mkdir -p /tmp/nuclio-patch
cp /tmp/msgpack-arm64/msgpack*.whl /tmp/nuclio-patch/

cat > /tmp/nuclio-patch/Dockerfile <<'EOF'
FROM quay.io/nuclio/handler-builder-python-onbuild:1.15.26-arm64
COPY msgpack-1.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl \
     /home/nuclio/bin/py3.11-whl/
EOF

# Build the patched image
docker buildx build --platform linux/arm64 \
  -t quay.io/nuclio/handler-builder-python-onbuild:1.15.26-arm64 \
  --load \
  /tmp/nuclio-patch

# Clean up
rm -rf /tmp/nuclio-patch
docker builder prune -f

# Copy the wheel file to the current directory for Nuclio to use
mkdir -p whl
cp /tmp/msgpack-arm64/msgpack-1.1.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl whl/
```

```bash
DOCKER_DEFAULT_PLATFORM=linux/arm64 nuctl deploy pytorch-ultralytics-yolov8-pose-gate \
  --project-name cvat \
  --path . \
  --platform local

# delete the function
nuctl delete function pytorch-ultralytics-yolov8-pose-gate --namespace cvat --platform local
nuctl delete project cvat --platform local
```

#### Multiple Models

Create folder again for the new model, copy the ```function.yaml``` and ```main.py``` from the previous model, update the ```best.pt``` with the new weights, and deploy again. 

In ```function.yaml```, make sure to update the function name.

In the new model folder run: (but change the name ```pytorch-ultralytics-yolov8-pose-gate-new_version```)
```bash
conda activate gate_detection_env

DOCKER_DEFAULT_PLATFORM=linux/arm64 nuctl deploy pytorch-ultralytics-yolov8-pose-gate-v3bw \
  --project-name cvat \
  --path . \
  --platform local
```

## Dataset Structure

Split the dataset into training and validation sets. Use ```split_dataset.py``` .

## Model Training

### Conda Environment

```bash
conda create -n gate_detection_env python=3.13
conda activate gate_detection_env
pip install -r requirements.txt
```

Go to ```detection_model```folder and run ```python training_YOLOv8-Pose.py```.

### Improve Model Performance

- **Data Augmentation**: Experiment with different augmentation techniques (e.g., rotation, scaling, color jitter) to increase dataset diversity.
- **Hyperparameter Tuning**: Adjust learning rate, batch size, and number of epochs to find the optimal training configuration.
- **Model Architecture**: Try different YOLOv8 variants (e.g., YOLOv8m, YOLOv8l) for potentially better performance at the cost of increased computational requirements.
- **Transfer Learning**: Start with pre-trained weights on a similar dataset to speed up convergence and improve accuracy, especially if the dataset is small.

### Further Possible Actions

- **copy_paste Augmentation** 1a
- **More Training Epochs** 1a
- **Larger Model Variants** 1b
- **Dataset Enhancement**: 2
- **Undistorted Pipeline** 1c
- **Close-ups of Corners** 2
- **Train upon v3bw_r1 model** 1d

## Inference

Run ```python inference_YOLOv8-Pose.py``` to test the trained model on new images or video streams.
