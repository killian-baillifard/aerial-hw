# Gate Detection via YOLOv8-Pose

## Image Annotation

### CVAT (Computer Vision Annotation Tool)

https://docs.cvat.ai/docs/administration/community/basics/installation/

```
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

## Dataset Structure

Split the dataset into training and validation sets. Use ```split_dataset.py``` .

## Model Training

### Conda Environment

```
conda create -n gate_detection_env python=3.13
conda activate gate_detection_env
pip install -r requirements.txt
```

Go to ```detection_model```folder and run ```python training_YOLOv8-Pose.py```.


## Inference

Run ```python inference_YOLOv8-Pose.py``` to test the trained model on new images or video streams.
