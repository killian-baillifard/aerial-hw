from pathlib import Path
import cv2
import sys

#!/usr/bin/env python3
# GitHub Copilot


# Name of the video file (expected to be next to this script)
VIDEO_NAME = "2026-05-23-23-33-07.avi"

script_dir = Path(__file__).resolve().parent
video_path = script_dir / VIDEO_NAME
if not video_path.exists():
    print(f"Error: video not found: {video_path}")
    sys.exit(1)

out_dir = script_dir / video_path.stem
out_dir.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video_path))
if not cap.isOpened():
    print(f"Error: cannot open video: {video_path}")
    sys.exit(1)

count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    count += 1
    out_file = out_dir / f"{count:08d}.png"
    cv2.imwrite(str(out_file), frame)

cap.release()
print(f"Saved {count} frames to {out_dir}")