You need to **calibrate your camera** using a known pattern. OpenCV's standard approach uses a checkerboard:

## 1. Collect calibration images

Take 20–30 photos of a checkerboard from different angles/distances. The same camera + lens + resolution you use for gate detection.

## 2. Run calibration

```python
import cv2
import numpy as np
import glob

CHECKERBOARD = (9, 6)  # inner corners — count carefully, not squares
SQUARE_SIZE  = 0.025   # meters (measure your printed board)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []  # 3D points in world space
img_points = []  # 2D points in image space

images = glob.glob('calibration_images/*.png')

for fname in images:
    img  = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        # Refine to sub-pixel — valid here because checkerboard has
        # sharp geometric corners (unlike your round LEDs)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        obj_points.append(objp)
        img_points.append(corners_refined)

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None
)

print("K =\n", K)
print("dist =\n", dist)
print("Reprojection error:", ret)  # aim for < 0.5 px
```

## 3. Save and reuse

```python
np.save('K.npy', K)
np.save('dist.npy', dist)

# Later, in your triangulation pipeline:
K    = np.load('K.npy')
dist = np.load('dist.npy')
```

## What K looks like

```
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

- `fx, fy` — focal length in pixels (usually close to each other)
- `cx, cy` — principal point, typically near image center `(w/2, h/2)`

## Practical tips

- Print the checkerboard on **rigid material** — a warped paper board ruins calibration
- Cover the **full field of view** in your image set, including corners
- Vary tilt and rotation, not just distance
- Reprojection error **< 0.5 px** is good; < 0.3 px is excellent
- If you already know your camera model (e.g. it's a specific drone/webcam), the manufacturer sometimes publishes K — but measuring it yourself is always more accurate