import cv2
import numpy as np
import glob
import os
import sys

CHECKERBOARD = (9, 6)  # inner corners — count carefully, not squares
SQUARE_SIZE  = 0.019   # meters (measure your printed board)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

obj_points = []  # 3D points in world space
img_points = []  # 2D points in image space

# collect images
try:
    images = glob.glob('calibration_images/*.png')
except Exception as e:
    print(f"Error accessing images: {e}")
    print("Run this script from direct parent folder of 'calibration_images' and ensure the folder contains .png images of the checkerboard pattern.")
    sys.exit(1)

shape = None

for fname in images:
    img = cv2.imread(fname)
    if img is None:
        print(f"Warning: could not read image {fname}, skipping.")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if shape is None:
        # set shape from first valid image: (width, height)
        shape = gray.shape[::-1]

    # Use more robust flags for detection; skip fast_check if you want better recall
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)

    if ret:
        # Refine to sub-pixel — valid here because checkerboard has
        # sharp geometric corners (unlike your round LEDs)
        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1),
            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        obj_points.append(objp)
        img_points.append(corners_refined)
    else:
        print(f"Warning: chessboard not found in {fname}")

# Ensure we have at least one successful detection
if not obj_points or not img_points:
    raise RuntimeError(
        "No checkerboard corners were detected in any image. "
        "Check that images in the provided folder exist, are readable, "
        "and contain the expected checkerboard pattern."
    )

ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    obj_points, img_points, shape, None, None
)

print("K =\n", K)
print("dist =\n", dist)
print("Reprojection error:", ret)  # aim for < 0.5 px

np.save('K.npy', K)
np.save('dist.npy', dist)
