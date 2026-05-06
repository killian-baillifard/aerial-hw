import os
import cv2
from matplotlib import image
import numpy as np
import detection_controller as dc

import itertools
import math

def main():

    controller = dc.DetectionController()

    # import test images and sensor data from ../saved_captures
    measures = None
    test_images = []
    current_dir_path = os.path.dirname(os.path.realpath(__file__))
    saved_captures_dir = os.path.join(os.path.dirname(current_dir_path), 'saved_captures')
    image_data_dir = os.path.join(saved_captures_dir, '2026-05-05-18-57-43')
    list_of_files = sorted(os.listdir(image_data_dir))
    for filename in list_of_files:
        if filename.endswith('.png'):
            img = cv2.imread(os.path.join(image_data_dir, filename))
            test_images.append(img)
    measures_file = os.path.join(image_data_dir, 'measures.npy')
    if os.path.exists(measures_file):
        measures = np.load(measures_file, allow_pickle=True)
    else:
        print(f"Measures file not found: {measures_file}")
        return
    
    # Print the loaded measures
    num_images = len(test_images)
    print(f"Loaded {num_images} test images.")
    img_shape = test_images[0].shape if num_images > 0 else None
    print(f"Image shape: {img_shape}")
    measures_shape = measures.shape if measures is not None else None
    print(f"Measures shape for image 0: {measures_shape if measures_shape is not None else 'No measures shape for image 0'}")
    measures_sample = measures[0] if measures is not None and len(measures) > 0 else None
    print(f"Sample measures for image 0: {measures_sample if measures_sample is not None else 'No measures sample for image 0'}")

    # Select image and process it
    for frame in test_images:

        blur = cv2.GaussianBlur(frame, (9,9), 0)


        kernel = np.ones((5, 5), np.uint8)
        closing = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, kernel)

        gray = cv2.cvtColor(closing, cv2.COLOR_BGR2GRAY)

        # Isolate the bright LEDs from the dim room.
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # Close small gaps to ensure lines are continuous
        kernel = np.ones((10, 10), np.uint8)
        closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        edges = cv2.Canny(closing, threshold1=50, threshold2=150)
        
        # Probabilistic Hough Line Transform to find line segments
        # Parameters will need tuning based on how thick the gates appear
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi/180, threshold=40, 
                                minLineLength=30, maxLineGap=15)
        
        unique_corners = []

        if lines is not None:

            # --- 4. Corner Logic & Classification ---
            corners = []
            
            # Helper to find intersection point of two line segments
            def find_intersection(line1, line2):
                x1, y1, x2, y2 = line1[0]
                x3, y3, x4, y4 = line2[0]
                
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if denom == 0:
                    return None # Lines are parallel
                
                px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
                py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
                
                # Ensure intersection occurs relatively close to the line segments
                tolerance = 20
                if (min(x1, x2) - tolerance <= px <= max(x1, x2) + tolerance and
                    min(y1, y2) - tolerance <= py <= max(y1, y2) + tolerance and
                    min(x3, x4) - tolerance <= px <= max(x3, x4) + tolerance and
                    min(y3, y4) - tolerance <= py <= max(y3, y4) + tolerance):
                    return (int(px), int(py))
                return None

            # Helper to determine the outward vector direction of a line from the intersection
            def get_direction(x, y, line):
                x1, y1, x2, y2 = line[0]
                # Compare distances to find which endpoint extends away from the corner
                if (x1 - x)**2 + (y1 - y)**2 > (x2 - x)**2 + (y2 - y)**2:
                    return x1 - x, y1 - y
                else:
                    return x2 - x, y2 - y

            # Check intersections of all pairs of lines
            for l1, l2 in itertools.combinations(lines, 2):
                pt = find_intersection(l1, l2)
                if pt:
                    px, py = pt
                    
                    # Ignore out-of-bounds intersections
                    if px < 0 or py < 0 or px >= frame.shape[1] or py >= frame.shape[0]:
                        continue
                    
                    # Determine the vector directions of the intersecting lines
                    v1 = get_direction(px, py, l1)
                    v2 = get_direction(px, py, l2)
                    
                    # Separate into mostly horizontal and mostly vertical vectors
                    if abs(v1[0]) > abs(v1[1]): 
                        v_horiz, v_vert = v1, v2
                    else:
                        v_horiz, v_vert = v2, v1
                    
                    # Filter out lines that don't roughly form a 90-degree corner
                    if abs(v_horiz[0]) < 1e-5 or abs(v_vert[1]) < 1e-5:
                        continue
                    
                    # Classify based on the vectors. 
                    # In OpenCV, +X is Right, and +Y is Down.
                    right = v_horiz[0] > 0
                    down = v_vert[1] > 0
                    
                    if right and down:
                        corner_type = "TL"
                    elif not right and down:
                        corner_type = "TR"
                    elif right and not down:
                        corner_type = "BL"
                    else:
                        corner_type = "BR"
                        
                    corners.append({"point": (px, py), "type": corner_type})
                    
            # --- 5. Simplistic Non-Maximum Suppression ---
            # Because lines have thickness, multiple intersections will be found at the same corner.
            # Group them by proximity and type.
            unique_corners = []
            for c in corners:
                is_duplicate = False
                for uc in unique_corners:
                    if c["type"] == uc["type"]:
                        dist = math.hypot(c["point"][0] - uc["point"][0], c["point"][1] - uc["point"][1])
                        if dist < 15: # Pixel distance threshold for merging
                            is_duplicate = True
                            break
                if not is_duplicate:
                    unique_corners.append(c)





        
        # Draw lines on a copy of the original image for visualization
        line_image = frame.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # Draw detected corners
        for corner in unique_corners if unique_corners is not None else []:
            cv2.circle(line_image, corner["point"], 5, (255, 0, 0), -1)
            cv2.putText(line_image, corner["type"], (corner["point"][0] + 10, corner["point"][1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


        # Display image processing steps side by side
        steps = [frame, blur, gray, thresh, closing, edges, line_image]
        processed_steps = []
        for step in steps:
            if len(step.shape) == 2: # Check if the image is grayscale (has only 2 dimensions)
                step_bgr = cv2.cvtColor(step, cv2.COLOR_GRAY2BGR) # Convert grayscale to BGR
            else:
                step_bgr = step # It's already BGR, just keep it as is
            processed_steps.append(step_bgr)

        # Now they all have 3 channels and can be stacked
        combined_view = np.hstack(processed_steps)
        cv2.imshow('Image Processing Pipeline', combined_view)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    pass

if __name__ == "__main__":
    main()