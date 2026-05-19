import cv2
import numpy as np
import pandas as pd
import colorsys

import detection_controller as DC

def extract_frames(video_path, frame_indices):
    cap = cv2.VideoCapture(video_path)

    frames = []
    frame_set = set(frame_indices)  # faster lookup
    max_idx = max(frame_indices)

    idx = 0
    success = True

    while success and idx <= max_idx:
        success, frame = cap.read()

        if not success:
            break

        if idx in frame_set:
            frames.append(frame)

        idx += 1

    cap.release()
    return frames

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return (b, g, r)  # OpenCV uses BGR

def generate_color_dict(data):
    keys = list(data.keys())
    n = len(keys)

    color_dict = {}

    for i, key in enumerate(keys):
        hue = i / n
        rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)

        # convert RGB floats to hex color
        hex_color = '#{:02x}{:02x}{:02x}'.format(
            int(rgb[0] * 255),
            int(rgb[1] * 255),
            int(rgb[2] * 255)
        )

        color_dict[key] = hex_to_bgr(hex_color)

    return color_dict

def main():
    controller = DC.DetectionController()

    video_path = '../saved_recordings/2026-05-13-14-55-19.avi'
    csv_path = '../saved_recordings/2026-05-13-14-55-19.csv'

    # Select indices
    # indices = [7, 11, 15, 17, 19, 20, 21, 22, 23, 25, 26, 27, 28, 29]
    # indices = [11, 15, 22] # 11
    indices = range(7, 30)
    indices = range(29, 35)

    # Read the video file
    images = extract_frames(video_path, indices)

    # Read the CSV file
    df = pd.read_csv(csv_path).loc[indices, ['x', 'y', 'z', 'yaw', 'timestamp']].to_numpy()

    # detect gates and add to gate candidates
    for i, img in enumerate(images):
        controller.current_timestamp = df[i][4]
        print(f"\nProcessing frame {indices[i]} at timestamp {controller.current_timestamp:.2f}s")

        detections = controller.detect(img)
        
        P = controller._compute_world_to_camera_projection(df[i][0:3], [0,0,df[i][3]])
        controller.associate_and_update(detections, P)

        # Draw detected corners
        colors = generate_color_dict(controller.gate_candidates)
        for candidate_id, candidate in controller.gate_candidates.items():
            color = colors.get(candidate_id, (0, 255, 0))
            for obs_list in candidate.observations.values():
                for obs in obs_list:
                    if obs.timestamp != controller.current_timestamp:
                        continue
                    if obs.conf < 0.95:
                        cv2.circle(img, tuple(obs.uv.astype(int)), 5, (0, 255, 255), -1)
                    else:
                        cv2.circle(img, tuple(obs.uv.astype(int)), 5, color, -1)

            uvs = [
                obs.uv
                for obs_list in candidate.observations.values()
                for obs in obs_list
                if obs.timestamp == controller.current_timestamp
            ]
            if len(uvs) == 0:
                continue  # nothing to draw / label
            uvs = np.array(uvs, dtype=float)
            detection_center = uvs.mean(axis=0)

            cv2.putText(img, f"G{candidate_id}G", (int(detection_center[0]), int(detection_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
        print(f"Drone position: x: {df[i][0]:.2f}, y: {df[i][1]:.2f}, z: {df[i][2]:.2f}, yaw: {df[i][2]:.2f}")

        cv2.imshow('Image Processing Pipeline', img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

    # triangulate
    print(f"\nNumber of candidates: {len(controller.gate_candidates)}\n")

    for i, candidate in enumerate(controller.gate_candidates.values()):

        controller.triangulate(candidate)
        gate, validation_conf = controller.validate_gate_pos(candidate.corners_world)
        candidate.val_conf = validation_conf

        if candidate.corners_world:
            lines = [f"\nCandidate {i} triangulated corners (world):"]
            for cid in (DC.CornerID.BL, DC.CornerID.TL, DC.CornerID.TR, DC.CornerID.BR):
                pos = candidate.corners_world.get(cid)
                if pos is None:
                    lines.append(f"  {cid.name:>3}: None")
                else:
                    lines.append(f"  {cid.name:>3}: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")

            if candidate.val_conf is not None:
                lines.append(f" Validation confidence: {candidate.val_conf:.2f}")
            print("\n".join(lines))

            if len(candidate.corners_world) == 4:
                # Require explicit CornerID keys (BL, TL, TR, BR)
                bl = candidate.corners_world[DC.CornerID.BL]
                tl = candidate.corners_world[DC.CornerID.TL]
                tr = candidate.corners_world[DC.CornerID.TR]
                br = candidate.corners_world[DC.CornerID.BR]
                pts = np.vstack([bl, tl, tr, br])
                gate_center = pts.mean(axis=0)

                projection = tr - tl
                projection[2] = 0.0  # ignore Z component since gates are vertical
                theta = float(np.arctan2(projection[1], projection[0]))

                print(f" Gate center: {gate_center}, {theta}")

        else:
            print(f"Candidate {i} triangulated corners (world): None") 

    # compare result



if __name__ == "__main__":
    main()
