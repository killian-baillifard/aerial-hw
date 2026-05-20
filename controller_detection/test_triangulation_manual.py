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

    # Explicitly define two (or more) frame indices to use for manual triangulation
    indices = [11, 15, 21]
    # indices = [22, 25, 27, 36, 37]
    # indices = range(11, 25)
    # 0.80, y: -0.93, z: 1.39, yaw: 1.39
    
    # === load data ===
    images = extract_frames(video_path, indices)
    df = pd.read_csv(csv_path).loc[indices, ['x', 'y', 'z', 'yaw', 'timestamp']].to_numpy()

    selected_candidates = []

    for i, img in enumerate(images):
        controller.current_timestamp = df[i][4]
        frame_idx = indices[i]
        print(f"\nProcessing frame {frame_idx} at timestamp {controller.current_timestamp:.2f}s")

        # Clear candidates so old frames don't automatically associate with this one via timestamps
        controller.gate_candidates.clear()

        # === run detection ===
        detections = controller.detect(img)
        
        P = controller._compute_world_to_camera_projection(df[i][0:3], [0, 0, df[i][3]])
        controller.associate_and_update(detections, P)

        # === visualize detection ===
        img_display = img.copy()
        colors = generate_color_dict(controller.gate_candidates)
        
        for candidate_id, candidate in controller.gate_candidates.items():
            color = colors.get(candidate_id, (0, 255, 0))
            for obs_list in candidate.observations.values():
                for obs in obs_list:
                    if obs.conf < 0.95:
                        cv2.circle(img_display, tuple(obs.uv.astype(int)), 5, (0, 255, 255), -1)
                    else:
                        cv2.circle(img_display, tuple(obs.uv.astype(int)), 5, color, -1)

            uvs = [
                obs.uv
                for obs_list in candidate.observations.values()
                for obs in obs_list
            ]
            if len(uvs) == 0:
                continue 
            uvs = np.array(uvs, dtype=float)
            detection_center = uvs.mean(axis=0)

            # Label with ID so you know what to type in the console
            cv2.putText(img_display, f"ID: {candidate_id}", (int(detection_center[0]), int(detection_center[1])), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        print(f"Drone position: x: {df[i][0]:.2f}, y: {df[i][1]:.2f}, z: {df[i][2]:.2f}, yaw: {df[i][3]:.2f}")
        print(f"Available Candidate IDs in this frame: {list(controller.gate_candidates.keys())}")
        
        cv2.imshow('Image Processing Pipeline - Manual Selection', img_display)
        cv2.waitKey(200)  # Short pause to force UI window refresh before console blocks

        # === after each visualisation select detection and only add this to one gate candidate object ===
        while True:
            user_input = input(f"Enter the Candidate ID to keep for frame {frame_idx} (or 's' to skip frame): ").strip()
            if user_input.lower() == 's':
                break
            
            # Handle both integer and string dict keys safely depending on underlying controller implementation
            chosen_id = int(user_input) if user_input.isdigit() else user_input
            if chosen_id in controller.gate_candidates:
                selected_candidates.append(controller.gate_candidates[chosen_id])
                print(f"Stored candidate {chosen_id} from frame {frame_idx}.")
                break
            elif str(chosen_id) in controller.gate_candidates:
                selected_candidates.append(controller.gate_candidates[str(chosen_id)])
                print(f"Stored candidate {chosen_id} from frame {frame_idx}.")
                break
            else:
                print("Invalid ID. Look at the popup window and select a valid visible ID.")

    cv2.destroyAllWindows()

    if len(selected_candidates) < 2:
        print("\nError: You must select detections from at least 2 frames to perform triangulation.")
        return

    # Merge all selected independent frame candidates into a single master candidate object
    master_candidate = selected_candidates[0]
    for extra_candidate in selected_candidates[1:]:
        for corner_id, obs_list in extra_candidate.observations.items():
            if corner_id not in master_candidate.observations:
                master_candidate.observations[corner_id] = []
            master_candidate.observations[corner_id].extend(obs_list)

    print(f"\nSuccessfully combined selections from {len(selected_candidates)} frames into one candidate object.")

    # === after that triangulate ===
    print("\nRunning triangulation on the unified candidate object...")
    controller.triangulate(master_candidate)
    gate, validation_conf = controller.validate_gate_pos(master_candidate.corners_world)
    master_candidate.val_conf = validation_conf

    if master_candidate.corners_world:
        lines = ["\nManual Selection Triangulated corners (world):"]
        for cid in (DC.CornerID.BL, DC.CornerID.TL, DC.CornerID.TR, DC.CornerID.BR):
            pos = master_candidate.corners_world.get(cid)
            if pos is None:
                lines.append(f"  {cid.name:>3}: None")
            else:
                lines.append(f"  {cid.name:>3}: [{pos[0]:8.3f}, {pos[1]:8.3f}, {pos[2]:8.3f}]")

        if master_candidate.val_conf is not None:
            lines.append(f" Validation confidence: {master_candidate.val_conf:.2f}")
        print("\n".join(lines))

        if len(master_candidate.corners_world) == 4:
            bl = master_candidate.corners_world[DC.CornerID.BL]
            tl = master_candidate.corners_world[DC.CornerID.TL]
            tr = master_candidate.corners_world[DC.CornerID.TR]
            br = master_candidate.corners_world[DC.CornerID.BR]
            pts = np.vstack([bl, tl, tr, br])
            gate_center = pts.mean(axis=0)

            v1 = tl - bl
            v2 = br - bl
            gate_normal = np.cross(v1, v2)
            theta = float(np.arctan2(gate_normal[1], gate_normal[0]))

            print(f" Gate center: {gate_center}, {theta}")
    else:
        print("Manual Triangulated corners (world): None") 

if __name__ == "__main__":
    main()
