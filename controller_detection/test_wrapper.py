# take recording .avi and .csv with row for each frame (time, x,y,z, roll, pitch, yaw,battery)
# mport DetectionController and run control command function for each frame
import cv2
import pandas as pd
import detection_controller_Vincent as DC
#import os

def test_wrapper(video_path, csv_path):

    # Read the video file
    cap = cv2.VideoCapture(video_path)

    # Read the CSV file
    df = pd.read_csv(csv_path)

    frame_index = 0
    previous_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get the corresponding row from the CSV for the current frame
        if frame_index < len(df):
            row = df.iloc[frame_index]
            sensor_data = {
                'x_global': row['x'],
                'y_global': row['y'],
                'z_global': row['z'],
                'roll': row['roll'],
                'pitch': row['pitch'],
                'yaw': row['yaw'],
                'battery': row['battery'],
            }
            dt = row['timestamp'] - previous_time
            previous_time = row['timestamp']

            # Call the control command function with the extracted data
            print("get_command", DC.get_command(sensor_data, frame, dt))

        # wait for a short period to simulate real-time processing (optional)
        #cv2.waitKey(1000) 
        # cv2.waitKey(333)  # Adjust the delay as needed - 333 ms simulates ~3 FPS

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":

    video_path = '../saved_recordings/2026-05-13-14-55-19.avi'
    csv_path = '../saved_recordings/2026-05-13-14-55-19.csv'

    #script_dir = os.path.dirname(os.path.abspath(__file__))
    #video_path = os.path.join(script_dir, r"..\saved_recordings\2026-05-13-14-55-19.avi")
    #csv_path = os.path.join(script_dir, r"..\saved_recordings\2026-05-13-14-55-19.csv")

    test_wrapper(video_path, csv_path)

