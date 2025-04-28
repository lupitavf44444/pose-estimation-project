import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import pandas as pd

# Load MoveNet MultiPose model from TF Hub
model = hub.load("https://tfhub.dev/google/movenet/multipose/lightning/1")
input_size = 256

def detect_poses(img):
    img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), input_size, input_size)
    input_img = tf.cast(img, dtype=tf.int32)
    outputs = model.signatures['serving_default'](input_img)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def classify_pose(keypoints):
    # keypoints: (17, 3) - x, y, score
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_hip = keypoints[11]
    right_hip = keypoints[12]
    left_knee = keypoints[13]
    right_knee = keypoints[14]

    avg_shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    avg_hip_y = (left_hip[1] + right_hip[1]) / 2
    avg_knee_y = (left_knee[1] + right_knee[1]) / 2

    if abs(avg_hip_y - avg_knee_y) < 0.05:
        return "sitting"
    elif abs(avg_shoulder_y - avg_hip_y) < 0.1:
        return "lying down"
    else:
        return "standing"

# Load your video
video_path = 'your_video.mp4'  # replace with your video filename
cap = cv2.VideoCapture(video_path)

rows = []
frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detect_poses(input_frame)

    # Each frame can have multiple people
    for person_id, person in enumerate(detections[0]):
        if person[55] < 0.3:  # person score threshold (confidence)
            continue
        
        keypoints = person[:51].reshape((17, 3))
        label = classify_pose(keypoints)

        rows.append({
            'frame': frame_idx,
            'person_id': person_id,
            'label': label
        })

    frame_idx += 1

cap.release()

# Save DataFrame
df = pd.DataFrame(rows)
print(df.head())
df.to_csv('poses_labels.csv', index=False)
