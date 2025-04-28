import tensorflow as tf
import tensorflow_hub as hub
import cv2
import pandas as pd

# Load the pose estimation model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")

# Function to estimate pose
def estimate_pose(frame):
    input_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    input_frame = tf.image.resize(input_frame, (192, 192))
    input_frame = input_frame[tf.newaxis,...]
    result = model(input_frame)
    return result

# Read the video or image
cap = cv2.VideoCapture('your_video.mp4')

# Process each frame of the video
frame_data = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    pose = estimate_pose(frame)
    frame_data.append({"frame": frame, "pose": pose})

# Convert frame data to a pandas DataFrame for easy analysis
df = pd.DataFrame(frame_data)
df.to_csv("pose_estimation_data.csv", index=False)
