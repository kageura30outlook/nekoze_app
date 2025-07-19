import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time
import math
import streamlit as st

st.set_page_config(page_title="Nekoze Posture Checker", layout="centered")
st.title("📸 Nekoze Posture Checker")
st.markdown("Check your posture in real-time using your webcam.")

# === Global State ===
if "baseline_dist" not in st.session_state:
    st.session_state["baseline_dist"] = 126  # Default value

cap = cv2.VideoCapture(0)
model_path = '/Users/Kageura/Documents/nekoze_app/pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

FRAME_WINDOW = st.image([])


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = rgb_image.copy()

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2))
    return annotated_image


if st.button('Take photo of good posture (baseline)'):
    st.info("Capturing image in 3 seconds...")
    time.sleep(3)

    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture frame")
    else:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = detector.detect(mp_image)
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

        pose_landmarks_list = detection_result.pose_landmarks
        image_height, image_width, _ = frame.shape

        if len(pose_landmarks_list) > 0:
            landmarks = pose_landmarks_list[0]
            nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
            left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

            mid_x = (left_shoulder.x + right_shoulder.x) / 2
            mid_y = (left_shoulder.y + right_shoulder.y) / 2
            nx, ny = int(nose.x * image_width), int(nose.y * image_height)
            mx, my = int(mid_x * image_width), int(mid_y * image_height)

            distance = ((nx - mx)**2 + (ny - my)**2)**0.5
            st.session_state["baseline_dist"] = distance
            st.success(f"Baseline nose-to-shoulder distance set: {int(distance)} pixels")
            FRAME_WINDOW.image(annotated_frame)
        else:
            st.warning("No pose detected. Please try again.")


if st.button("Start Posture Check"):
    CHECK_INTERVAL = 1.0
    st.warning("Press 'Stop' in the terminal or close the window to stop checking.")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera frame failed.")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        detection_result = detector.detect(mp_image)
        pose_landmarks_list = detection_result.pose_landmarks
        image_height, image_width, _ = frame.shape
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

        if len(pose_landmarks_list) > 0:
            try:
                landmarks = pose_landmarks_list[0]
                nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
                left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

                mid_x = (left_shoulder.x + right_shoulder.x) / 2
                mid_y = (left_shoulder.y + right_shoulder.y) / 2
                nx, ny = int(nose.x * image_width), int(nose.y * image_height)
                mx, my = int(mid_x * image_width), int(mid_y * image_height)

                nose_dist = ((nx - mx) ** 2 + (ny - my) ** 2) ** 0.5

                if nose_dist < st.session_state["baseline_dist"]:
                    st.error(f"🚫 Bad posture detected! Distance: {int(nose_dist)} < Baseline: {int(st.session_state['baseline_dist'])}")
                else:
                    st.success(f"✅ Good posture! Distance: {int(nose_dist)}")

            except IndexError:
                st.warning("Pose detection failed. Please retry.")

        FRAME_WINDOW.image(annotated_frame)
        time.sleep(CHECK_INTERVAL)

cap.release()
