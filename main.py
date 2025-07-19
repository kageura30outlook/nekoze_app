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

first = 0
cap = cv2.VideoCapture(0)
nose_dist = 126
frame_count = 0

# === SETUP ===
model_path = '/Users/Kageura/Documents/nekoze_app/pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2),
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2))
    return annotated_image

# === INITIAL BAD POSTURE CAPTURE ===
if st.button('Take photo of bad posture(Default would be 126)'):
    for i in range(10):
        print('Taking photo of bad posture in', i)
        time.sleep(1)
        
    ret, frame = cap.read()
    if not ret:
        print("⚠️ フレームが取得できませんでした")

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

        # Midpoint of shoulders
        mid_x = (left_shoulder.x + right_shoulder.x) / 2
        mid_y = (left_shoulder.y + right_shoulder.y) / 2

        nx, ny = int(nose.x * image_width), int(nose.y * image_height)
        mx, my = int(mid_x * image_width), int(mid_y * image_height)

        nose_dist = ((nx - mx) ** 2 + (ny - my) ** 2) ** 0.5
        p1_nose_dist = nose_dist
        print('Your baseline nose-shoulder distance is:', math.floor(p1_nose_dist))

if st.button('Start nekoze checker?'):
    # === MAIN CAMERA LOOP ===
    cap = cv2.VideoCapture(0)
    last_check = time.time()
    CHECK_INTERVAL = 1  # seconds

    print("📸 カメラ起動中。1秒ごとに姿勢チェックします。")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ フレームが取得できませんでした")
            break

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_time = time.time()
        detection_result = None
        annotated_frame = frame.copy()

        if current_time - last_check >= CHECK_INTERVAL:
            last_check = current_time
            detection_result = detector.detect(mp_image)
            annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

            pose_landmarks_list = detection_result.pose_landmarks
            image_height, image_width, _ = frame.shape

            if len(pose_landmarks_list) > 0:
                landmarks = pose_landmarks_list[0]
                try:
                    nose = landmarks[mp.solutions.pose.PoseLandmark.NOSE]
                    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

                    mid_x = (left_shoulder.x + right_shoulder.x) / 2
                    mid_y = (left_shoulder.y + right_shoulder.y) / 2

                    nx, ny = int(nose.x * image_width), int(nose.y * image_height)
                    mx, my = int(mid_x * image_width), int(mid_y * image_height)

                    nose_dist = ((nx - mx) ** 2 + (ny - my) ** 2) ** 0.5

                    if nose_dist < p1_nose_dist:
                        warning_img = cv2.imread("/Users/Kageura/Documents/nekoze_app/nekozedayo.png")
                        if warning_img is not None:
                            cv2.imshow("Posture Warning!", warning_img)
                        CHECK_INTERVAL = 2
                    elif nose_dist > p1_nose_dist:
                        CHECK_INTERVAL = 3
                        cv2.destroyWindow("Posture Warning!")

                    print("👃 Distance:", math.floor(nose_dist))

                except IndexError:
                    print("⚠️ ランドマークの読み取りに失敗しました")
            else:
                print("⚠️ ポーズが検出されませんでした")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
