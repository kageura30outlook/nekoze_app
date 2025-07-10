import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import time


# === POSE DETECTION SETUP ===
model_path = '/Users/Kageura/Documents/nekoze_app/pose_landmarker_lite.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)

# === DRAW FUNCTION（太くて緑の線）===
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

# === カメラ映像取得 ===
cap = cv2.VideoCapture(0)  # 0 = 内蔵カメラ（外部なら 1 などに変える）

frame_count = 0  # Add this before the while loop


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

# === CAMERA ===
cap = cv2.VideoCapture(0)

last_check = time.time()
CHECK_INTERVAL = 10  # seconds

print("📸 カメラ起動中。10秒ごとに姿勢チェックします。")

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

    # === Run posture check only every 10s ===
    if current_time - last_check >= CHECK_INTERVAL:
        last_check = current_time
        detection_result = detector.detect(mp_image)
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

        pose_landmarks_list = detection_result.pose_landmarks
        image_height, image_width, _ = frame.shape

        if len(pose_landmarks_list) > 0:
            landmarks = pose_landmarks_list[0]
            try:
                left_eye = landmarks[mp.solutions.pose.PoseLandmark.LEFT_EYE]
                left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
                right_eye = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_EYE]
                right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]

                lx, ly = int(left_eye.x * image_width), int(left_eye.y * image_height)
                lsx, lsy = int(left_shoulder.x * image_width), int(left_shoulder.y * image_height)
                rx, ry = int(right_eye.x * image_width), int(right_eye.y * image_height)
                rsx, rsy = int(right_shoulder.x * image_width), int(right_shoulder.y * image_height)

                left_dist = ((lx - lsx)**2 + (ly - lsy)**2)**0.5
                right_dist = ((rx - rsx)**2 + (ry - rsy)**2)**0.5


                if left_dist < 520 or right_dist < 520:
                    warning_img = cv2.imread("/Users/Kageura/Documents/nekoze_app/nekozedayo.png")
                    if warning_img is not None:
                        cv2.imshow("Posture Warning!", warning_img)
                    CHECK_INTERVAL = 2
                elif left_dist >520 or right_dist > 520:
                    CHECK_INTERVAL = 10
                    cv2.destroyWindow("Posture Warning!")

            except IndexError:
                print("⚠️ ランドマークの読み取りに失敗しました")
        else:
            print("⚠️ ポーズが検出されませんでした")

    # === Exit key ===
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()