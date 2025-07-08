import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

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

print("📸 カメラ起動中。'q'キーで終了")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ フレームが取得できませんでした")
        break

    # MediapipeはRGB画像を要求する
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Pose 検出
    detection_result = detector.detect(mp_image)

    # Annotated フレームを描画
    annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)

    # 表示
    cv2.imshow('Pose Detection (Press q to Quit)', annotated_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 後始末
cap.release()
cv2.destroyAllWindows()
