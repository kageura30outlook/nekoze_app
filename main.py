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

# === DRAW FUNCTION (改善済: 線が太くて見やすいスタイル) ===
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        # 線の色・太さを指定して見やすくする
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=3, circle_radius=2),
            connection_drawing_spec=solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2))
    return annotated_image

# === VIDEO INPUT/OUTPUT SETUP ===
input_path = "/Users/Kageura/Desktop/input.mp4"
output_path = "annotated_output.mp4"
frame_interval = 5  # フレーム間隔

cap = cv2.VideoCapture(input_path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_idx = 0
last_detection_result = None  # 前回の検出結果を保持

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Nフレームごとに検出、それ以外は前の結果を再利用
    if frame_idx % frame_interval == 0:
        detection_result = detector.detect(mp_image)
        last_detection_result = detection_result
    else:
        detection_result = last_detection_result

    if detection_result:
        annotated_frame = draw_landmarks_on_image(rgb_frame, detection_result)
    else:
        annotated_frame = frame  # 検出失敗時はそのまま

    out.write(annotated_frame)
    frame_idx += 1

cap.release()
out.release()
print("✅ 書き出し完了：annotated_output.mp4")
