import cv2
import numpy as np
from face_cropper import FaceCropper
import os

VIDEO_PATH = "videos/test.mp4"
OUTPUT_PATH = "outputs/test_face_crops.npz"

TARGET_SIZE = (96, 96)

cropper = FaceCropper(
    landmark_detector_static_image_mode=FaceCropper.TRACKING_MODE
)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frames = []
frame_index = 0
failed_frames = 0

while True:
    ret, frame_bgr = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    faces = cropper.get_faces(
        frame_rgb,
        remove_background=False,
        correct_roll=True
    )

    if len(faces) == 0:
        failed_frames += 1
        frame_index += 1
        continue

    face = faces[0]
    face = cv2.resize(face, TARGET_SIZE)

    frames.append(face)
    frame_index += 1

cap.release()

frames = np.array(frames, dtype=np.uint8)


os.makedirs("outputs", exist_ok=True)
np.savez_compressed(
    OUTPUT_PATH,
    frames=frames,
    video_path=VIDEO_PATH,
    total_frames=frame_index,
    kept_frames=len(frames),
    failed_frames=failed_frames
)

print("Done.")
print("Total frames:", frame_index)
print("Saved frames:", len(frames))
print("Failed frames:", failed_frames)
print("Output shape:", frames.shape)
print("Saved to:", OUTPUT_PATH)