# Luxembourgish Lip-Reading BSP

This repository contains preprocessing experiments for a Bachelor Semester Project (BSP) on Luxembourgish visual speech recognition (lip-reading).

## Project Goal

The objective is to preprocess video data into face/mouth regions that can later be used for training a lightweight lip-reading model.

## Files

- `face_cropper.py`
  - Adapted external preprocessing code used for face extraction and tracking.
  - Source:
    https://github.com/MarshallT-99/VALLR

- `test_face_crop_video.py`
  - Loads a video, applies frame-by-frame face extraction, and saves processed outputs into `.npz` format.

- `view_npz.py`
  - Loads and visualizes saved `.npz` preprocessing outputs for verification.

- `test.py`
  - Simple environment and library test script.

## Libraries Used

- PyTorch
- OpenCV
- MediaPipe
- NumPy
- Pandas
- jiwer

## Environment Setup

Create and activate a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate

pip install torch opencv-python mediapipe numpy pandas jiwer

python test_face_crop_video.py



External Code Notice:
face_cropper.py contains adapted code from the VALLR repository and is used strictly for preprocessing purposes within this BSP project. Proper attribution is provided both in this repository and in the final academic report.