# utils/landmark_utils.py

import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)
MOUTH_IDX = [61, 291, 13, 14]

def extract_mouth_landmarks(image):
    results = mp_face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    landmarks = results.multi_face_landmarks[0].landmark
    return [(landmarks[i].x, landmarks[i].y) for i in MOUTH_IDX]
