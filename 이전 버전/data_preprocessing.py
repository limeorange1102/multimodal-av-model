# 📦 data_preprocessing.py (LandmarkEncoder 버전)
# n개의 단일화자 영상에서 nC2 쌍의 멀티화자 샘플 생성 + 입 landmark 좌표 시퀀스 저장 + 단어 텍스트 저장

import os
import cv2
import itertools
from glob import glob
from moviepy.editor import VideoFileClip, clips_array
from pydub import AudioSegment
import numpy as np
import json

from utils.landmark_utils import extract_mouth_landmarks  # ✅ 외부 유틸 모듈에서 임포트

# ✅ 경로 설정
input_dir = "input_videos"
output_root = "sample_pairs"
os.makedirs(output_root, exist_ok=True)

# ✅ 모든 영상 파일 경로
video_paths = sorted(glob(os.path.join(input_dir, "*.mp4")))

# ✅ 영상 이름 → 단어 레이블 불러오기
labels = {}
with open(os.path.join(input_dir, "labels.txt"), "r", encoding="utf-8") as f:
    for line in f:
        name, word = line.strip().split()
        labels[name] = word

# ✅ 영상 조합
group_id = 0
for path1, path2 in itertools.combinations(video_paths, 2):
    name1 = os.path.splitext(os.path.basename(path1))[0]
    name2 = os.path.splitext(os.path.basename(path2))[0]
    word1 = labels.get(name1, "unknown")
    word2 = labels.get(name2, "unknown")

    group_folder = os.path.join(output_root, f"sample_{group_id:03d}")
    os.makedirs(group_folder, exist_ok=True)

    # ✅ Step 1: 영상 병합
    clip1 = VideoFileClip(path1).resize(height=224)
    clip2 = VideoFileClip(path2).resize(height=224)
    min_w = min(clip1.w, clip2.w)
    clip1 = clip1.resize(width=min_w)
    clip2 = clip2.resize(width=min_w)

    merged_clip = clips_array([[clip1, clip2]])
    merged_path = os.path.join(group_folder, "combined.mp4")
    merged_clip.write_videofile(merged_path, codec="libx264", audio=True)

    # ✅ Step 2: 프레임 분리 + 입모양 landmark 추출
    cap = cv2.VideoCapture(merged_path)
    landmarks_A = []
    landmarks_B = []
    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        H, W, _ = frame.shape
        mid = W // 2
        frame_A = frame[:, :mid]
        frame_B = frame[:, mid:]

        lm_A = extract_mouth_landmarks(frame_A)
        lm_B = extract_mouth_landmarks(frame_B)

        if lm_A:
            landmarks_A.append(lm_A)
        if lm_B:
            landmarks_B.append(lm_B)
        idx += 1
    cap.release()

    # landmark 좌표 저장
    with open(os.path.join(group_folder, "landmarks_A.json"), "w", encoding="utf-8") as f:
        json.dump(landmarks_A, f)
    with open(os.path.join(group_folder, "landmarks_B.json"), "w", encoding="utf-8") as f:
        json.dump(landmarks_B, f)

    # ✅ Step 3: 오디오 병합
    audio1 = AudioSegment.from_file(path1)
    audio2 = AudioSegment.from_file(path2)
    min_len = min(len(audio1), len(audio2))
    mixed = audio1[:min_len].overlay(audio2[:min_len])
    mixed.export(os.path.join(group_folder, "mixed.wav"), format="wav")

    # ✅ Step 4: 라벨 저장
    with open(os.path.join(group_folder, "gt_A.txt"), "w", encoding="utf-8") as f:
        f.write(word1 + "\n")
    with open(os.path.join(group_folder, "gt_B.txt"), "w", encoding="utf-8") as f:
        f.write(word2 + "\n")

    print(f"✅ sample_{group_id:03d} 완료")
    group_id += 1
