import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import mediapipe as mp

def crop_lip(video_path, output_path, frame_indices):
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True)

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    crops = []
    for frame_idx in frame_indices:
        if frame_idx >= total_frames:
            print(f"⚠️ frame {frame_idx} out of bounds for {video_path}")
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print(f"⚠️ Failed to read frame {frame_idx} in {video_path}")
            continue

        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            print(f"⚠️ No face detected in frame {frame_idx} of {video_path}")
            continue

        landmarks = results.multi_face_landmarks[0]
        lip_points = [
            (int(lm.x * w), int(lm.y * h))
            for i, lm in enumerate(landmarks.landmark)
            if 61 <= i <= 88
        ]

        if len(lip_points) == 0:
            print(f"⚠️ No lip points in frame {frame_idx} of {video_path}")
            continue

        x_coords, y_coords = zip(*lip_points)
        x_min, x_max = max(min(x_coords) - 10, 0), min(max(x_coords) + 10, w)
        y_min, y_max = max(min(y_coords) - 10, 0), min(max(y_coords) + 10, h)

        lip_crop = frame[y_min:y_max, x_min:x_max]
        if lip_crop.size == 0:
            print(f"⚠️ Empty crop in frame {frame_idx} of {video_path}")
            continue

        lip_crop = cv2.resize(lip_crop, (128, 128))
        crops.append(lip_crop)

    cap.release()

    if len(crops) > 0:
        crops = np.stack(crops, axis=0)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.save(output_path, crops)
        print(f"✅ Saved crop: {output_path} ({crops.shape[0]} frames)")
    else:
        print(f"❌ No valid crops found for {video_path}, skipping.")

def save_sentence_labels(json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)[0]

    video_filename = os.path.splitext(os.path.basename(json_path))[0]
    sentence_info = metadata["Sentence_info"]

    for sentence in sentence_info:
        sent_id = sentence["ID"]
        text = sentence["sentence_text"].strip()
        save_path = os.path.join(save_dir, f"{video_filename}_sentence_{sent_id}.txt")
        with open(save_path, 'w', encoding='utf-8') as f_out:
            f_out.write(text + "\n")

    print(f"✅ {len(sentence_info)}개의 문장 텍스트 라벨을 저장했습니다: {save_dir}")

def build_data_list(json_folder, npy_dir, text_dir, wav_dir):
    data_list = []
    for filename in os.listdir(json_folder):
        if not filename.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, filename)
        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)[0]

        base_name = os.path.splitext(filename)[0]
        wav_path = os.path.join(wav_dir, base_name + ".wav")

        for sent in metadata["Sentence_info"]:
            sent_id = sent["ID"]
            lip_path = os.path.join(npy_dir, f"{base_name}_sentence_{sent_id}.npy")
            text_path = os.path.join(text_dir, f"{base_name}_sentence_{sent_id}.txt")

            if not os.path.exists(lip_path) or not os.path.exists(text_path):
                print(f"⚠️ 파일 누락 → 제외: {lip_path}, {text_path}")
                continue

            data_list.append({
                "lip_path": lip_path,
                "text_path": text_path,
                "audio_path": wav_path,
                "start_time": float(sent["start_time"]),
                "end_time": float(sent["end_time"]),
            })

    return data_list

def crop_lip_all(json_folder, video_folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob(os.path.join(json_folder, "*.json"))
    for json_path in json_files:
        filename = os.path.splitext(os.path.basename(json_path))[0]
        video_path = os.path.join(video_folder, filename + ".mp4")

        if not os.path.exists(video_path):
            print(f"❌ 영상 파일 없음: {filename}.mp4")
            continue

        with open(json_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)[0]

        for sentence in metadata["Sentence_info"]:
            sent_id = sentence["ID"]
            start_frame = int(sentence["start_frame"])
            end_frame = int(sentence["end_frame"])
            frame_indices = list(range(start_frame, end_frame + 1))

            output_path = os.path.join(save_dir, f"{filename}_sentence_{sent_id}.npy")
            crop_lip(video_path, output_path, frame_indices)

def save_all_sentence_labels(json_folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob(os.path.join(json_folder, "*.json"))
    for json_path in json_files:
        save_sentence_labels(json_path, save_dir)

video_folder = "input_videos"
json_folder = "input_texts"
npy_dir = "processed_dataset/npy"
text_dir = "processed_dataset/text"
wav_dir = "input_videos"

crop_lip_all(json_folder, video_folder, npy_dir)
