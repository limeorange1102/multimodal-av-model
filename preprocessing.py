import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from glob import glob

def crop_lip(video_path, json_path, save_dir, resize=(128, 128), fps=30):
    """
    개선 버전: 모든 프레임을 메모리에 올려 디스크 접근 최소화 (속도 향상)
    """
    os.makedirs(save_dir, exist_ok=True)

    # JSON 로드
    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)[0]

    sentence_info = metadata["Sentence_info"]
    lip_bboxes = metadata["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)

    # ✅ 전체 프레임을 미리 메모리에 읽기
    all_frames = []
    success = True
    while success:
        success, frame = cap.read()
        if success:
            all_frames.append(frame)
    cap.release()

    # ✅ 문장 단위로 crop
    for sentence in tqdm(sentence_info, desc=f"Processing {video_filename}"):
        sent_id = sentence["ID"]
        start_frame = int(sentence["start_time"] * fps)
        end_frame = int(sentence["end_time"] * fps)

        frames = []
        for frame_idx in range(start_frame, min(end_frame, len(lip_bboxes), len(all_frames))):
            frame = all_frames[frame_idx]
            x1, y1, x2, y2 = map(int, lip_bboxes[frame_idx])
            crop = frame[y1:y2, x1:x2]
            crop_resized = cv2.resize(crop, resize)
            frames.append(crop_resized)

        if frames:
            arr = np.stack(frames)  # shape: [T, H, W, C]
            save_path = os.path.join(save_dir, f"{video_filename}_sentence_{sent_id}.npy")
            np.save(save_path, arr)

    print(f"✅ {video_filename}: 모든 문장의 crop이 저장되었습니다 (.npy).")

def save_sentence_labels(json_path, save_dir):
    """
    OLKAVS JSON 파일에서 각 문장의 텍스트 라벨을 추출해 .txt 파일로 저장한다.
    """
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
    """
    모든 JSON을 순회하며 문장 단위의 metadata 리스트를 생성한다.
    이 리스트는 RandomSentencePairDataset에 입력으로 사용된다.
    """
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
            data_list.append({
                "lip_path": os.path.join(npy_dir, f"{base_name}_sentence_{sent_id}.npy"),
                "text_path": os.path.join(text_dir, f"{base_name}_sentence_{sent_id}.txt"),
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
        if os.path.exists(video_path):
            crop_lip(video_path, json_path, save_dir)

def save_all_sentence_labels(json_folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob(os.path.join(json_folder, "*.json"))
    for json_path in json_files:
        save_sentence_labels(json_path, save_dir)

# crop_lip_all("input_texts", "input_videos", "processed_dataset/npy")
# save_all_sentence_labels("input_texts", "processed_dataset/text")
