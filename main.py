import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import torch
import logging

logging.basicConfig(level=logging.INFO)

def crop_lip(video_path, json_path, save_dir, resize=(128, 128), fps=30):
    """
    메모리 최적화 + bbox 유효성 검사 + skip 문장 카운트
    """
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)[0]

    sentence_info = metadata["Sentence_info"]
    lip_bboxes = metadata["Bounding_box_info"]["Lip_bounding_box"]["xtl_ytl_xbr_ybr"]

    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    skipped_count = 0

    for sentence in tqdm(sentence_info, desc=f"Processing {video_filename}"):
        sent_id = sentence["ID"]
        start_frame = int(sentence["start_time"] * fps)
        end_frame = int(sentence["end_time"] * fps)

        if start_frame >= len(lip_bboxes):
            print(f"⚠️ 문장 ID {sent_id}의 start_frame({start_frame})이 bbox 개수({len(lip_bboxes)})보다 큽니다. 건너뜀.")
            skipped_count += 1
            continue

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        invalid = False

        for frame_idx in range(start_frame, min(end_frame, len(lip_bboxes))):
            success, frame = cap.read()
            if not success or frame is None:
                print(f"⚠️ frame 읽기 실패: frame {frame_idx} (영상: {video_filename})")
                invalid = True
                break

            x1, y1, x2, y2 = map(int, lip_bboxes[frame_idx])
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                print(f"⚠️ 잘못된 bbox: frame {frame_idx}, box=({x1}, {y1}, {x2}, {y2}) → 문장 전체 건너뜀")
                invalid = True
                break

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                print(f"⚠️ 빈 crop 발생: frame {frame_idx}, 문장 전체 건너뜀")
                invalid = True
                break

            crop_resized = cv2.resize(crop, resize)
            frames.append(crop_resized)

        if invalid or not frames:
            skipped_count += 1
            continue

        arr = np.stack(frames)
        save_path = os.path.join(save_dir, f"{video_filename}_sentence_{sent_id}.npy")
        np.save(save_path, arr)

    cap.release()
    print(f"✅ {video_filename}: 모든 문장 crop 완료 (스킵된 문장 수: {skipped_count})")

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
        if os.path.exists(video_path):
            crop_lip(video_path, json_path, save_dir)
        else:
            print(f"❌ 영상 파일 없음: {filename}.mp4")

def save_all_sentence_labels(json_folder, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    json_files = glob(os.path.join(json_folder, "*.json"))
    for json_path in json_files:
        save_sentence_labels(json_path, save_dir)

def debug_dataloader(train_loader):
    print("🧪 첫 배치 로딩 시도 중...")
    try:
        batch = next(iter(train_loader))
        print("✅ 첫 배치 로딩 성공")
    except Exception as e:
        print(f"❌ 첫 배치 로딩 중 오류 발생: {e}")
