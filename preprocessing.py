import os
import cv2
import json
import numpy as np
from tqdm import tqdm
from glob import glob
import mediapipe as mp

def crop_lip_with_mediapipe(video_path, json_path, save_dir, resize=(128, 128), fps=30):
    os.makedirs(save_dir, exist_ok=True)

    with open(json_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)[0]

    sentence_info = metadata["Sentence_info"]
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    cap = cv2.VideoCapture(video_path)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

    skipped_count = 0

    for sentence in tqdm(sentence_info, desc=f"Processing {video_filename}"):
        sent_id = sentence["ID"]
        start_frame = int(sentence["start_time"] * fps)
        end_frame = int(sentence["end_time"] * fps)

        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        invalid = False

        for frame_idx in range(start_frame, end_frame):
            success, frame = cap.read()
            if not success or frame is None:
                print(f"⚠️ frame 읽기 실패: frame {frame_idx} (영상: {video_filename})")
                invalid = True
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if not results.multi_face_landmarks:
                print(f"⚠️ 얼굴 인식 실패: frame {frame_idx}")
                invalid = True
                break

            landmarks = results.multi_face_landmarks[0].landmark
            lip_indices = list(range(48, 68))  # 입 주위 landmark index

            xs = [int(landmarks[i].x * w) for i in lip_indices]
            ys = [int(landmarks[i].y * h) for i in lip_indices]
            x1, x2 = max(0, min(xs)), min(w, max(xs))
            y1, y2 = max(0, min(ys)), min(h, max(ys))

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
    face_mesh.close()
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

# video_folder = "input_videos"
# json_folder = "input_texts"
# npy_dir = "processed_dataset/npy"
# text_dir = "processed_dataset/text"
# wav_dir = "input_videos"    

# crop_lip_all(json_folder, video_folder, npy_dir)
# save_all_sentence_labels(json_folder, text_dir)