import random
import torch
import numpy as np
import librosa
import cv2
import os

class MultiSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, sentence_list, tokenizer):
        self.sentence_list = sentence_list
        self.tokenizer = tokenizer

    def load_pair(self, s1,s2):
        # 1. 오디오 불러오기 및 자르기
        a1, sr = librosa.load(s1["audio_path"], sr=16000)
        a1 = a1[int(s1["start_time"] * sr):int(s1["end_time"] * sr)]

        a2, sr = librosa.load(s2["audio_path"], sr=16000)
        a2 = a2[int(s2["start_time"] * sr):int(s2["end_time"] * sr)]

        # 2. padding to same length
        len1 = len(a1)
        len2 = len(a2)
        max_len = max(len1, len2)

        a1 = np.pad(a1, (0, max_len - len1), mode="constant")
        a2 = np.pad(a2, (0, max_len - len2), mode="constant")

        # 3. 혼합
        mixed = a1 + a2
        mixed = mixed.astype(np.float32)
        mixed /= np.max(np.abs(mixed)) + 1e-6

        # 4. 마스크 생성
        mask1 = np.zeros(max_len, dtype=np.int64)
        mask2 = np.zeros(max_len, dtype=np.int64)

        min_len = min(len1, len2)
        mask1[:min_len] = 1
        mask2[:min_len] = 1

        if len1 > len2:
            mask1[len2:len1] = 2
        elif len2 > len1:
            mask2[len1:len2] = 2

        # 5. 입모양 npy 로딩
        try:
            lip1 = np.load(s1["lip_path"]).astype(np.float32)
            lip1 = lip1.mean(axis=-1)                               # grayscale (T, 128, 128)
            lip1 = np.array([cv2.resize(f, (96, 96)) for f in lip1]) # resize
            lip1 = lip1 / 255.0                                     # normalize
            lip1 = torch.tensor(lip1, dtype=torch.float32).unsqueeze(1)  # (T, 1, 96, 96)

            lip2 = np.load(s2["lip_path"]).astype(np.float32)
            lip2 = lip2.mean(axis=-1)                               # grayscale (T, 128, 128)
            lip2 = np.array([cv2.resize(f, (96, 96)) for f in lip2]) # resize
            lip2 = lip2 / 255.0                                     # normalize
            lip2 = torch.tensor(lip2, dtype=torch.float32).unsqueeze(1)  # (T, 1, 96, 96)
        except Exception as e:
            raise RuntimeError(f"❌ pair 로딩 실패: {e}")
        if lip1.shape[0] == 0 or lip2.shape[0] == 0:
            raise RuntimeError("❌ 비어 있는 입모양 npy 파일")

        # 6. 라벨 로딩
        with open(s1["text_path"], "r", encoding="utf-8") as f:
            label1 = self.tokenizer.encode(f.read().strip())
        with open(s2["text_path"], "r", encoding="utf-8") as f:
            label2 = self.tokenizer.encode(f.read().strip())

        # 7. 최종 반환
        return {
            "audio": mixed,
            "mask1": mask1,
            "mask2": mask2,

            "lip1": lip1,
            "label1": np.array(label1, dtype=np.int64),
            "lip1_len": lip1.shape[0],

            "lip2": lip2,
            "label2": np.array(label2, dtype=np.int64),
            "lip2_len": lip2.shape[0],
        }


class RandomSentencePairDataset(MultiSpeakerDataset):
    def __init__(self, sentence_list, tokenizer, num_pairs_per_epoch=10000):
        super().__init__(sentence_list, tokenizer)
        self.num_pairs_per_epoch = num_pairs_per_epoch

    # 🔽 화자 ID 추출 함수 추가
    def get_speaker_id(self, path):
        filename = os.path.splitext(os.path.basename(path))[0]  # "lip_J_1_M_03_C486_A_012_sentence_41"
        return "_".join(filename.split("_")[:7])                # "lip_J_1_M_03_C486_A"

    def __len__(self):
        return self.num_pairs_per_epoch

    # 🔽 동일 화자 제거된 getitem
    def __getitem__(self, idx):
        for _ in range(10): 
            s1, s2 = random.sample(self.sentence_list, 2)
            id1 = self.get_speaker_id(s1["text_path"])
            id2 = self.get_speaker_id(s2["text_path"])

            if id1 == id2:
                continue  # 동일 화자 조합은 무시

            try:
                return self.load_pair(s1, s2)
            except Exception as e:
                print(f"[Retry] 샘플 로딩 실패: {s1['lip_path']} / {s2['lip_path']} → {e}")
        raise RuntimeError("❌ 최대 재시도 실패 (RandomSentencePairDataset)")

class FixedSentencePairDataset(MultiSpeakerDataset): 
    def __init__(self, pair_list, tokenizer):
        self.pair_list = pair_list
        self.tokenizer = tokenizer

    # 🔽 화자 ID 추출 함수 추가
    def get_speaker_id(self, path):
        filename = os.path.splitext(os.path.basename(path))[0]  # "lip_J_1_M_03_C486_A_012_sentence_41"
        return "_".join(filename.split("_")[:7])                # "lip_J_1_M_03_C486_A"

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        for _ in range(10):
            s1, s2 = self.pair_list[idx]
            id1 = self.get_speaker_id(s1["text_path"])
            id2 = self.get_speaker_id(s2["text_path"])

            if id1 == id2:
                idx = (idx + 1) % len(self.pair_list)
                continue  # 동일 화자 조합은 무시
            try:
                return self.load_pair(s1, s2)
            except Exception as e:
                print(f"[Retry] 샘플 로딩 실패: {s1['lip_path']} / {s2['lip_path']} → {e}")
                idx = (idx + 1) % len(self.pair_list)
        raise RuntimeError("❌ 최대 재시도 실패 (FixedSentencePairDataset)")