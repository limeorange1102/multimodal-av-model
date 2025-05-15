import random
import torch
import numpy as np
import librosa

class MultiSpeakerDataset(torch.utils.data.Dataset):
    def __init__(self, sentence_list, tokenizer):
        self.sentence_list = sentence_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentence_list)

    def __getitem__(self, idx):
        s1, s2 = random.sample(self.sentence_list, 2)

        # Load waveforms (1D np.array)
        a1, sr1 = librosa.load(s1["audio_path"], sr=None)
        a2, sr2 = librosa.load(s2["audio_path"], sr=None)

        # 🎯 sr mismatch 확인
        assert sr1 == sr2, f"[오류] Sampling rates do not match! sr1={sr1}, sr2={sr2}" 

        a1 = np.asarray(a1).flatten()
        a2 = np.asarray(a2).flatten()

        # 🔄 두 화자의 오디오 길이 맞추기
        max_len = max(len(a1), len(a2))
        if len(a1) < max_len:
            a1 = np.pad(a1, (0, max_len - len(a1)), mode="constant")
        if len(a2) < max_len:
            a2 = np.pad(a2, (0, max_len - len(a2)), mode="constant")

        # 혼합 오디오 생성
        mix = a1 + a2
        mix = mix.astype(np.float32)
        mix = mix / (np.max(np.abs(mix)) + 1e-6)  # 정규화

        # Load lips
        lip1 = np.load(s1["lip_path"])  # (T1, 27, 2)
        lip2 = np.load(s2["lip_path"])  # (T2, 27, 2)

        # Load labels
        with open(s1["text_path"], "r", encoding="utf-8") as f:
            label1 = self.tokenizer.encode(f.read().strip())
        with open(s2["text_path"], "r", encoding="utf-8") as f:
            label2 = self.tokenizer.encode(f.read().strip())

        return {
            "audio": mix.astype(np.float32),
            "audio1_raw": a1.astype(np.float32),
            "audio2_raw": a2.astype(np.float32),
            "lip1": lip1.astype(np.float32),
            "lip2": lip2.astype(np.float32),
            "label1": np.array(label1, dtype=np.int64),
            "label2": np.array(label2, dtype=np.int64),
            "lip1_len": lip1.shape[0],
            "lip2_len": lip2.shape[0],
        }


class RandomSentencePairDataset(MultiSpeakerDataset):
    def __init__(self, sentence_list, tokenizer, num_pairs_per_epoch=10000):
        super().__init__(sentence_list, tokenizer)
        self.num_pairs_per_epoch = num_pairs_per_epoch

    def __len__(self):
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        return super().__getitem__(idx)


class FixedSentencePairDataset(MultiSpeakerDataset):
    def __init__(self, pair_list, tokenizer):
        super().__init__(pair_list, tokenizer)
        self.pair_list = pair_list  # for clarity; used below

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        s1, s2 = self.pair_list[idx]

        a1, sr1 = librosa.load(s1["audio_path"], sr=None)
        a2, sr2 = librosa.load(s2["audio_path"], sr=None)

        # 🎯 sr mismatch 확인
        assert sr1 == sr2, f"[오류] Sampling rates do not match! sr1={sr1}, sr2={sr2}"
        a1 = np.asarray(a1).flatten()
        a2 = np.asarray(a2).flatten()

        max_len = max(len(a1), len(a2))
        if len(a1) < max_len:
            a1 = np.pad(a1, (0, max_len - len(a1)), mode="constant")
        if len(a2) < max_len:
            a2 = np.pad(a2, (0, max_len - len(a2)), mode="constant")

        mix = a1 + a2
        mix = mix.astype(np.float32)
        mix = mix / (np.max(np.abs(mix)) + 1e-6)

        lip1 = np.load(s1["lip_path"])
        lip2 = np.load(s2["lip_path"])

        with open(s1["text_path"], "r", encoding="utf-8") as f:
            label1 = self.tokenizer.encode(f.read().strip())
        with open(s2["text_path"], "r", encoding="utf-8") as f:
            label2 = self.tokenizer.encode(f.read().strip())

        return {
            "audio": mix.astype(np.float32),
            "audio1_raw": a1.astype(np.float32),
            "audio2_raw": a2.astype(np.float32),
            "lip1": lip1.astype(np.float32),
            "lip2": lip2.astype(np.float32),
            "label1": np.array(label1, dtype=np.int64),
            "label2": np.array(label2, dtype=np.int64),
            "lip1_len": lip1.shape[0],
            "lip2_len": lip2.shape[0],
        }
