import random
import torch
import numpy as np
import librosa
import cv2

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
        a1 = a1[int(s1["start_time"] * sr1):int(s1["end_time"] * sr1)]

        a2, sr2 = librosa.load(s2["audio_path"], sr=None)
        a2 = a2[int(s2["start_time"] * sr2):int(s2["end_time"] * sr2)]

        assert sr1 == sr2, f"[오류] Sampling rates do not match! sr1={sr1}, sr2={sr2}"

        a1 = np.asarray(a1).flatten()
        a2 = np.asarray(a2).flatten()

        # Pad to match length
        len1 = len(a1)
        len2 = len(a2)

        if len1 <= len2: #화자 1이 더 짧은 경우
            a2_trimmed = a2[:len1]
            mix1 = a1 + a2_trimmed

            a1_padded = np.pad(a1, (0, len2 - len1), mode="constant")
            mix2 = a1_padded + a2
        else: #화자 2가 더 짧은 경우
            a2_padded = np.pad(a2, (0, len1 - len2), mode="constant")
            mix1 = a1 + a2_padded

            a1_trimmed = a1[:len2]
            mix2 = a1_trimmed + a2

        mix1 = mix1.astype(np.float32)
        mix1 = mix1 / (np.max(np.abs(mix1)) + 1e-6)

        mix2 = mix2.astype(np.float32)
        mix2 = mix2 / (np.max(np.abs(mix2)) + 1e-6)

        # Load lips
        try:
            lip1_raw = np.load(s1["lip_path"])
            lip1 = np.stack([frame for frame in lip1_raw])
        except Exception as e:
            print(f"❌ lip1 로딩 실패: {s1['lip_path']} - {e}")
            return self.__getitem__(random.randint(0, len(self.sentence_list) - 1))

        try:
            lip2_raw = np.load(s2["lip_path"])
            lip2 = np.stack([frame for frame in lip2_raw])
        except Exception as e:
            print(f"❌ lip2 로딩 실패: {s2['lip_path']} - {e}")
            return self.__getitem__(random.randint(0, len(self.sentence_list) - 1))

        # Load labels
        with open(s1["text_path"], "r", encoding="utf-8") as f:
            label1 = self.tokenizer.encode(f.read().strip())
        with open(s2["text_path"], "r", encoding="utf-8") as f:
            label2 = self.tokenizer.encode(f.read().strip())

        return {
            "audio1": mix1.astype(np.float32),
            "audio2": mix2.astype(np.float32),

            "lip1": lip1.astype(np.float32),
            "label1": np.array(label1, dtype=np.int64),
            "lip1_len": lip1.shape[0],

            "lip2": lip2.astype(np.float32),
            "label2": np.array(label2, dtype=np.int64),
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
        self.pair_list = pair_list

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        s1, s2 = self.pair_list[idx]

        a1, sr1 = librosa.load(s1["audio_path"], sr=None)
        a1 = a1[int(s1["start_time"] * sr1):int(s1["end_time"] * sr1)]

        a2, sr2 = librosa.load(s2["audio_path"], sr=None)
        a2 = a2[int(s2["start_time"] * sr2):int(s2["end_time"] * sr2)]

        assert sr1 == sr2, f"[오류] Sampling rates do not match! sr1={sr1}, sr2={sr2}"

        a1 = np.asarray(a1).flatten()
        a2 = np.asarray(a2).flatten()

        # Pad to match length
        len1 = len(a1)
        len2 = len(a2)

        if len1 <= len2: #화자 1이 더 짧은 경우
            a2_trimmed = a2[:len1]
            mix1 = a1 + a2_trimmed

            a1_padded = np.pad(a1, (0, len2 - len1), mode="constant")
            mix2 = a1_padded + a2
        else: #화자 2가 더 짧은 경우
            a2_padded = np.pad(a2, (0, len1 - len2), mode="constant")
            mix1 = a1 + a2_padded
            
            a1_trimmed = a1[:len2]
            mix2 = a1_trimmed + a2

        mix1 = mix1.astype(np.float32)
        mix1 = mix1 / (np.max(np.abs(mix1)) + 1e-6)
        mix2 = mix2.astype(np.float32)
        mix2 = mix2 / (np.max(np.abs(mix2)) + 1e-6)

        lip1 = np.load(s1["lip_path"])
        lip1 = np.stack([frame for frame in lip1])

        lip2 = np.load(s2["lip_path"])
        lip2 = np.stack([frame for frame in lip2])

        with open(s1["text_path"], "r", encoding="utf-8") as f:
            label1 = self.tokenizer.encode(f.read().strip())
        with open(s2["text_path"], "r", encoding="utf-8") as f:
            label2 = self.tokenizer.encode(f.read().strip())

        return {
            "audio1": mix1.astype(np.float32),
            "audio2": mix2.astype(np.float32),

            "lip1": lip1.astype(np.float32),
            "label1": np.array(label1, dtype=np.int64),
            "lip1_len": lip1.shape[0],

            "lip2": lip2.astype(np.float32),
            "label2": np.array(label2, dtype=np.int64),
            "lip2_len": lip2.shape[0],
        }
