import os
import random
import torch
import numpy as np
import librosa
from torch.utils.data import Dataset
import torchaudio.transforms as T


class BaseSentencePairDataset(Dataset):
    def __init__(self, sr=16000):
        self.sr = sr
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sr,
            n_fft=400,
            hop_length=160,
            n_mels=80
        )

    def waveform_to_log_mel(self, waveform):
        mel_spec = self.mel_transform(waveform)  # [1, 80, T]
        log_mel_spec = torch.log(mel_spec + 1e-6)
        return log_mel_spec.squeeze(0).transpose(0, 1)  # [T, 80]

    def _process_pair(self, s1, s2, tokenizer):
        # 입술 영상 로딩
        lip1 = torch.from_numpy(np.load(s1["lip_path"])).permute(0, 3, 1, 2).float() / 255.0
        lip2 = torch.from_numpy(np.load(s2["lip_path"])).permute(0, 3, 1, 2).float() / 255.0

        # 텍스트 로딩 및 토크나이징
        with open(s1["text_path"], 'r', encoding='utf-8') as f:
            label1 = tokenizer.encode(f.read().strip()) 
        with open(s2["text_path"], 'r', encoding='utf-8') as f:
            label2 = tokenizer.encode(f.read().strip())
        label1 = torch.tensor(label1, dtype=torch.long)
        label2 = torch.tensor(label2, dtype=torch.long)

        # 오디오 로딩 및 혼합
        dur1 = s1["end_time"] - s1["start_time"]
        dur2 = s2["end_time"] - s2["start_time"]
        a1, _ = librosa.load(s1["audio_path"], sr=self.sr, offset=s1["start_time"], duration=dur1)
        a2, _ = librosa.load(s2["audio_path"], sr=self.sr, offset=s2["start_time"], duration=dur2)

        min_len = min(len(a1), len(a2))
        mix = a1[:min_len] + a2[:min_len]
        mix = mix / (np.max(np.abs(mix)) + 1e-6)

        mix_tensor = torch.from_numpy(mix).float().unsqueeze(0)  # [1, T]
        log_mel = self.waveform_to_log_mel(mix_tensor)           # [T, 80]

        return {
            "lip1": lip1,
            "lip2": lip2,
            "text1": label1,
            "text2": label2,
            "audio": log_mel
        }


class RandomSentencePairDataset(BaseSentencePairDataset):
    def __init__(self, sentence_list, tokenizer, sr=16000, num_pairs_per_epoch=10000):
        super().__init__(sr)
        self.sentence_list = sentence_list
        self.tokenizer = tokenizer
        self.num_pairs_per_epoch = num_pairs_per_epoch

    def __len__(self):
        return self.num_pairs_per_epoch

    def __getitem__(self, idx):
        s1, s2 = random.sample(self.sentence_list, 2)
        return self._process_pair(s1, s2, self.tokenizer)


class FixedSentencePairDataset(BaseSentencePairDataset):
    def __init__(self, pair_list, tokenizer, sr=16000):
        super().__init__(sr)
        self.pair_list = pair_list
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.pair_list)

    def __getitem__(self, idx):
        s1, s2 = self.pair_list[idx]
        return self._process_pair(s1, s2, self.tokenizer)
