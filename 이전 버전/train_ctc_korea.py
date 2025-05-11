# 🏋️ train_ctc_korean.py
# CTC 기반 멀티모달 한글 단어 예측 학습 루프

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.korean_vocab_utils import text_to_indices, VOCAB
from multimodal_ctc_korean import MultimodalCTCKoreanModel
import os
import numpy as np
import torchaudio
import cv2

# ✅ Custom Dataset
class KoreanMultimodalDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = sorted([os.path.join(root_dir, d) for d in os.listdir(root_dir) if d.startswith("sample_")])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]

        # 영상 A, B → (T, H, W, C)
        def load_frames(folder):
            frame_files = sorted(os.listdir(folder))
            frames = [cv2.imread(os.path.join(folder, f)) for f in frame_files]
            frames = [cv2.resize(f, (96, 96)) for f in frames]
            frames = np.stack(frames, axis=0)  # (T, H, W, C)
            frames = frames.transpose(0, 3, 1, 2)  # (T, C, H, W)
            return frames / 255.0

        frames_A = load_frames(os.path.join(sample_path, "frames_A"))
        frames_B = load_frames(os.path.join(sample_path, "frames_B"))

        # 오디오 → mel spectrogram (T, F)
        waveform, sr = torchaudio.load(os.path.join(sample_path, "mixed.wav"))
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80)(waveform)
        mel = mel_spec.squeeze(0).transpose(0, 1)  # (T, 80)

        # 텍스트 라벨
        with open(os.path.join(sample_path, "gt_A.txt"), "r", encoding="utf-8") as f:
            label_A = f.read().strip()
        with open(os.path.join(sample_path, "gt_B.txt"), "r", encoding="utf-8") as f:
            label_B = f.read().strip()

        return {
            "frames_A": torch.FloatTensor(frames_A),
            "frames_B": torch.FloatTensor(frames_B),
            "mel": torch.FloatTensor(mel),
            "label_A": torch.LongTensor(text_to_indices(label_A)),
            "label_B": torch.LongTensor(text_to_indices(label_B)),
        }

# ✅ Collate for padding CTC batch

def collate_fn(batch):
    def pad_sequence(sequences, batch_first=False):
        lengths = torch.LongTensor([len(seq) for seq in sequences])
        padded = nn.utils.rnn.pad_sequence(sequences, batch_first=batch_first)
        return padded, lengths

    frames_A = [item["frames_A"] for item in batch]
    frames_B = [item["frames_B"] for item in batch]
    mel = [item["mel"] for item in batch]
    label_A = [item["label_A"] for item in batch]
    label_B = [item["label_B"] for item in batch]

    # B, T, C, H, W
    frames_A = torch.stack(frames_A)
    frames_B = torch.stack(frames_B)
    mel_padded, mel_lengths = pad_sequence(mel, batch_first=True)

    label_A_padded, len_A = pad_sequence(label_A, batch_first=False)
    label_B_padded, len_B = pad_sequence(label_B, batch_first=False)

    return frames_A, frames_B, mel_padded, mel_lengths, label_A_padded, len_A, label_B_padded, len_B

# ✅ 학습 루프

def train():
    model = MultimodalCTCKoreanModel(vocab_size=len(VOCAB))
    model.train()

    dataset = KoreanMultimodalDataset("./")
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    for epoch in range(10):
        total_loss = 0
        for frames_A, frames_B, mel, mel_lengths, label_A, len_A, label_B, len_B in dataloader:
            logits_A, logits_B = model(frames_A, frames_B, mel)

            log_probs_A = logits_A.log_softmax(2).transpose(0, 1)  # (T, B, C)
            log_probs_B = logits_B.log_softmax(2).transpose(0, 1)

            loss_A = loss_fn(log_probs_A, label_A, mel_lengths, len_A)
            loss_B = loss_fn(log_probs_B, label_B, mel_lengths, len_B)
            loss = loss_A + loss_B

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()
