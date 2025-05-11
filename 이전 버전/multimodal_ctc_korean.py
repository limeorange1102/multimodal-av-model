# 📦 multimodal_ctc_korean.py
# 혼합 오디오 + 입모양 → 화자별 자유발화 단어 복원 (한글 시퀀스 예측)

import torch
import torch.nn as nn
import torch.nn.functional as F

class LipEncoder(nn.Module):
    def __init__(self, in_channels=3, hidden_dim=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.rnn = nn.GRU(input_size=64*24*24, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, frames):  # (B, T, C, H, W)
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        x = self.cnn(x)  # (B*T, C, H', W')
        x = x.view(B, T, -1)  # (B, T, F)
        out, _ = self.rnn(x)  # (B, T, 2*hidden_dim)
        return out

class AudioEncoder(nn.Module):
    def __init__(self, input_dim=80, hidden_dim=256):
        super().__init__()
        self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=2, batch_first=True, bidirectional=True)

    def forward(self, mel):  # (B, T, F)
        out, _ = self.rnn(mel)
        return out  # (B, T, 2*hidden_dim)

class MultimodalCTCKoreanModel(nn.Module):
    def __init__(self, vocab_size=200, hidden_dim=256):
        super().__init__()
        self.lip_encoder = LipEncoder(hidden_dim=hidden_dim)
        self.audio_encoder = AudioEncoder(hidden_dim=hidden_dim)
        self.fc = nn.Linear(4 * hidden_dim, vocab_size)  # 2x lip + 2x audio

    def forward(self, frames_A, frames_B, mel):
        lip_feat_A = self.lip_encoder(frames_A)  # (B, T, 2H)
        lip_feat_B = self.lip_encoder(frames_B)
        audio_feat = self.audio_encoder(mel)     # (B, T, 2H)

        fusion_A = torch.cat([lip_feat_A, audio_feat], dim=-1)  # (B, T, 4H)
        fusion_B = torch.cat([lip_feat_B, audio_feat], dim=-1)

        logits_A = self.fc(fusion_A)  # (B, T, vocab_size)
        logits_B = self.fc(fusion_B)
        return logits_A, logits_B  # CTC용 시퀀스 출력

# ⚠️ 이 모델은 torch.nn.CTCLoss와 함께 사용해야 합니다.
# 예: loss_fn = nn.CTCLoss(blank=0) # blank token ID는 vocab 사전에 맞게 설정
