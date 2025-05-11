import torch
import torch.nn as nn
import torch.nn.functional as F

class FusionModule(nn.Module):
    def __init__(self, visual_dim, audio_dim, fused_dim=512):
        super(FusionModule, self).__init__()

        # 입력 feature를 같은 차원으로 projection
        self.visual_proj = nn.Linear(visual_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)

        # Gating score 계산 (Late Fusion-like)
        self.visual_gate = nn.Linear(visual_dim, 1)
        self.audio_gate = nn.Linear(audio_dim, 1)

        # 후처리 layer
        self.post_fusion = nn.Sequential(
            nn.Linear(fused_dim, fused_dim),
            nn.ReLU(),
            nn.LayerNorm(fused_dim)
        )

    def forward(self, visual_feat, audio_feat):
        """
        visual_feat: [B, T, D_v]
        audio_feat:  [B, T, D_a]
        Returns:
            fused: [B, T, fused_dim]
        """
        # 차원 맞추기
        visual_proj = self.visual_proj(visual_feat)  # [B, T, fused_dim]
        audio_proj = self.audio_proj(audio_feat)     # [B, T, fused_dim]

        # gating score 계산
        visual_score = torch.sigmoid(self.visual_gate(visual_feat))  # [B, T, 1]
        audio_score  = torch.sigmoid(self.audio_gate(audio_feat))    # [B, T, 1]

        # normalize (두 score 합이 1이 되도록)
        sum_score = visual_score + audio_score + 1e-6
        visual_weight = visual_score / sum_score
        audio_weight  = audio_score / sum_score

        # weighted sum
        fused = visual_weight * visual_proj + audio_weight * audio_proj  # [B, T, fused_dim]

        return self.post_fusion(fused)  # [B, T, fused_dim]
