import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionFusion(nn.Module):
    def __init__(self, visual_dim, audio_dim, fused_dim, num_heads=4):
        super().__init__()

        # Step 1: 두 modality를 동일한 차원으로 projection
        self.visual_proj = nn.Linear(visual_dim, fused_dim)
        self.audio_proj = nn.Linear(audio_dim, fused_dim)

        # Step 2: 서로 교차로 attention
        self.cross_attn_visual = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)
        self.cross_attn_audio = nn.MultiheadAttention(embed_dim=fused_dim, num_heads=num_heads, batch_first=True)

        # Step 3: concat된 결과를 하나의 fused vector로 projection
        self.fusion_proj = nn.Linear(fused_dim * 2, fused_dim)

    def forward(self, visual_feat, audio_feat):
        """
        visual_feat: [B, T_v, D_v]
        audio_feat:  [B, T_a, D_a]
        Returns: fused_feat [B, T, fused_dim]
        """
        # Step 0: 시간축 보정 (보통 visual 기준으로 맞춤)
        T_v = visual_feat.size(1)
        T_a = audio_feat.size(1)

        if T_v != T_a:
            audio_feat = F.interpolate(audio_feat.permute(0, 2, 1), size=T_v, mode='linear', align_corners=True)
            audio_feat = audio_feat.permute(0, 2, 1)

        # Step 1: 각 modality projection
        v = self.visual_proj(visual_feat)  # [B, T, D]
        a = self.audio_proj(audio_feat)    # [B, T, D]

        # Step 2: 서로에게 cross attention
        v2a, _ = self.cross_attn_visual(query=v, key=a, value=a)  # visual이 audio attend
        a2v, _ = self.cross_attn_audio(query=a, key=v, value=v)   # audio가 visual attend

        # Step 3: concat 후 projection
        fused = torch.cat([v2a, a2v], dim=-1)      # [B, T, 2D]
        fused = self.fusion_proj(fused)            # [B, T, D]

        return fused
