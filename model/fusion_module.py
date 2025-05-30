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

        # ✅ Step 4: Temporal modeling (BiLSTM + projection)
        self.temporal_model = nn.LSTM(
            input_size=fused_dim,
            hidden_size=fused_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, visual_feat, audio_feat, mask=None):
        """
        visual_feat: [B, T_v, D_v]
        audio_feat:  [B, T_a, D_a]
        mask:        [B, T_a] — audio에 대응되는 mask (0/3: 무시, 1/2: 사용)
        Returns: fused_feat [B, T, fused_dim]
        """
        B, T_v, _ = visual_feat.shape
        _, T_a, _ = audio_feat.shape

        if T_v != T_a:
            audio_feat = F.interpolate(audio_feat.permute(0, 2, 1), size=T_v, mode='linear', align_corners=True)
            audio_feat = audio_feat.permute(0, 2, 1)
            if mask is not None:
                # mask도 동일하게 interpolate 해줌 (float로 변환 후 round)
                mask = F.interpolate(mask.unsqueeze(1).float(), size=T_v, mode='nearest').squeeze(1).long()

        v = self.visual_proj(visual_feat)  # [B, T, D]
        a = self.audio_proj(audio_feat)    # [B, T, D]

        # mask == 0, 3인 구간 무시 (즉, 화자가 말하지 않는 구간 무시)
        key_padding_mask = (mask == 0) | (mask == 3) if mask is not None else None # [B, T]

        # visual attends to audio
        v2a, _ = self.cross_attn_visual(query=v, key=a, value=a, key_padding_mask=key_padding_mask)

        fused = self.fusion_proj(v2a)           # [B, T, D]
        fused_seq, _ = self.temporal_model(fused)  # [B, T, 2*D]
        return fused_seq

