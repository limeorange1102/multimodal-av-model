import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPERATURE = 0.07
WEIGHT_STRONG_POS = 1.0
WEIGHT_WEAK_POS = 0.3


def contrastive_loss_with_mask(audio_feat, mask):
    """
    audio_feat: [B, T, D] - audio encoder output
    mask: [B, T] - values in {0, 1, 2, 3}
    """
    B, T, D = audio_feat.shape
    device = audio_feat.device

    flat_feat = audio_feat.reshape(B * T, D)     # [B*T, D]
    flat_mask = mask.reshape(B * T)              # [B*T]

    # mask == 3 구간 제거
    valid_mask = flat_mask != 3
    flat_feat = flat_feat[valid_mask]            # [N, D]
    flat_feat = F.normalize(flat_feat, dim=1)    # cosine similarity 기반 정규화
    flat_mask = flat_mask[valid_mask]

    # 각 인덱스 추출
    pos_strong_idx = torch.nonzero(flat_mask == 2).squeeze(1)  # 단독발화
    pos_weak_idx   = torch.nonzero(flat_mask == 1).squeeze(1)  # 동시발화
    neg_idx        = torch.nonzero(flat_mask == 0).squeeze(1)  # 타화자 단독발화

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    # 단독발화 vs 타화자 단독발화 → strong positive
    if len(pos_strong_idx) > 0 and len(neg_idx) > 0:
        pos_strong_feat = flat_feat[pos_strong_idx]   # [P1, D]
        neg_feat = flat_feat[neg_idx]                 # [N, D]
        sim_strong = torch.matmul(pos_strong_feat, neg_feat.T) / TEMPERATURE  # [P1, N]
        loss_strong = -F.log_softmax(sim_strong, dim=1).mean()
        total_loss = total_loss + WEIGHT_STRONG_POS * loss_strong

    # 동시발화 vs 타화자 단독발화 → weak positive (실제로는 negative-only)
    if len(pos_weak_idx) > 0 and len(neg_idx) > 0:
        pos_weak_feat = flat_feat[pos_weak_idx]       # [P2, D]
        neg_feat = flat_feat[neg_idx]                 # [N, D]
        sim_weak = torch.matmul(pos_weak_feat, neg_feat.T) / TEMPERATURE      # [P2, N]
        loss_weak = -F.log_softmax(sim_weak, dim=1).mean()
        total_loss = total_loss + WEIGHT_WEAK_POS * loss_weak

    return total_loss
