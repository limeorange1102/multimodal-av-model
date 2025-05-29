import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPERATURE = 0.07
WEIGHT_STRONG_POS = 1.0
WEIGHT_WEAK_POS = 0.3


def contrastive_loss_with_mask(audio_feat, mask_list):
    """
    audio_feat: list of B tensors, each of shape [T_i, D] (after encoder)
    mask_list:  list of B tensors, each of shape [T_i] with values in {0,1,2}
                0 = 타화자 음성, 1 = 동시발화, 2 = 단독발화
    """

    # 🔧 Normalize each sample individually
    normalized_feat_list = [F.normalize(f, dim=1) for f in audio_feat]  # List of [T_i, D]
    
    # 🔧 Flatten all features and masks
    flat_feat = torch.cat(normalized_feat_list, dim=0)  # [sum(T_i), D]
    flat_mask = torch.cat(mask_list, dim=0)             # [sum(T_i)]

    device = flat_feat.device

    # 🔍 각 구간의 인덱스
    pos_strong_idx = torch.nonzero(flat_mask == 2).squeeze(1)  # 단독발화
    pos_weak_idx = torch.nonzero(flat_mask == 1).squeeze(1)    # 동시발화
    neg_idx       = torch.nonzero(flat_mask == 0).squeeze(1)   # 타화자 음성

    # 🔒 최소 조건: 단독 발화와 음성 분리 구간이 있어야 학습 가능
    if len(pos_strong_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    # 🎯 feature 추출
    pos_strong_feat = flat_feat[pos_strong_idx]                   # [P1, D]
    neg_feat        = flat_feat[neg_idx]                          # [N, D]
    pos_weak_feat   = flat_feat[pos_weak_idx] if len(pos_weak_idx) > 0 else None  # [P2, D]

    # ✅ strong positive loss
    sim_strong = torch.matmul(pos_strong_feat, neg_feat.T) / TEMPERATURE   # [P1, N]
    loss_strong = -F.log_softmax(sim_strong, dim=1).mean()

    # ✅ weak positive loss (if available)
    if pos_weak_feat is not None:
        sim_weak = torch.matmul(pos_weak_feat, neg_feat.T) / TEMPERATURE   # [P2, N]
        loss_weak = -F.log_softmax(sim_weak, dim=1).mean()
    else:
        loss_weak = torch.tensor(0.0, device=device)

    # 🎯 최종 contrastive loss
    total_loss = WEIGHT_STRONG_POS * loss_strong + WEIGHT_WEAK_POS * loss_weak
    return total_loss