import torch
import torch.nn.functional as F

TEMPERATURE = 0.07
WEIGHT_STRONG_POS = 1.0
WEIGHT_WEAK_POS = 0.3

def contrastive_loss_with_mask(audio_feat, mask):
    """
    audio_feat: [B, T, D]
    mask: [B, T] with values in {0, 1, 2, 3}
    """
    B, T, D = audio_feat.shape
    device = audio_feat.device

    flat_feat = audio_feat.reshape(-1, D)         # [B*T, D]
    flat_mask = mask.reshape(-1)                  # [B*T]

    valid_mask = flat_mask != 3                   # 제거할 영역 제외
    flat_feat = flat_feat[valid_mask]             # [N, D]
    flat_mask = flat_mask[valid_mask]             # [N]

    if flat_feat.size(0) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    flat_feat = F.normalize(flat_feat, dim=1)

    pos_strong_idx = (flat_mask == 2).nonzero(as_tuple=False).squeeze(1)
    pos_weak_idx = (flat_mask == 1).nonzero(as_tuple=False).squeeze(1)
    neg_idx = (flat_mask == 0).nonzero(as_tuple=False).squeeze(1)

    if len(pos_strong_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    pos_strong_feat = flat_feat[pos_strong_idx]
    neg_feat = flat_feat[neg_idx]

    sim_strong = torch.matmul(pos_strong_feat, neg_feat.T) / TEMPERATURE
    loss_strong = -F.log_softmax(sim_strong, dim=1).mean()

    if len(pos_weak_idx) > 0:
        pos_weak_feat = flat_feat[pos_weak_idx]
        sim_weak = torch.matmul(pos_weak_feat, neg_feat.T) / TEMPERATURE
        loss_weak = -F.log_softmax(sim_weak, dim=1).mean()
    else:
        loss_weak = torch.tensor(0.0, device=device)

    total_loss = WEIGHT_STRONG_POS * loss_strong + WEIGHT_WEAK_POS * loss_weak
    return total_loss
