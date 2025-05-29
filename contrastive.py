import torch
import torch.nn.functional as F

TEMPERATURE = 0.07
WEIGHT_STRONG_POS = 1.0
WEIGHT_WEAK_POS = 0.3

def contrastive_loss_with_mask(flat_feat, flat_mask):
    device = flat_feat.device

    # mask == 3 제거
    valid_mask = flat_mask != 3
    flat_feat = flat_feat[valid_mask]
    flat_mask = flat_mask[valid_mask]

    flat_feat = F.normalize(flat_feat, dim=1)

    pos_strong_idx = torch.nonzero(flat_mask == 2).squeeze(1)
    pos_weak_idx = torch.nonzero(flat_mask == 1).squeeze(1)
    neg_idx = torch.nonzero(flat_mask == 0).squeeze(1)

    if len(pos_strong_idx) == 0 or len(neg_idx) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    pos_strong_feat = flat_feat[pos_strong_idx]
    pos_weak_feat = flat_feat[pos_weak_idx] if len(pos_weak_idx) > 0 else None
    neg_feat = flat_feat[neg_idx]

    sim_strong = torch.matmul(pos_strong_feat, neg_feat.T) / TEMPERATURE
    loss_strong = -F.log_softmax(sim_strong, dim=1).mean()

    if pos_weak_feat is not None:
        sim_weak = torch.matmul(pos_weak_feat, neg_feat.T) / TEMPERATURE
        loss_weak = -F.log_softmax(sim_weak, dim=1).mean()
    else:
        loss_weak = torch.tensor(0.0, device=device)

    total_loss = WEIGHT_STRONG_POS * loss_strong + WEIGHT_WEAK_POS * loss_weak
    return total_loss

