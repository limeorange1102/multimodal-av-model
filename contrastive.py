import torch
import torch.nn.functional as F

TEMPERATURE = 0.07
WEIGHT_POS_ALIGN = 1.0
WEIGHT_NEG_SUPPRESS = 0.3

def contrastive_loss_with_mask(middle_feat, flat_mask, projection_layer=None):
    device = middle_feat.device
    B, T_enc, D = middle_feat.shape
    flat_feat = middle_feat.reshape(B * T_enc, D)  # [B*T_enc, D]

    # mask == 3 제거
    valid_mask = flat_mask != 3
    flat_feat = flat_feat[valid_mask]
    flat_mask = flat_mask[valid_mask]

    if projection_layer is not None:
        flat_feat = projection_layer(flat_feat)

    flat_feat = F.normalize(flat_feat, dim=1)


    pos_strong_idx = torch.nonzero(flat_mask == 2).squeeze(1)
    pos_weak_idx = torch.nonzero(flat_mask == 1).squeeze(1)
    neg_idx = torch.nonzero(flat_mask == 0).squeeze(1)

    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    if len(pos_weak_idx) > 0 and len(pos_strong_idx) > 0:
        anchor_feat = flat_feat[pos_weak_idx]
        positive_feat = flat_feat[pos_strong_idx]
        sim = torch.matmul(anchor_feat, positive_feat.T) / TEMPERATURE
        pos_loss = -F.log_softmax(sim, dim=1).mean()
        total_loss = total_loss + WEIGHT_POS_ALIGN * pos_loss

    # Case 2: 동시발화(1) vs 타화자 단독발화(0)
    if len(pos_weak_idx) > 0 and len(neg_idx) > 0:
        anchor_feat = flat_feat[pos_weak_idx]
        neg_feat = flat_feat[neg_idx]
        sim_weak = torch.matmul(anchor_feat, neg_feat.T) / TEMPERATURE
        loss_weak = -F.log_softmax(sim_weak, dim=1).mean()
        total_loss = total_loss + WEIGHT_NEG_SUPPRESS * loss_weak
    return total_loss

