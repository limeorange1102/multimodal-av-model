import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCDecoder(nn.Module):
    def __init__(self, input_dim=512, vocab_size=5000, dropout=0.1):
        super(CTCDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_dim, vocab_size)  # vocab 차원으로 projection
        self.ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)

    def forward(self, fused_feat, target=None, input_lengths=None, target_lengths=None):
        """
        fused_feat: [B, T, D] - fusion된 feature
        target: [B, S] - 정답 문장 토큰 시퀀스 (훈련 시에만)
        input_lengths: [B] - 입력 시퀀스 길이
        target_lengths: [B] - 정답 시퀀스 길이

        Returns:
            훈련 시: CTC loss
            추론 시: log_probs [B, T, V]
        """
        x = self.dropout(fused_feat)
        logits = self.fc(x)  # [B, T, V]
        log_probs = F.log_softmax(logits, dim=-1)

        if target is not None:
            return self.ctc_loss(
                log_probs.transpose(0, 1),  # [T, B, V]
                target,                    # [B, S]
                input_lengths,            # [B]
                target_lengths            # [B]
            )
        else:
            return log_probs  # [B, T, V]