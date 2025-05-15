import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_id=0):
    """
    batch: list of dicts from Dataset
    Each dict contains:
        - lip1: [T1, C, H, W]
        - lip2: [T2, C, H, W]
        - text1: [L1]
        - text2: [L2]
        - audio: [T] (waveform)
    """

    # 입술 영상
    lip1_seqs = [torch.tensor(item["lip1"]).permute(0, 3, 1, 2) for item in batch]  # [T, H, W, C] → [T, C, H, W]
    lip2_seqs = [torch.tensor(item["lip2"]).permute(0, 3, 1, 2) for item in batch]
    lip1_lengths = [seq.shape[0] for seq in lip1_seqs]
    lip2_lengths = [seq.shape[0] for seq in lip2_seqs]
    lip1_padded = pad_sequence(lip1_seqs, batch_first=True)
    lip2_padded = pad_sequence(lip2_seqs, batch_first=True)

    # 텍스트 (CTC용)
    text1_seqs = [torch.tensor(item["label1"], dtype=torch.long) for item in batch]
    text2_seqs = [torch.tensor(item["label2"], dtype=torch.long) for item in batch]

    text1_lengths = [len(seq) for seq in text1_seqs]
    text2_lengths = [len(seq) for seq in text2_seqs]
    text1_padded = pad_sequence(text1_seqs, batch_first=True, padding_value=pad_id)
    text2_padded = pad_sequence(text2_seqs, batch_first=True, padding_value=pad_id)

    # 오디오 (waveform 기준)
    audio_seqs = [torch.tensor(item["audio"]) for item in batch]  # list of [T]
    audio_lengths = [seq.shape[0] for seq in audio_seqs]
    audio_padded = pad_sequence(audio_seqs, batch_first=True)  # [B, T]

    attention_mask = torch.zeros_like(audio_padded, dtype=torch.bool)
    for i, length in enumerate(audio_lengths):
        attention_mask[i, :length] = 1

    return {
        "lip1": lip1_padded,                          # [B, T1_max, C, H, W]
        "lip2": lip2_padded,
        "lip1_lengths": torch.tensor(lip1_lengths),
        "lip2_lengths": torch.tensor(lip2_lengths),

        "text1": text1_padded,                        # [B, L1_max]
        "text2": text2_padded,
        "text1_lengths": torch.tensor(text1_lengths),
        "text2_lengths": torch.tensor(text2_lengths),

        "audio": audio_padded,                        # [B, T]
        "audio_attention_mask": attention_mask        # [B, T]
    }
