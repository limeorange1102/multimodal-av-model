import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_id=0):
    """
    batch: list of dicts from Dataset
    Each dict contains:
        - lip1: [T, H, W, C]
        - label1: [L]
        - audio: [T]
    """

    # 입술 영상
    lip1_seqs = [torch.tensor(item["lip1"]).permute(0, 3, 1, 2) for item in batch]  # [T, H, W, C] → [T, C, H, W]
    lip1_lengths = [seq.shape[0] for seq in lip1_seqs]
    lip1_padded = pad_sequence(lip1_seqs, batch_first=True)  # [B, T, C, H, W]

    # 텍스트
    text1_seqs = [torch.tensor(item["label1"], dtype=torch.long) for item in batch]
    text1_lengths = [len(seq) for seq in text1_seqs]
    text1_padded = pad_sequence(text1_seqs, batch_first=True, padding_value=pad_id)

    # 오디오
    audio_seqs = [torch.tensor(item["audio"]) for item in batch]
    audio_lengths = [seq.shape[0] for seq in audio_seqs]
    audio_padded = pad_sequence(audio_seqs, batch_first=True)

    attention_mask = torch.zeros_like(audio_padded, dtype=torch.bool)
    for i, length in enumerate(audio_lengths):
        attention_mask[i, :length] = 1

    return {
        "lip1": lip1_padded,
        "lip1_lengths": torch.tensor(lip1_lengths),

        "text1": text1_padded,
        "text1_lengths": torch.tensor(text1_lengths),

        "audio": audio_padded,
        "audio_attention_mask": attention_mask
    }
