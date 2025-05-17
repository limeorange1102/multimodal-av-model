import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, pad_id=0):
    """
    batch: list of dicts from Dataset
    Each dict contains:
        - lip1: [T, H, W, C]
        - label1: [L]
        - lip2: [T, H, W, C]
        - label2: [L]
        - audio: [T]
    """

    # 화자 1: 입술
    lip1_seqs = [torch.tensor(item["lip1"]).permute(0, 3, 1, 2) for item in batch]  # [T, H, W, C] → [T, C, H, W]
    lip1_lengths = [seq.shape[0] for seq in lip1_seqs]
    lip1_padded = pad_sequence(lip1_seqs, batch_first=True)  # [B, T, C, H, W]

    # 화자 1: 텍스트
    text1_seqs = [torch.tensor(item["label1"], dtype=torch.long) for item in batch]
    text1_lengths = [len(seq) for seq in text1_seqs]
    text1_padded = pad_sequence(text1_seqs, batch_first=True, padding_value=pad_id)

    # 화자 2: 입술
    lip2_seqs = [torch.tensor(item["lip2"]).permute(0, 3, 1, 2) for item in batch]
    lip2_lengths = [seq.shape[0] for seq in lip2_seqs]
    lip2_padded = pad_sequence(lip2_seqs, batch_first=True)

    # 화자 2: 텍스트
    text2_seqs = [torch.tensor(item["label2"], dtype=torch.long) for item in batch]
    text2_lengths = [len(seq) for seq in text2_seqs]
    text2_padded = pad_sequence(text2_seqs, batch_first=True, padding_value=pad_id)

    # 오디오 (혼합)
    audio_seqs = [torch.tensor(item["audio"]) for item in batch]
    audio_lengths = [seq.shape[0] for seq in audio_seqs]
    audio_padded = pad_sequence(audio_seqs, batch_first=True)

    attention_mask = torch.zeros_like(audio_padded, dtype=torch.bool)
    for i, length in enumerate(audio_lengths):
        attention_mask[i, :length] = 1

    return {
        # 화자 1
        "lip1": lip1_padded,
        "lip1_lengths": torch.tensor(lip1_lengths),
        "text1": text1_padded,
        "text1_lengths": torch.tensor(text1_lengths),

        # 화자 2
        "lip2": lip2_padded,
        "lip2_lengths": torch.tensor(lip2_lengths),
        "text2": text2_padded,
        "text2_lengths": torch.tensor(text2_lengths),

        # 혼합 오디오
        "audio": audio_padded,
        "audio_attention_mask": attention_mask
    }
