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

    # 화자 1: 오디오
    audio1_seqs = [torch.tensor(item["audio1"]) for item in batch]
    audio1_lengths = [len(x) for x in audio1_seqs]
    audio1_mixed = pad_sequence(audio1_seqs, batch_first=True)
    audio1_mask = torch.zeros_like(audio1_mixed, dtype=torch.bool)
    for i, l in enumerate(audio1_lengths):
        audio1_mask[i, :l] = 1

    # 화자 1: 텍스트
    text1_seqs = [torch.tensor(item["label1"], dtype=torch.long) for item in batch]
    text1_lengths = [len(seq) for seq in text1_seqs]
    text1_padded = pad_sequence(text1_seqs, batch_first=True, padding_value=pad_id)

    # 화자 2: 입술
    lip2_seqs = [torch.tensor(item["lip2"]).permute(0, 3, 1, 2) for item in batch]
    lip2_lengths = [seq.shape[0] for seq in lip2_seqs]
    lip2_padded = pad_sequence(lip2_seqs, batch_first=True)

    # 화자 2: 오디오
    audio2_seqs = [torch.tensor(item["audio2"]) for item in batch]
    audio2_lengths = [len(x) for x in audio2_seqs]
    audio2_mixed = pad_sequence(audio2_seqs, batch_first=True)
    audio2_mask = torch.zeros_like(audio2_mixed, dtype=torch.bool)
    for i, l in enumerate(audio2_lengths):
        audio2_mask[i, :l] = 1

    # 화자 2: 텍스트
    text2_seqs = [torch.tensor(item["label2"], dtype=torch.long) for item in batch]
    text2_lengths = [len(seq) for seq in text2_seqs]
    text2_padded = pad_sequence(text2_seqs, batch_first=True, padding_value=pad_id)

    return {
        # 화자 1
        "lip1": lip1_padded,
        "lip1_lengths": torch.tensor(lip1_lengths),
        "text1": text1_padded,
        "text1_lengths": torch.tensor(text1_lengths),
        "audio1": audio1_mixed,
        "audio1_mask": audio1_mask,

        # 화자 2
        "lip2": lip2_padded,
        "lip2_lengths": torch.tensor(lip2_lengths),
        "text2": text2_padded,
        "text2_lengths": torch.tensor(text2_lengths),
        "audio2": audio2_mixed,
        "audio2_mask": audio2_mask
    }
