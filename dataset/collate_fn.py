import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    """
    batch: list of dicts from Dataset
    Each dict contains:
        - lip1, lip2: [T, C, H, W]
        - label1, label2: [L]
        - audio: [T]
        - mask1, mask2: [T]
    """
    # 화자 1: 입술
    lip1_seqs = [item["lip1"] for item in batch]  # [T, C, H, W]
    lip1_lengths = [seq.shape[0] for seq in lip1_seqs]
    lip1_padded_seqs = pad_sequence(lip1_seqs, batch_first=True)  # [B, T, C, H, W], T padding

    # 화자 2: 입술
    lip2_seqs = [item["lip2"] for item in batch]
    lip2_lengths = [seq.shape[0] for seq in lip2_seqs]
    lip2_padded_seqs = pad_sequence(lip2_seqs, batch_first=True)

    # 화자 1: 텍스트
    text1_seqs = [torch.tensor(item["label1"], dtype=torch.long) for item in batch]
    text1_lengths = [len(seq) for seq in text1_seqs]
    text1_padded_seqs = pad_sequence(text1_seqs, batch_first=True)

    # 화자 2: 텍스트
    text2_seqs = [torch.tensor(item["label2"], dtype=torch.long) for item in batch]
    text2_lengths = [len(seq) for seq in text2_seqs]
    text2_padded_seqs = pad_sequence(text2_seqs, batch_first=True)

    # audio (혼합음성)
    audio_seqs = [torch.tensor(item["audio"], dtype=torch.float32) for item in batch]
    audio_lengths = [len(x) for x in audio_seqs]
    audio_padded_seqs = pad_sequence(audio_seqs, batch_first=True)

    # mask1
    mask1_seqs = [torch.tensor(item["mask1"], dtype=torch.long) for item in batch]
    mask1_padded_seqs = pad_sequence(mask1_seqs, batch_first=True, padding_value=3) # 3: padding value

    # mask2
    mask2_seqs = [torch.tensor(item["mask2"], dtype=torch.long) for item in batch]
    mask2_padded_seqs = pad_sequence(mask2_seqs, batch_first=True, padding_value=3)

    return {
        # 화자 1
        "lip1": lip1_padded_seqs, #list of [B, T, C, H, W]
        "lip1_lengths": torch.tensor(lip1_lengths),
        "text1": text1_padded_seqs,
        "text1_lengths": torch.tensor(text1_lengths),

        # 화자 2
        "lip2": lip2_padded_seqs,
        "lip2_lengths": torch.tensor(lip2_lengths),
        "text2": text2_padded_seqs,
        "text2_lengths": torch.tensor(text2_lengths),

        "audio": audio_padded_seqs,
        "audio_lengths": torch.tensor(audio_lengths),
        "mask1": mask1_padded_seqs,
        "mask2": mask2_padded_seqs,
    }
