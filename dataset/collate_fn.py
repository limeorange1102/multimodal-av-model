import torch
from torch.nn.utils.rnn import pad_sequence

def pad_video_sequence(seq_list):
    """
    Zero-pad a list of 5D lip video tensors: [T, C, H, W]
    Returns: [B, T_max, C, H, W]
    """
    max_len = max([x.size(0) for x in seq_list])
    padded = []
    for seq in seq_list:
        pad_len = max_len - seq.size(0)
        if pad_len > 0:
            pad = torch.zeros((pad_len, *seq.shape[1:]), dtype=seq.dtype)
            padded_seq = torch.cat([seq, pad], dim=0)
        else:
            padded_seq = seq
        padded.append(padded_seq)
    return torch.stack(padded)

def collate_fn(batch, pad_id=0):
    """
    batch: list of dicts from Dataset
    Each dict contains:
        - lip1: [T1, C, H, W]
        - lip2: [T2, C, H, W]
        - text1: [L1]
        - text2: [L2]
        - audio: Tensor (log-Mel: [T, 80])
    """
    # Lip sequences
    lip1_seqs = [item["lip1"] for item in batch]
    lip2_seqs = [item["lip2"] for item in batch]
    lip1_lengths = [seq.size(0) for seq in lip1_seqs]
    lip2_lengths = [seq.size(0) for seq in lip2_seqs]

    lip1_padded = pad_video_sequence(lip1_seqs)  # [B, T1_max, C, H, W]
    lip2_padded = pad_video_sequence(lip2_seqs)

    # Text sequences
    text1_seqs = [item["text1"] for item in batch]
    text2_seqs = [item["text2"] for item in batch]
    text1_lengths = [len(seq) for seq in text1_seqs]
    text2_lengths = [len(seq) for seq in text2_seqs]

    text1_padded = pad_sequence(text1_seqs, batch_first=True, padding_value=pad_id)
    text2_padded = pad_sequence(text2_seqs, batch_first=True, padding_value=pad_id)

    # Audio sequences (log-Mel)
    audio_seqs = [item["audio"] for item in batch]  # [T, 80]
    audio_lengths = [seq.size(0) for seq in audio_seqs]
    audio_padded = pad_sequence(audio_seqs, batch_first=True)  # [B, T_max, 80]

    return {
        "lip1": lip1_padded,
        "lip2": lip2_padded,
        "lip1_lengths": torch.tensor(lip1_lengths),
        "lip2_lengths": torch.tensor(lip2_lengths),

        "text1": text1_padded,
        "text2": text2_padded,
        "text1_lengths": torch.tensor(text1_lengths),
        "text2_lengths": torch.tensor(text2_lengths),

        "audio": audio_padded,
        "audio_lengths": torch.tensor(audio_lengths),
    }
