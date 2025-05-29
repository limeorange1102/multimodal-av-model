import torch
def simple_beam_search(log_probs: torch.Tensor, beam_width=5, blank=0):
    """
    log_probs: (T, V) log-probabilities (GPU tensor OK)
    beam_width: number of beams to keep
    blank: blank token ID
    """
    T, V = log_probs.shape

    # 초기 beam: (sequence, log_score)
    beams = [([], 0.0)]

    for t in range(T):
        next_beams = {}
        topk_log_probs, topk_ids = torch.topk(log_probs[t], beam_width)  # (beam_width,)

        for seq, score in beams:
            for k in range(beam_width):
                c = topk_ids[k].item()
                log_p = topk_log_probs[k].item()
                new_seq = seq + [c]
                new_score = score + log_p
                key = tuple(new_seq)
                if key not in next_beams or new_score > next_beams[key]:
                    next_beams[key] = new_score

        # beam_width 상위만 유지
        beams = sorted(next_beams.items(), key=lambda x: x[1], reverse=True)[:beam_width]
        beams = [(list(seq), score) for seq, score in beams]

    # 최종 best sequence 선택
    best_seq = beams[0][0]

    # CTC 규칙 적용 (중복 제거 + blank 제거)
    final = []
    prev = None
    for idx in best_seq:
        if idx != prev and idx != blank:
            final.append(idx)
        prev = idx

    return final

# ✅ 빠른 decode 함수 정의 (SimpleTokenizer에 맞게)
def fast_decode(ids, tokenizer):
    return ''.join([
        tokenizer.id_to_token[i] for i in ids
        if i != tokenizer.blank_id and 0 <= i < tokenizer.vocab_size
    ]).replace('▁', ' ').strip()