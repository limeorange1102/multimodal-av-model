import torch
from torch.utils.data import DataLoader
import os, random, numpy as np
from sklearn.model_selection import train_test_split

from dataset.multi_speaker_dataset import RandomSentencePairDataset, FixedSentencePairDataset
from dataset.collate_fn import collate_fn
from model.encoder import PositionalEncoding, RivaConformerAudioEncoder
from model.fusion_module import CrossAttentionFusion
from model.decoder import CTCDecoder
from model.trainer import MultimodalTrainer
from utils.tokenizer import Tokenizer
from preprocessing import build_data_list

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_fixed_pairs(sentence_list, n_pairs=1000):
    pairs = []
    indices = list(range(len(sentence_list)))
    for _ in range(n_pairs):
        i, j = random.sample(indices, 2)
        pairs.append((sentence_list[i], sentence_list[j]))
    return pairs

# ✅ 체크포인트 저장
def save_checkpoint(epoch, trainer, path):
    torch.save({
        'epoch': epoch,
        'visual_encoder': trainer.visual_encoder.state_dict(),
        'audio_encoder': trainer.audio_encoder.state_dict(),
        'fusion': trainer.fusion_module.state_dict(),
        'decoder': trainer.decoder.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }, path)

# ✅ 체크포인트 불러오기
def load_checkpoint(trainer, path):
    checkpoint = torch.load(path, map_location=trainer.device)
    trainer.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
    trainer.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    trainer.fusion_module.load_state_dict(checkpoint['fusion'])
    trainer.decoder.load_state_dict(checkpoint['decoder'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'] + 1  # 다음 epoch부터 시작

def main():
    set_seed()

    # ✅ 경로 설정
    json_folder = "input_texts"
    npy_dir = "processed_dataset/npy"
    text_dir = "processed_dataset/text"
    wav_dir = "input_videos"

    tokenizer = Tokenizer(vocab_path="utils/tokenizer800.vocab")
    sentence_list = build_data_list(json_folder, npy_dir, text_dir, wav_dir)
    train_sent, val_sent = train_test_split(sentence_list, test_size=0.1, random_state=42)
    val_pairs = generate_fixed_pairs(val_sent, n_pairs=500)

    train_dataset = RandomSentencePairDataset(train_sent, tokenizer, num_pairs_per_epoch=10000)
    val_dataset = FixedSentencePairDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # ✅ 모델 구성
    visual_encoder = PositionalEncoding(
        hidden_dim=256, lstm_layers=2, bidirectional=True
    )
    audio_encoder = RivaConformerAudioEncoder(freeze=False)
    fusion = CrossAttentionFusion(
        visual_dim=visual_encoder.output_dim,
        audio_dim=audio_encoder.output_dim,
        fused_dim=512
    )
    decoder = CTCDecoder(input_dim=512, vocab_size=tokenizer.vocab_size, blank_id=tokenizer.blank_id)

    trainer = MultimodalTrainer(
        visual_encoder, audio_encoder, fusion, decoder,
        tokenizer,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ✅ 체크포인트 폴더 및 경로
    os.makedirs("checkpoints", exist_ok=True)
    last_ckpt_path = "checkpoints/last_checkpoint.pt"
    best_ckpt_path = "checkpoints/best_checkpoint.pt"
    start_epoch = 1
    best_wer = 1.0  # 낮을수록 좋음

    # ✅ 기존 체크포인트가 있으면 이어서 시작
    if os.path.exists(last_ckpt_path):
        logging.info("🔁 기존 체크포인트 불러오는 중...")
        start_epoch = load_checkpoint(trainer, last_ckpt_path)
        logging.info(f"➡️  Epoch {start_epoch}부터 재개")

    # ✅ 학습 루프
    for epoch in range(start_epoch, 21):
        logging.info(f"\n📚 Epoch {epoch}/20")
        loss = trainer.train_epoch(train_loader)

        wer_score = trainer.evaluate(val_loader)

        # 💾 마지막 상태 저장
        save_checkpoint(epoch, trainer, last_ckpt_path)
        logging.info("💾 마지막 체크포인트 저장 완료")

        # 🏅 Best 성능 모델 따로 저장
        if wer_score < best_wer:
            best_wer = wer_score
            save_checkpoint(epoch, trainer, best_ckpt_path)
            logging.info("🏅 Best 모델 갱신 및 저장 완료")

if __name__ == "__main__":
    main()
