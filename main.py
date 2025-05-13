import torch
from torch.utils.data import DataLoader
import os, random, numpy as np
from sklearn.model_selection import train_test_split

from dataset.multi_speaker_dataset import RandomSentencePairDataset, FixedSentencePairDataset
from dataset.collate_fn import collate_fn
from model.encoder import VisualEncoder, AudioEncoder
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

def save_checkpoint(epoch, trainer, path):
    torch.save({
        'epoch': epoch,
        'visual_encoder': trainer.visual_encoder.state_dict(),
        'audio_encoder': trainer.audio_encoder.state_dict(),
        'fusion': trainer.fusion_module.state_dict(),
        'decoder1': trainer.decoder1.state_dict(),
        'decoder2': trainer.decoder2.state_dict(),
        'decoder_audio': trainer.decoder_audio.state_dict(),
        'decoder_visual': trainer.decoder_visual.state_dict(),
        'optimizer': trainer.optimizer.state_dict(),
    }, path)

def load_checkpoint(trainer, path):
    checkpoint = torch.load(path, map_location=trainer.device)
    trainer.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
    trainer.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    trainer.fusion_module.load_state_dict(checkpoint['fusion'])
    trainer.decoder1.load_state_dict(checkpoint['decoder1'])
    trainer.decoder2.load_state_dict(checkpoint['decoder2'])
    trainer.decoder_audio.load_state_dict(checkpoint['decoder_audio'])
    trainer.decoder_visual.load_state_dict(checkpoint['decoder_visual'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'] + 1

def main():
    set_seed()

    json_folder = "input_texts"
    npy_dir = "processed_dataset/npy"
    text_dir = "processed_dataset/text"
    wav_dir = "input_videos"

    tokenizer = Tokenizer(vocab_path="input_videos/tokenizer800.vocab")
    sentence_list = build_data_list(json_folder, npy_dir, text_dir, wav_dir)
    train_sent, val_sent = train_test_split(sentence_list, test_size=0.1, random_state=42)
    val_pairs = generate_fixed_pairs(val_sent, n_pairs=500)

    train_dataset = RandomSentencePairDataset(train_sent, tokenizer, num_pairs_per_epoch=10000)
    val_dataset = FixedSentencePairDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    visual_encoder = VisualEncoder(
        pretrained_path="weights/Video_only_model.pt",
        hidden_dim=256,
        lstm_layers=2,
        bidirectional=True
    )

    audio_encoder = AudioEncoder(freeze=False)

    fusion = CrossAttentionFusion(
        visual_dim=visual_encoder.output_dim,
        audio_dim=audio_encoder.output_dim,
        fused_dim=512
    )

    decoder1 = CTCDecoder(
        input_dim=512,
        vocab_size=tokenizer.vocab_size,
        blank_id=tokenizer.blank_id
    )

    decoder2 = CTCDecoder(
        input_dim=512,
        vocab_size=tokenizer.vocab_size,
        blank_id=tokenizer.blank_id
    )

    decoder_audio = CTCDecoder(
        input_dim=audio_encoder.output_dim,
        vocab_size=tokenizer.vocab_size,
        blank_id=tokenizer.blank_id
    )

    decoder_visual = CTCDecoder(
        input_dim=visual_encoder.output_dim,
        vocab_size=tokenizer.vocab_size,
        blank_id=tokenizer.blank_id
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = MultimodalTrainer(
        visual_encoder, audio_encoder, fusion,
        decoder1, decoder2, decoder_audio, decoder_visual,
        tokenizer,
        learning_rate=1e-4,
        device=device
    )

    os.makedirs("checkpoints", exist_ok=True)
    last_ckpt_path = "checkpoints/last_checkpoint.pt"
    best_ckpt_path = "checkpoints/best_checkpoint.pt"
    start_epoch = 1
    best_wer = 1.0

    if os.path.exists(last_ckpt_path):
        logging.info("🔁 기존 체크포인트 불러오는 중...")
        start_epoch = load_checkpoint(trainer, last_ckpt_path)
        logging.info(f"➡️  Epoch {start_epoch}부터 재개")

    for epoch in range(start_epoch, 21):
        logging.info(f"\n📚 Epoch {epoch}/20")
        loss = trainer.train_epoch(train_loader)

        wer_score = trainer.evaluate(val_loader)

        save_checkpoint(epoch, trainer, last_ckpt_path)
        logging.info("💾 마지막 체크포인트 저장 완료")

        if wer_score < best_wer:
            best_wer = wer_score
            save_checkpoint(epoch, trainer, best_ckpt_path)
            logging.info("🏅 Best 모델 갱신 및 저장 완료")

if __name__ == "__main__":
    main()
