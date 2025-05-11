import torch
from torch.utils.data import DataLoader
import os, random, numpy as np
from sklearn.model_selection import train_test_split

from dataset.multi_speaker_dataset import RandomSentencePairDataset, FixedSentencePairDataset
from dataset.collate_fn import collate_fn
from model.encoder import VisualEncoder, RivaConformerAudioEncoder
from model.fusion_module import FusionModule
from model.decoder import CTCDecoder
from model.trainer import MultimodalTrainer
from utils.tokenizer import Tokenizer
from preprocessing import build_data_list

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def generate_fixed_pairs(sentence_list, n_pairs=1000):
    # validation용 고정 쌍 만들기
    pairs = []
    indices = list(range(len(sentence_list)))
    for _ in range(n_pairs):
        i, j = random.sample(indices, 2)
        pairs.append((sentence_list[i], sentence_list[j]))
    return pairs

def main():
    set_seed()

    # ✅ 경로 설정
    json_folder = "input_texts"
    npy_dir = "processed_dataset/npy"
    text_dir = "processed_dataset/text"
    wav_dir = "input_videos"

    # ✅ tokenizer
    tokenizer = Tokenizer(vocab_path="utils/tokenizer800.vocab")

    # ✅ 전체 문장 리스트 생성
    sentence_list = build_data_list(json_folder, npy_dir, text_dir, wav_dir)

    # ✅ train / val 분할
    train_sent, val_sent = train_test_split(sentence_list, test_size=0.1, random_state=42)
    val_pairs = generate_fixed_pairs(val_sent, n_pairs=500)

    # ✅ Dataset & Loader
    train_dataset = RandomSentencePairDataset(train_sent, tokenizer, num_pairs_per_epoch=10000)
    val_dataset = FixedSentencePairDataset(val_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # ✅ 모델
    visual_encoder = VisualEncoder(
        pretrained_path="weights/Video_only_model.pt",
        hidden_dim=256, lstm_layers=2, bidirectional=True
    )
    audio_encoder = RivaConformerAudioEncoder(freeze=False)
    fusion = FusionModule(
        visual_dim=visual_encoder.output_dim,
        audio_dim=audio_encoder.output_dim,
        fused_dim=512
    )
    decoder = CTCDecoder(input_dim=512, vocab_size=tokenizer.vocab_size)

    trainer = MultimodalTrainer(
        visual_encoder, audio_encoder, fusion, decoder,
        tokenizer,
        learning_rate=1e-4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ✅ 학습
    os.makedirs("checkpoints", exist_ok=True)
    for epoch in range(1, 21):
        print(f"\n📚 Epoch {epoch}/20")
        trainer.train_epoch(train_loader)
        trainer.evaluate(val_loader)

        torch.save({
            'visual_encoder': visual_encoder.state_dict(),
            'audio_encoder': audio_encoder.state_dict(),
            'fusion': fusion.state_dict(),
            'decoder': decoder.state_dict(),
        }, f"checkpoints/epoch_{epoch}.pt")

if __name__ == "__main__":
    main()
