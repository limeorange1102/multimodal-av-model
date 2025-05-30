import torch
from torch.utils.data import DataLoader
import os, random, numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from dataset.multi_speaker_dataset import RandomSentencePairDataset, FixedSentencePairDataset
from dataset.collate_fn import collate_fn
from model.encoder import VisualEncoder, AudioEncoder
from model.fusion_module import CrossAttentionFusion
from model.decoder import CTCDecoder
from model.trainer import MultimodalTrainer
from utils.tokenizer import Tokenizer
from preprocessing import build_data_list

import logging

import soundfile as sf
import imageio

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def unfreeze_last_n_layers(model, n=3):
    for name, param in model.named_parameters():
        if any(f"encoder.layers.{i}." in name for i in range(12 - n, 12)):
            param.requires_grad = True
        else:
            param.requires_grad = False

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
        'optimizer': trainer.optimizer.state_dict(),
    }, path)

def load_checkpoint(trainer, path):
    checkpoint = torch.load(path, map_location=trainer.device)
    trainer.visual_encoder.load_state_dict(checkpoint['visual_encoder'])
    trainer.audio_encoder.load_state_dict(checkpoint['audio_encoder'])
    trainer.fusion_module.load_state_dict(checkpoint['fusion'])
    trainer.decoder1.load_state_dict(checkpoint['decoder1'])
    trainer.optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch'] + 1

def main():
    set_seed()

    json_folder = "input_texts"
    npy_dir = "npy"
    text_dir = "processed_dataset/text"
    wav_dir = "input_wav/input_wav"

    tokenizer = Tokenizer(vocab_path="input_videos/tokenizer800.vocab")
    sentence_list = build_data_list(json_folder, npy_dir, text_dir, wav_dir)    

    train_set, temp_set = train_test_split(sentence_list, test_size=0.1, random_state=42)
    val_set, test_set = train_test_split(temp_set, test_size=0.5, random_state=42)
    
    train_dataset = RandomSentencePairDataset(train_set, tokenizer, num_pairs_per_epoch=10000)

    val_pairs = generate_fixed_pairs(val_set, n_pairs=500)
    val_dataset = FixedSentencePairDataset(val_pairs, tokenizer)

    test_pairs = generate_fixed_pairs(test_set, n_pairs=500)
    test_dataset = FixedSentencePairDataset(test_pairs, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=collate_fn)

    visual_encoder = VisualEncoder(relu_type='prelu')
    # 🔽 best_checkpoint.pt에서 visual encoder만 불러오기
    best_ckpt_path = "/content/drive/MyDrive/lip_audio_multimodal/checkpoints_visual_loss/best_loss_encoder.pt"
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    visual_encoder.load_state_dict(ckpt)
    print("✅ visual encoder loaded from best_checkpoint.pt")

    # 🔽 freeze
    for param in visual_encoder.trunk.parameters():
        param.requires_grad = False
    for param in visual_encoder.frontend3D.parameters():
        param.requires_grad = False

    audio_encoder = AudioEncoder(freeze=True)
    unfreeze_last_n_layers(audio_encoder.model)

    fusion = CrossAttentionFusion(
        visual_dim=visual_encoder.output_dim,
        audio_dim=audio_encoder.output_dim,
        fused_dim=512
    )

    decoder1 = CTCDecoder(
        input_dim=1024,
        vocab_size=tokenizer.vocab_size,
        blank_id=tokenizer.blank_id
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainer = MultimodalTrainer(
        visual_encoder=visual_encoder,
        audio_encoder=audio_encoder,
        fusion_module=fusion,
        decoder1=decoder1,
        tokenizer=tokenizer,
        learning_rate=3e-4,
        device=device
    )

    drive_ckpt_dir = "/content/drive/MyDrive/lip_audio_multimodal/checkpoints"
    os.makedirs(drive_ckpt_dir, exist_ok=True)

    last_ckpt_path = os.path.join(drive_ckpt_dir, "last_checkpoint.pt")
    best_ckpt_path = os.path.join(drive_ckpt_dir, "best_checkpoint.pt")
    best_loss_path = os.path.join(drive_ckpt_dir, "best_loss.pt")
    eval_log_path = os.path.join(drive_ckpt_dir, "eval_log.csv")
    train_log_path = os.path.join(drive_ckpt_dir, "train_log.csv")
    start_epoch = 1
    best_wer = float('inf')
    best_loss = float('inf')
    patience = 5
    wer_history = []
    loss_history = []

    if os.path.exists(last_ckpt_path):
        logging.info("🔁 기존 체크포인트 불러오는 중...")
        print("🔁 기존 체크포인트 불러오는 중...", flush=True)
        start_epoch = load_checkpoint(trainer, last_ckpt_path)
        logging.info(f"➡️  Epoch {start_epoch}부터 재개")
        print(f"➡️  Epoch {start_epoch}부터 재개", flush=True)
    print(f"🧪 start_epoch={start_epoch}")

    with open(eval_log_path, "w") as f:
        f.write("epoch,wer1,wer2,average_wer\n")
    with open(train_log_path, "w") as f:
        f.write("epoch,loss\n")
    print("▶️ for epoch 진입", flush=True)

    max_epoch = 50
    for epoch in range(start_epoch, max_epoch + 1):
        logging.info(f"\n📚 Epoch {epoch}/20")
        print(f"\n📚 Epoch {epoch}/20", flush=True)

        loss = trainer.train_epoch(train_loader)
        loss_history.append(loss)

        eval_loss, eval_wer = trainer.evaluate(val_loader)
        
        loss_history.append(eval_loss)
        wer_history.append(eval_wer)

        with open(eval_log_path, "a") as f:
            f.write(f"{epoch},{eval_loss:.4f},{eval_wer:.4f}\n")
        with open(train_log_path, "a") as f:
            f.write(f"{epoch},{loss:.4f}\n")

        save_checkpoint(epoch, trainer, last_ckpt_path)
        logging.info("💾 마지막 체크포인트 저장 완료")
        print("💾 마지막 체크포인트 저장 완료", flush=True)

        if eval_wer < best_wer:
            best_wer = eval_wer
            save_checkpoint(epoch, trainer, best_ckpt_path)
            logging.info("🏅 Best 모델 갱신 및 저장 완료")
            print("🏅 Best 모델 갱신 및 저장 완료", flush=True)

        if eval_loss < best_loss:
            best_loss = eval_loss
            no_improve_counter = 0
            save_checkpoint(epoch, trainer, best_loss_path)
            logging.info("🏅 Best Loss 모델 갱신 및 저장 완료")
            print("🏅 Best Loss 모델 갱신 및 저장 완료", flush=True)
        else:
            no_improve_counter += 1
            print(f"🔻 Loss 감소 무: {no_improve_counter}, {best_loss}/", flush=True)
        
        if no_improve_counter >= patience:
            logging.info(f"⏳ Early stopping: {patience} epochs without improvement")
            print(f"⏳ Early stopping: {patience} epochs without improvement", flush=True)
            break

if __name__ == "__main__":
    main()
