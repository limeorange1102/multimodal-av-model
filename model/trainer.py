import torch
import torch.nn as nn
import torch.optim as optim
from jiwer import wer  # for evaluation metric
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


class MultimodalTrainer:
    def __init__(self, visual_encoder, audio_encoder, fusion_module, decoder, tokenizer, device='cuda', learning_rate=1e-4):
        self.visual_encoder = visual_encoder.to(device)
        self.audio_encoder = audio_encoder.to(device)
        self.fusion_module = fusion_module.to(device)
        self.decoder = decoder.to(device)
        self.tokenizer = tokenizer
        self.device = device

        self.parameters = list(self.visual_encoder.parameters()) + \
                          list(self.audio_encoder.parameters()) + \
                          list(self.fusion_module.parameters()) + \
                          list(self.decoder.parameters())

        self.optimizer = optim.Adam(self.parameters, lr=learning_rate)

    def train_step(self, batch):
        self.visual_encoder.train()
        self.audio_encoder.train()
        self.fusion_module.train()
        self.decoder.train()

        visual = batch["lip1"].to(self.device)
        audio = batch["audio"].to(self.device)
        target = batch["text1"].to(self.device)
        v_len = batch["lip1_lengths"].to(self.device)
        a_len = batch["audio_lengths"].to(self.device)
        t_len = batch["text1_lengths"].to(self.device)

        visual_feat = self.visual_encoder(visual)
        audio_feat = self.audio_encoder(audio)
        fused_feat = self.fusion_module(visual_feat, audio_feat)

        loss = self.decoder(fused_feat, target, input_lengths=v_len, target_lengths=t_len)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_epoch(self, train_loader):
        self.visual_encoder.train()
        self.audio_encoder.train()
        self.fusion_module.train()
        self.decoder.train()

        total_loss = 0
        for batch in train_loader:
            loss = self.train_step(batch)
            total_loss += loss
        avg_loss = total_loss / len(train_loader)
        logging.info(f"✅ 평균 Training Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, dataloader):
        self.visual_encoder.eval()
        self.audio_encoder.eval()
        self.fusion_module.eval()
        self.decoder.eval()

        hypotheses1, references1 = [], []
        hypotheses2, references2 = [], []

        with torch.no_grad():
            for batch in dataloader:
                audio = batch["audio"].to(self.device)

                # --- text1 기준 평가 ---
                visual1 = batch["lip1"].to(self.device)
                target1 = batch["text1"].to(self.device)
                v_len1 = batch["lip1_lengths"].to(self.device)

                visual_feat1 = self.visual_encoder(visual1)
                audio_feat = self.audio_encoder(audio)
                fused_feat1 = self.fusion_module(visual_feat1, audio_feat)
                log_probs1 = self.decoder(fused_feat1, None, input_lengths=v_len1)
                pred1 = log_probs1.argmax(dim=-1)

                for p, t in zip(pred1, target1):
                    hypotheses1.append(self.tokenizer.decode(p[p != self.tokenizer.blank_id].cpu().numpy()))
                    references1.append(self.tokenizer.decode(t[t != self.tokenizer.pad_id].cpu().numpy()))

                # --- text2 기준 평가 ---
                visual2 = batch["lip2"].to(self.device)
                target2 = batch["text2"].to(self.device)
                v_len2 = batch["lip2_lengths"].to(self.device)

                visual_feat2 = self.visual_encoder(visual2)
                fused_feat2 = self.fusion_module(visual_feat2, audio_feat)
                log_probs2 = self.decoder(fused_feat2, None, input_lengths=v_len2)
                pred2 = log_probs2.argmax(dim=-1)

                for p, t in zip(pred2, target2):
                    hypotheses2.append(self.tokenizer.decode(p[p != self.tokenizer.blank_id].cpu().numpy()))
                    references2.append(self.tokenizer.decode(t[t != self.tokenizer.pad_id].cpu().numpy()))

        wer1 = wer(references1, hypotheses1)
        wer2 = wer(references2, hypotheses2)
        avg_wer = (wer1 + wer2) / 2

        logging.info(f"🔎 WER1 (text1 기준): {wer1:.4f}")
        logging.info(f"🔎 WER2 (text2 기준): {wer2:.4f}")
        logging.info(f"🔎 평균 WER: {avg_wer:.4f}")

        return avg_wer