import torch
import torch.nn as nn
import torch.optim as optim
from jiwer import wer

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

        visual1 = batch["lip1"].to(self.device)
        visual2 = batch["lip2"].to(self.device)
        audio = batch["audio"].to(self.device)
        attention_mask = batch["audio_attention_mask"].to(self.device)
        target1 = batch["text1"].to(self.device)
        target2 = batch["text2"].to(self.device)
        v_len1 = batch["lip1_lengths"].to(self.device)
        v_len2 = batch["lip2_lengths"].to(self.device)
        t_len1 = batch["text1_lengths"].to(self.device)
        t_len2 = batch["text2_lengths"].to(self.device)

        audio_feat = self.audio_encoder(audio, attention_mask=attention_mask)
        visual_feat1 = self.visual_encoder(visual1)
        visual_feat2 = self.visual_encoder(visual2)

        fused_feat1 = self.fusion_module(visual_feat1, audio_feat)
        fused_feat2 = self.fusion_module(visual_feat2, audio_feat)

        loss1 = self.decoder(fused_feat1, target1, input_lengths=v_len1, target_lengths=t_len1)
        loss2 = self.decoder(fused_feat2, target2, input_lengths=v_len2, target_lengths=t_len2)
        loss = (loss1 + loss2) / 2

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate(self, dataloader):
        self.visual_encoder.eval()
        self.audio_encoder.eval()
        self.fusion_module.eval()
        self.decoder.eval()

        hypotheses = []
        references = []

        with torch.no_grad():
            for batch in dataloader:
                visual1 = batch["lip1"].to(self.device)
                visual2 = batch["lip2"].to(self.device)
                audio = batch["audio"].to(self.device)
                attention_mask = batch["audio_attention_mask"].to(self.device)
                target1 = batch["text1"].to(self.device)
                target2 = batch["text2"].to(self.device)
                v_len1 = batch["lip1_lengths"].to(self.device)
                v_len2 = batch["lip2_lengths"].to(self.device)

                audio_feat = self.audio_encoder(audio, attention_mask=attention_mask)
                visual_feat1 = self.visual_encoder(visual1)
                visual_feat2 = self.visual_encoder(visual2)

                fused_feat1 = self.fusion_module(visual_feat1, audio_feat)
                fused_feat2 = self.fusion_module(visual_feat2, audio_feat)

                log_probs1 = self.decoder(fused_feat1, None, input_lengths=v_len1)
                log_probs2 = self.decoder(fused_feat2, None, input_lengths=v_len2)
                pred1 = log_probs1.argmax(dim=-1)
                pred2 = log_probs2.argmax(dim=-1)

                for p1, t1, p2, t2 in zip(pred1, target1, pred2, target2):
                    pred_txt1 = self.tokenizer.decode(p1[p1 != 0].cpu().numpy())
                    ref_txt1 = self.tokenizer.decode(t1[t1 != 0].cpu().numpy())
                    pred_txt2 = self.tokenizer.decode(p2[p2 != 0].cpu().numpy())
                    ref_txt2 = self.tokenizer.decode(t2[t2 != 0].cpu().numpy())
                    hypotheses.extend([pred_txt1, pred_txt2])
                    references.extend([ref_txt1, ref_txt2])

        return wer(references, hypotheses)

    def train_epoch(self, dataloader):
        total_loss = 0
        for batch in dataloader:
            loss = self.train_step(batch)
            total_loss += loss
        avg_loss = total_loss / len(dataloader)
        print(f"✅ Train Loss: {avg_loss:.4f}")
        return avg_loss
