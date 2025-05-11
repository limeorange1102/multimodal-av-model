import torch
import torch.nn as nn
import torch.optim as optim
from jiwer import wer  # for evaluation metric


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
        return total_loss / len(train_loader)

    def evaluate(self, dataloader):
        self.visual_encoder.eval()
        self.audio_encoder.eval()
        self.fusion_module.eval()
        self.decoder.eval()

        hypotheses = []
        references = []

        with torch.no_grad():
            for batch in dataloader:
                visual = batch["lip1"].to(self.device)
                audio = batch["audio"].to(self.device)
                target = batch["text1"].to(self.device)
                v_len = batch["lip1_lengths"].to(self.device)

                visual_feat = self.visual_encoder(visual)
                audio_feat = self.audio_encoder(audio)
                fused_feat = self.fusion_module(visual_feat, audio_feat)

                log_probs = self.decoder(fused_feat, None, input_lengths=v_len)
                pred = log_probs.argmax(dim=-1)

                for p, t in zip(pred, target):
                    pred_txt = self.tokenizer.decode(p[p != 0].cpu().numpy())
                    ref_txt = self.tokenizer.decode(t[t != 0].cpu().numpy())
                    hypotheses.append(pred_txt)
                    references.append(ref_txt)

        return wer(references, hypotheses)

