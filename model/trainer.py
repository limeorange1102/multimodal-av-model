import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer

class MultimodalTrainer:
    def __init__(self, visual_encoder, audio_encoder, fusion_module,
                 decoder, decoder_audio, decoder_visual,
                 tokenizer, learning_rate=1e-4, device="cuda"):
        self.visual_encoder = visual_encoder.to(device)
        self.audio_encoder = audio_encoder.to(device)
        self.fusion_module = fusion_module.to(device)

        self.decoder = decoder.to(device)
        self.decoder_audio = decoder_audio.to(device)
        self.decoder_visual = decoder_visual.to(device)

        self.tokenizer = tokenizer
        self.device = device

        self.ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

        self.parameters = (
            list(self.visual_encoder.parameters()) +
            list(self.audio_encoder.parameters()) +
            list(self.fusion_module.parameters()) +
            list(self.decoder.parameters()) +
            list(self.decoder_audio.parameters()) +
            list(self.decoder_visual.parameters())
        )

        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)

    def train_epoch(self, dataloader):
        self.visual_encoder.train()
        self.audio_encoder.train()
        self.fusion_module.train()
        self.decoder.train()
        self.decoder_audio.train()
        self.decoder_visual.train()

        total_loss = 0
        for batch in dataloader:
            self.optimizer.zero_grad()

            lip1 = batch["lip1"].to(self.device)
            lip2 = batch["lip2"].to(self.device)
            audio = batch["audio"].to(self.device)
            text1 = batch["text1"].to(self.device)
            text2 = batch["text2"].to(self.device)

            target = text1  # 주 화자의 문장만 인식
            target_lengths = batch["text1_len"]

            visual_feat = self.visual_encoder(lip1)
            audio_feat = self.audio_encoder(audio)
            fused_feat = self.fusion_module(visual_feat, audio_feat)

            # 길이 계산
            B = fused_feat.size(0)
            T_fused = fused_feat.size(1)
            T_audio = audio_feat.size(1)
            T_visual = visual_feat.size(1)

            input_lengths_fused = torch.full((B,), T_fused, dtype=torch.long).to(self.device)
            input_lengths_audio = torch.full((B,), T_audio, dtype=torch.long).to(self.device)
            input_lengths_visual = torch.full((B,), T_visual, dtype=torch.long).to(self.device)

            log_probs_fused = self.decoder(fused_feat)
            log_probs_audio = self.decoder_audio(audio_feat)
            log_probs_visual = self.decoder_visual(visual_feat)

            loss_fused = self.ctc_loss(log_probs_fused.transpose(0, 1), target, input_lengths_fused, target_lengths)
            loss_audio = self.ctc_loss(log_probs_audio.transpose(0, 1), target, input_lengths_audio, target_lengths)
            loss_visual = self.ctc_loss(log_probs_visual.transpose(0, 1), target, input_lengths_visual, target_lengths)

            loss = loss_fused + 0.5 * (loss_audio + loss_visual)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.visual_encoder.eval()
        self.audio_encoder.eval()
        self.fusion_module.eval()
        self.decoder.eval()

        all_refs, all_hyps = [], []

        with torch.no_grad():
            for batch in dataloader:
                lip1 = batch["lip1"].to(self.device)
                audio = batch["audio"].to(self.device)
                text1 = batch["text1"].to(self.device)

                visual_feat = self.visual_encoder(lip1)
                audio_feat = self.audio_encoder(audio)
                fused_feat = self.fusion_module(visual_feat, audio_feat)
                log_probs = self.decoder(fused_feat)

                pred = torch.argmax(log_probs, dim=-1).cpu().numpy()

                for p, t in zip(pred, text1):
                    hyp = self.tokenizer.decode(p)
                    ref = self.tokenizer.decode(t.cpu().numpy())
                    all_hyps.append(hyp)
                    all_refs.append(ref)

        return wer(all_refs, all_hyps)
