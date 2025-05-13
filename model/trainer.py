import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer

class MultimodalTrainer:
    def __init__(self, visual_encoder, audio_encoder, fusion_module,
                 decoder1, decoder2, decoder_audio, decoder_visual,
                 tokenizer, learning_rate=1e-4, device="cuda"):
        self.visual_encoder = visual_encoder.to(device)
        self.audio_encoder = audio_encoder.to(device)
        self.fusion_module = fusion_module.to(device)

        self.decoder1 = decoder1.to(device)
        self.decoder2 = decoder2.to(device)
        self.decoder_audio = decoder_audio.to(device)
        self.decoder_visual = decoder_visual.to(device)

        self.tokenizer = tokenizer
        self.device = device

        self.ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

        self.parameters = (
            list(self.visual_encoder.parameters()) +
            list(self.audio_encoder.parameters()) +
            list(self.fusion_module.parameters()) +
            list(self.decoder1.parameters()) +
            list(self.decoder2.parameters()) +
            list(self.decoder_audio.parameters()) +
            list(self.decoder_visual.parameters())
        )

        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)

    def train_epoch(self, dataloader):
        print("✅ train_epoch() 진입")  # 디버깅 로그 추가

        self.visual_encoder.train()
        self.audio_encoder.train()
        self.fusion_module.train()
        self.decoder1.train()
        self.decoder2.train()
        self.decoder_audio.train()
        self.decoder_visual.train()

        total_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx == 0:
                print("🚀 첫 번째 배치 진입 성공")

            self.optimizer.zero_grad()

            lip1 = batch["lip1"].to(self.device)
            lip2 = batch["lip2"].to(self.device)
            audio = batch["audio"].to(self.device)
            text1 = batch["text1"].to(self.device)
            text2 = batch["text2"].to(self.device)
            len1 = batch["text1_lengths"].to(self.device)
            len2 = batch["text2_lengths"].to(self.device)

            visual_feat1 = self.visual_encoder(lip1)
            visual_feat2 = self.visual_encoder(lip2)
            audio_feat = self.audio_encoder(audio)

            fused_feat1 = self.fusion_module(visual_feat1, audio_feat)
            fused_feat2 = self.fusion_module(visual_feat2, audio_feat)

            B = audio_feat.size(0)
            input_lengths1 = torch.full((B,), fused_feat1.size(1), dtype=torch.long).to(self.device)
            input_lengths2 = torch.full((B,), fused_feat2.size(1), dtype=torch.long).to(self.device)
            input_lengths_audio = torch.full((B,), audio_feat.size(1), dtype=torch.long).to(self.device)
            input_lengths_visual1 = torch.full((B,), visual_feat1.size(1), dtype=torch.long).to(self.device)
            input_lengths_visual2 = torch.full((B,), visual_feat2.size(1), dtype=torch.long).to(self.device)

            log_probs1 = self.decoder1(fused_feat1)
            log_probs2 = self.decoder2(fused_feat2)
            log_probs_audio = self.decoder_audio(audio_feat)
            log_probs_visual1 = self.decoder_visual(visual_feat1)
            log_probs_visual2 = self.decoder_visual(visual_feat2)

            loss1 = self.ctc_loss(log_probs1.transpose(0, 1), text1, input_lengths1, len1)
            loss2 = self.ctc_loss(log_probs2.transpose(0, 1), text2, input_lengths2, len2)
            loss_audio = self.ctc_loss(log_probs_audio.transpose(0, 1), text1, input_lengths_audio, len1)
            loss_visual1 = self.ctc_loss(log_probs_visual1.transpose(0, 1), text1, input_lengths_visual1, len1)
            loss_visual2 = self.ctc_loss(log_probs_visual2.transpose(0, 1), text2, input_lengths_visual2, len2)

            loss = loss1 + loss2 + 0.5 * (loss_audio + loss_visual1 + loss_visual2)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        return total_loss / len(dataloader)

    def ctc_decode(self, pred_ids):
        result = []
        prev = None
        for idx in pred_ids:
            if idx == self.tokenizer.blank_id:
                continue
            if idx != prev:
                result.append(idx)
            prev = idx
        return result

    def evaluate(self, dataloader):
        self.visual_encoder.eval()
        self.audio_encoder.eval()
        self.fusion_module.eval()
        self.decoder1.eval()
        self.decoder2.eval()

        all_refs1, all_hyps1 = [], []
        all_refs2, all_hyps2 = [], []

        with torch.no_grad():
            for batch in dataloader:
                lip1 = batch["lip1"].to(self.device)
                lip2 = batch["lip2"].to(self.device)
                audio = batch["audio"].to(self.device)
                text1 = batch["text1"].to(self.device)
                text2 = batch["text2"].to(self.device)
                len1 = batch["text1_lengths"].to(self.device)
                len2 = batch["text2_lengths"].to(self.device)

                visual_feat1 = self.visual_encoder(lip1)
                visual_feat2 = self.visual_encoder(lip2)
                audio_feat = self.audio_encoder(audio)

                fused_feat1 = self.fusion_module(visual_feat1, audio_feat)
                fused_feat2 = self.fusion_module(visual_feat2, audio_feat)

                log_probs1 = self.decoder1(fused_feat1)
                log_probs2 = self.decoder2(fused_feat2)

                pred1 = torch.argmax(log_probs1, dim=-1).cpu().numpy()
                pred2 = torch.argmax(log_probs2, dim=-1).cpu().numpy()

                for p, t, l in zip(pred1, text1, len1):
                    p_ids = self.ctc_decode(p[:int(l)])
                    hyp = self.tokenizer.decode(p_ids)
                    ref = self.tokenizer.decode(t[:int(l)].cpu().numpy())
                    all_hyps1.append(hyp)
                    all_refs1.append(ref)

                for p, t, l in zip(pred2, text2, len2):
                    p_ids = self.ctc_decode(p[:int(l)])
                    hyp = self.tokenizer.decode(p_ids)
                    ref = self.tokenizer.decode(t[:int(l)].cpu().numpy())
                    all_hyps2.append(hyp)
                    all_refs2.append(ref)

        wer1 = wer(all_refs1, all_hyps1)
        wer2 = wer(all_refs2, all_hyps2)
        return (wer1 + wer2) / 2
