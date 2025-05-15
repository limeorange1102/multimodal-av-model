import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer
from tqdm import tqdm
import numpy as np

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

    def crop_or_pad_feat(self, feat, target_len):
        if feat.size(0) >= target_len:
            return feat[:target_len]
        else:
            pad_len = target_len - feat.size(0)
            pad = torch.zeros(pad_len, feat.size(1), device=feat.device, dtype=feat.dtype)
            return torch.cat([feat, pad], dim=0)

    def train_epoch(self, dataloader):
        print("✅ train_epoch() 짱입")

        self.visual_encoder.train()
        self.audio_encoder.train()
        self.fusion_module.train()
        self.decoder1.train()
        self.decoder2.train()
        self.decoder_audio.train()
        self.decoder_visual.train()

        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", ncols=100)):
            try:
                self.optimizer.zero_grad()

                lip1 = batch["lip1"].to(self.device)
                lip2 = batch["lip2"].to(self.device)
                audio = batch["audio"].to(self.device)
                audio_mask = batch["audio_attention_mask"].to(self.device)
                text1 = batch["text1"].to(self.device)
                text2 = batch["text2"].to(self.device)
                len1 = batch["text1_lengths"].to(self.device)
                len2 = batch["text2_lengths"].to(self.device)
                lip1_lengths = batch["lip1_lengths"].to(self.device)
                lip2_lengths = batch["lip2_lengths"].to(self.device)

                visual_feat1 = self.visual_encoder(lip1)
                visual_feat2 = self.visual_encoder(lip2)
                audio_feat = self.audio_encoder(audio, attention_mask=audio_mask)

                audio_visual_time_ratio = 2
                max_len1 = (lip1_lengths * audio_visual_time_ratio).max()
                max_len2 = (lip2_lengths * audio_visual_time_ratio).max()

                audio_feat1 = torch.stack([
                    self.crop_or_pad_feat(audio_feat[i], max_len1)
                    for i in range(audio_feat.size(0))
                ])
                audio_feat2 = torch.stack([
                    self.crop_or_pad_feat(audio_feat[i], max_len2)
                    for i in range(audio_feat.size(0))
                ])

                fused_feat1 = self.fusion_module(visual_feat1, audio_feat1)
                fused_feat2 = self.fusion_module(visual_feat2, audio_feat2)

                input_lengths1 = (lip1_lengths * audio_visual_time_ratio).to(torch.long)
                input_lengths2 = (lip2_lengths * audio_visual_time_ratio).to(torch.long)

                input_lengths_audio = torch.full((audio_feat.size(0),), audio_feat.size(1), dtype=torch.long).to(self.device)
                input_lengths_visual1 = torch.full((visual_feat1.size(0),), visual_feat1.size(1), dtype=torch.long).to(self.device)
                input_lengths_visual2 = torch.full((visual_feat2.size(0),), visual_feat2.size(1), dtype=torch.long).to(self.device)

                log_probs1 = self.decoder1(fused_feat1)
                log_probs2 = self.decoder2(fused_feat2)
                log_probs_audio1 = self.decoder_audio(audio_feat1)
                log_probs_audio2 = self.decoder_audio(audio_feat2)
                log_probs_visual1 = self.decoder_visual(visual_feat1)
                log_probs_visual2 = self.decoder_visual(visual_feat2)

                loss1 = self.ctc_loss(log_probs1.transpose(0, 1), text1, input_lengths1, len1)
                loss2 = self.ctc_loss(log_probs2.transpose(0, 1), text2, input_lengths2, len2)
                loss_audio1 = self.ctc_loss(log_probs_audio1.transpose(0, 1), text1, input_lengths1, len1)
                loss_audio2 = self.ctc_loss(log_probs_audio2.transpose(0, 1), text2, input_lengths2, len2)
                loss_visual1 = self.ctc_loss(log_probs_visual1.transpose(0, 1), text1, input_lengths_visual1, len1)
                loss_visual2 = self.ctc_loss(log_probs_visual2.transpose(0, 1), text2, input_lengths_visual2, len2)

                loss = loss1 + loss2 + 1.0 * (loss_audio1 + loss_audio2) + 1.5 * (loss_visual1 + loss_visual2)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

                if batch_idx % 100 == 0:
                    pred_ids = torch.argmax(log_probs1[0], dim=-1).cpu().tolist()
                    unique_ids = sorted(set(pred_ids))
                    print(f"[진단] Batch {batch_idx} - 예측 토큰 ID (앞 20개): {pred_ids[:20]}", flush=True)
                    print(f"[진단] 고유 토큰 ID들: {unique_ids}", flush=True)
                    print(f"\n🔎 [Batch {batch_idx}] 예측 결과 확인", flush = True)
                    with torch.no_grad():
                        pred1_ids = torch.argmax(log_probs1, dim=-1)
                        pred2_ids = torch.argmax(log_probs2, dim=-1)

                        for i in range(min(2, pred1_ids.size(0))):
                            pred_ids1 = self.ctc_decode(pred1_ids[i].cpu().tolist())
                            decoded1 = self.tokenizer.decode(pred_ids1)
                            true1 = self.tokenizer.decode(text1[i][:len1[i]].cpu().tolist())
                            print(f"[화자1 예측] {decoded1}", flush=True)
                            print(f"[화자1 정답] {true1}", flush=True)

                            pred_ids2 = self.ctc_decode(pred2_ids[i].cpu().tolist())
                            decoded2 = self.tokenizer.decode(pred_ids2)
                            true2 = self.tokenizer.decode(text2[i][:len2[i]].cpu().tolist())
                            print(f"[화자2 예측] {decoded2}", flush=True)
                            print(f"[화자2 정답] {true2}", flush=True)
            except Exception as e:
                print(f"❌ Error at batch {batch_idx}: {e}", flush=True)
                continue

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
                audio_mask = batch["audio_attention_mask"].to(self.device)
                text1 = batch["text1"].to(self.device)
                text2 = batch["text2"].to(self.device)
                len1 = batch["text1_lengths"].to(self.device)
                len2 = batch["text2_lengths"].to(self.device)
                lip1_lengths = batch["lip1_lengths"].to(self.device)
                lip2_lengths = batch["lip2_lengths"].to(self.device)

                audio_feat = self.audio_encoder(audio, attention_mask=audio_mask)
                visual_feat1 = self.visual_encoder(lip1)
                visual_feat2 = self.visual_encoder(lip2)

                audio_visual_time_ratio = 2
                max_len1 = (lip1_lengths * audio_visual_time_ratio).max()
                max_len2 = (lip2_lengths * audio_visual_time_ratio).max()

                audio_feat1 = torch.stack([
                    self.crop_or_pad_feat(audio_feat[i], max_len1)
                    for i in range(audio_feat.size(0))
                ])
                audio_feat2 = torch.stack([
                    self.crop_or_pad_feat(audio_feat[i], max_len2)
                    for i in range(audio_feat.size(0))
                ])

                fused_feat1 = self.fusion_module(visual_feat1, audio_feat1)
                fused_feat2 = self.fusion_module(visual_feat2, audio_feat2)

                log_probs1 = self.decoder1(fused_feat1)
                log_probs2 = self.decoder2(fused_feat2)

                pred1 = torch.argmax(log_probs1, dim=-1).cpu().numpy()
                pred2 = torch.argmax(log_probs2, dim=-1).cpu().numpy()

                input_lengths1 = (lip1_lengths * audio_visual_time_ratio).to(torch.long)
                input_lengths2 = (lip2_lengths * audio_visual_time_ratio).to(torch.long)

                for i in range(len(pred1)):
                    p_ids = self.ctc_decode(pred1[i][:input_lengths1[i]])
                    ref = self.tokenizer.decode(text1[i][:len1[i]].cpu().numpy())
                    hyp = self.tokenizer.decode(p_ids)
                    all_hyps1.append(hyp)
                    all_refs1.append(ref)

                for i in range(len(pred2)):
                    p_ids = self.ctc_decode(pred2[i][:input_lengths2[i]])
                    ref = self.tokenizer.decode(text2[i][:len2[i]].cpu().numpy())
                    hyp = self.tokenizer.decode(p_ids)
                    all_hyps2.append(hyp)
                    all_refs2.append(ref)

        wer1 = wer(all_refs1, all_hyps1)
        wer2 = wer(all_refs2, all_hyps2)
        avg_wer = (wer1 + wer2) / 2

        sentence_acc1 = np.mean([ref.strip() == hyp.strip() for ref, hyp in zip(all_refs1, all_hyps1)])
        sentence_acc2 = np.mean([ref.strip() == hyp.strip() for ref, hyp in zip(all_refs2, all_hyps2)])
        avg_sentence_acc = (sentence_acc1 + sentence_acc2) / 2

        print(f"✅ Eval Results: WER1={wer1:.3f}, WER2={wer2:.3f}, SentenceAcc={avg_sentence_acc:.3f}", flush=True)

        return avg_wer, avg_sentence_acc
