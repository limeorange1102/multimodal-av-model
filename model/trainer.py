import torch
import torch.nn as nn
import torch.nn.functional as F
from jiwer import wer
from tqdm import tqdm
import numpy as np
from contrastive import contrastive_loss_with_mask
from beam_search import simple_beam_search, fast_decode

class MultimodalTrainer:
    def __init__(self, visual_encoder, audio_encoder, fusion_module,
                 decoder1, tokenizer, learning_rate=1e-4, device="cuda", lambda_=0.1):
        self.visual_encoder = visual_encoder.to(device)
        self.audio_encoder = audio_encoder.to(device)
        self.fusion_module = fusion_module.to(device)

        self.decoder1 = decoder1.to(device)

        self.tokenizer = tokenizer
        self.device = device
        self.lambda_ = lambda_

        self.ctc_loss = nn.CTCLoss(blank=tokenizer.blank_id, zero_infinity=True)

        self.parameters = (
            list(self.visual_encoder.parameters()) +
            list(self.audio_encoder.parameters()) +
            list(self.fusion_module.parameters()) +
            list(self.decoder1.parameters()) 
        )

        self.optimizer = torch.optim.Adam(self.parameters, lr=learning_rate)

    def crop_or_pad_feat(self, feat, target_len):
        if feat.size(0) >= target_len:
            return feat[:target_len]
        else:
            pad_len = target_len - feat.size(0)
            pad = torch.zeros(pad_len, feat.size(1), device=feat.device)
            return torch.cat([feat, pad], dim=0)

    def train_epoch(self, dataloader):
        print("✅ train_epoch() 짱입")
        

        self.visual_encoder.train()
        self.audio_encoder.train()
        self.fusion_module.train()
        self.decoder1.train()
        self.projection_layer = None  # projection layer for contrastive loss, if needed
        lambda_ = self.lambda_

        total_loss = 0
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training", ncols=100)):
            try:
                self.optimizer.zero_grad()

                lip1 = batch["lip1"].to(self.device).permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W] → [B, C, T, H, W], C=1, H, W=96
                lip2 = batch["lip2"].to(self.device).permute(0, 2, 1, 3, 4).contiguous()
                audio = batch["audio"].to(self.device)
                mask1 = batch["mask1"].to(self.device) # [B, T_audio]
                mask2 = batch["mask2"].to(self.device)
                
                text1 = batch["text1"].to(self.device)
                len1 = batch["text1_lengths"].to(self.device)
                text2 = batch["text2"].to(self.device)
                len2 = batch["text2_lengths"].to(self.device)

                for i in range(text1.size(0)):
                    if len1[i] == 0:
                        print(f"[디버그] 🚨 text1[{i}] 길이가 0입니다. 라벨: {text1[i].tolist()}", flush = True)

                    if text1[i].max() >= self.tokenizer.vocab_size:
                        print(f"[디버그] 🚨 text1[{i}]에 vocab_size 이상 값 존재: {text1[i].tolist()}", flush = True)

                    if text1[i].min() < 0:
                        print(f"[디버그] 🚨 text1[{i}]에 음수 인덱스 존재: {text1[i].tolist()}", flush = True)


                visual_feat1 = self.visual_encoder(lip1)
                visual_feat2 = self.visual_encoder(lip2)
                attn_mask1 = (mask1 != 3)
                attn_mask2 = (mask2 != 3)
                audio_feat1, audio_feat1_middle = self.audio_encoder(audio, attention_mask=attn_mask1)
                audio_feat2, audio_feat2_middle = self.audio_encoder(audio, attention_mask=attn_mask2)

                B, T_enc, D = audio_feat1.shape
                mask1_ds = F.interpolate(mask1.unsqueeze(1).float(), size=T_enc, mode='nearest').squeeze(1).long()  # [B, T_enc]
                mask1_flat = mask1_ds.reshape(B * T_enc)  # [B*T_enc]

                B, T_enc, D = audio_feat2.shape
                mask2_ds = F.interpolate(mask2.unsqueeze(1).float(), size=T_enc, mode='nearest').squeeze(1).long()
                mask2_flat = mask2_ds.reshape(B * T_enc)

                if self.projection_layer is None:
                    self.projection_layer = nn.Linear(D, 128).to(self.device)

                loss_contrast1 = contrastive_loss_with_mask(audio_feat1_middle, mask1_flat, projection_layer=self.projection_layer)
                loss_contrast2 = contrastive_loss_with_mask(audio_feat2_middle, mask2_flat, projection_layer=self.projection_layer)
                fused_feat1 = self.fusion_module(visual_feat1, audio_feat1)
                fused_feat2 = self.fusion_module(visual_feat2, audio_feat2)
                input_lengths1 = torch.full((fused_feat1.size(0),), fused_feat1.size(1), dtype=torch.long).to(self.device)
                input_lengths2 = torch.full((fused_feat2.size(0),), fused_feat2.size(1), dtype=torch.long).to(self.device)

                log_probs1 = self.decoder1(fused_feat1)
                log_probs2 = self.decoder1(fused_feat2)

                loss1 = self.ctc_loss(log_probs1.transpose(0, 1), text1, input_lengths1, len1)
                loss2 = self.ctc_loss(log_probs2.transpose(0, 1), text2, input_lengths2, len2)

                loss_total = (loss1 + loss2)/2 + lambda_ * (loss_contrast1 + loss_contrast2)/2
                loss_total.backward()
                self.optimizer.step()

                total_loss += loss_total.item()

                if batch_idx % 100 == 0:
                    pred_ids = torch.argmax(log_probs1[0], dim=-1).cpu().tolist()
                    unique_ids = sorted(set(pred_ids))
                    print(f"[Batch {batch_idx}] "
                        f"CTC1: {loss1.item():.4f}, CTC2: {loss2.item():.4f}, "
                        f"Contrast1: {loss_contrast1.item():.4f}, Contrast2: {loss_contrast2.item():.4f}, "
                        f"Total: {loss_total.item():.4f}", flush=True)

                    # 🔍 디코더 출력 shape 확인
                    print(f"[디버그] log_probs1.shape: {log_probs1.shape}", flush=True)

                    # 🔍 softmax 확률 평균 분포 분석
                    probs = F.softmax(log_probs1[0], dim=-1)  # shape: [T, V]
                    mean_probs = probs.mean(dim=0).detach().cpu().numpy()  # 각 토큰 평균 확률
                    top_ids = mean_probs.argsort()[-10:][::-1]  # 상위 10개 토큰
                    top_tokens = [(i, round(mean_probs[i], 4)) for i in top_ids]
                    print(f"[디버그] 상위 10개 토큰 평균 확률: {top_tokens}", flush=True)

                    print(f"[진단] Batch {batch_idx} - 예측 토큰 ID (앞 20개): {pred_ids[:20]}", flush=True)
                    print(f"[진단] 고유 토큰 ID들: {unique_ids}", flush=True)
                    print(f"\n🔎 [Batch {batch_idx}] 예측 결과 확인", flush = True)
                    
                    with torch.no_grad():
                        pred1_ids = torch.argmax(log_probs1, dim=-1)
                        for i in range(min(2, pred1_ids.size(0))):
                            pred_ids1 = self.ctc_decode(pred1_ids[i].cpu().tolist())
                            decoded1 = self.tokenizer.decode(pred_ids1)
                            true1 = self.tokenizer.decode(text1[i][:len1[i]].cpu().tolist())
                            print(f"[화자1 예측] {decoded1}", flush=True)
                            print(f"[화자1 정답] {true1}", flush=True)
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

        all_refs1, all_hyps1 = [], []
        all_refs2, all_hyps2 = [], []
        
        total_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", unit= "batch"):
                lip1 = batch["lip1"].to(self.device).permute(0, 2, 1, 3, 4).contiguous()  # [B, T, C, H, W] → [B, C, T, H, W], C=1, H, W=96
                lip2 = batch["lip2"].to(self.device).permute(0, 2, 1, 3, 4).contiguous()

                text1 = batch["text1"].to(self.device)
                text2 = batch["text2"].to(self.device)

                len1 = batch["text1_lengths"].to(self.device)
                len2 = batch["text2_lengths"].to(self.device)

                audio = batch["audio"].to(self.device)
                mask1 = batch["mask1"].to(self.device)  # [B, T_audio]
                mask2 = batch["mask2"].to(self.device)

                visual_feat1 = self.visual_encoder(lip1)   
                attn_mask1 = (mask1 != 3)             
                audio_feat1, _ = self.audio_encoder(audio, attention_mask=attn_mask1)
                fused_feat1 = self.fusion_module(visual_feat1, audio_feat1)
                log_probs1 = self.decoder1(fused_feat1) #(log_probs1.shape: [B, T, V])
                log_probs1 = F.log_softmax(log_probs1, dim=-1)  # log softmax for CTC

                visual_feat2 = self.visual_encoder(lip2)
                attn_mask2 = (mask2 != 3)
                audio_feat2, _ = self.audio_encoder(audio, attention_mask=attn_mask2)
                fused_feat2 = self.fusion_module(visual_feat2, audio_feat2)
                log_probs2 = self.decoder1(fused_feat2)
                log_probs2 = F.log_softmax(log_probs2, dim=-1)

                input_lengths1 = torch.full(size=(log_probs1.size(0),), fill_value=log_probs1.size(1), dtype=torch.long).to(self.device)
                input_lengths2 = torch.full(size=(log_probs2.size(0),), fill_value=log_probs2.size(1), dtype=torch.long).to(self.device)

                # Loss 계산은 argmax 기반 log_probs 사용
                loss1 = self.ctc_loss(log_probs1.transpose(0, 1), text1, input_lengths1, len1)           
                loss2 = self.ctc_loss(log_probs2.transpose(0, 1), text2, input_lengths2, len2)
                total_loss += (loss1.item() + loss2.item()) / 2


                for i in range(log_probs1.size(0)):
                    pred_ids1 = simple_beam_search(log_probs1[i], beam_width=5, blank=self.tokenizer.blank_id)
                    decoded1 = fast_decode(pred_ids1, self.tokenizer)
                    label_ids1 = text1[i][:len1[i]].cpu().tolist()
                    true_text1 = self.tokenizer.decode(label_ids1)
                    all_refs1.append(true_text1)
                    all_hyps1.append(decoded1)

                    pred_ids2 = simple_beam_search(log_probs2[i], beam_width=5, blank=self.tokenizer.blank_id)
                    decoded2 = fast_decode(pred_ids2, self.tokenizer)
                    label_ids2 = text2[i][:len2[i]].cpu().tolist()
                    true_text2 = self.tokenizer.decode(label_ids2)
                    all_refs2.append(true_text2)
                    all_hyps2.append(decoded2)



        wer1 = wer(all_refs1, all_hyps1)
        wer2 = wer(all_refs2, all_hyps2)
        avg_wer = (wer1 + wer2) / 2
        avg_loss = total_loss / len(dataloader)

        print(f"[Eval] WER1: {wer1:.3f}, WER2: {wer2:.3f}, Avg: {avg_wer:.3f}, Loss: {avg_loss:.4f}")
        return avg_loss, avg_wer
