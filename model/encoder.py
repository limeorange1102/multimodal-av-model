import torch
import torch.nn as nn
import torchvision.models as models
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import EncDecCTCModel
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)


class VisualEncoder(nn.Module):
    def __init__(self, pretrained_path=None, hidden_dim=256, lstm_layers=2, bidirectional=True):
        super().__init__()
        self.resnet = models.resnet34(weights=None)
        self.resnet.fc = nn.Identity()

        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            try:
                self.resnet.load_state_dict(state_dict, strict=False)
                print(f"✅ VisualEncoder weights loaded from {pretrained_path}")
            except Exception as e:
                print(f"❌ Failed to load weights: {e}")

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.resnet(x)
        feats = feats.view(B, T, -1)
        output, _ = self.rnn(feats)
        return output


class RivaConformerAudioEncoder(nn.Module):
    def __init__(self, pretrained_name='stt_ko_conformer_ctc_large', freeze=True):
        super().__init__()
        self.model = EncDecCTCModel.from_pretrained(model_name=pretrained_name)
        if freeze:
            self.model.freeze()
        self.output_dim = self.model.encoder._feat_out  # Extracted feature dim

    def forward(self, x, lengths=None):
        # Expecting [B, T, F] log-mel spectrogram input
        # lengths is required for masking in some NeMo models
        if lengths is None:
            lengths = torch.full((x.size(0),), x.size(1), dtype=torch.long, device=x.device)
        features = self.model.encoder(x, lengths=lengths)
        return features
