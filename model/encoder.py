import torch
import torch.nn as nn
import torchvision.models as models
import nemo.collections.asr as nemo_asr
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
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
    def __init__(self, input_dim=80, output_dim=512, hidden_dim=512, num_layers=4, freeze=False):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=8, dim_feedforward=2048, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        self.output_dim = output_dim

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.input_proj(x)
        x = self.positional_encoding(x)
        x = self.encoder(x)
        x = self.output_proj(x)
        return x
