import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len).unsqueeze(1).float()  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [D/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # even
        pe[:, 1::2] = torch.cos(position * div_term)  # odd
        self.pe = pe.unsqueeze(0)  # [1, T, D]

    def forward(self, x):
        """
        x: [B, T, D]
        returns: [B, T, D] with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :].to(x.device)


class RivaConformerAudioEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=512, hidden_dim=512, num_layers=4, freeze=False):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.positional_encoding = PositionalEncoding(d_model=hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

        self.output_dim = output_dim

        if freeze:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):  # x: [B, T, 80]
        x = self.input_proj(x)                     # [B, T, hidden_dim]
        x = self.positional_encoding(x)            # [B, T, hidden_dim]
        x = self.encoder(x)                        # [B, T, hidden_dim]
        x = self.output_proj(x)                    # [B, T, output_dim]
        return x
