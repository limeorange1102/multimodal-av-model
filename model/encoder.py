import torch
import torch.nn as nn
import torch.nn as nn
import torchvision.models as models
import nemo.collections.asr as nemo_asr

class VisualEncoder(nn.Module):
    def __init__(self, pretrained_path=None, hidden_dim=256, lstm_layers=2, bidirectional=True):
        super().__init__()

        # 📌 resnet34 사용 (pt 파일이 resnet34 기반임)
        self.resnet = models.resnet34(weights=None)
        self.resnet.fc = nn.Identity()  # feature만 뽑기

        # RNN 인코더
        self.rnn = nn.LSTM(
            input_size=512,  # resnet34의 마지막 feature dim
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional
        )

        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        # pretrained weight 불러오기
        if pretrained_path is not None:
            state_dict = torch.load(pretrained_path, map_location='cpu')
            try:
                self.resnet.load_state_dict(state_dict, strict=False)
                print(f"✅ VisualEncoder weights loaded from {pretrained_path}")
            except Exception as e:
                print(f"❌ Failed to load weights: {e}")

    def forward(self, x):
        # x shape: (B, T, C, H, W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.resnet(x)  # (B*T, 512)
        feats = feats.view(B, T, -1)  # (B, T, 512)
        output, _ = self.rnn(feats)
        return output


class RivaConformerAudioEncoder(nn.Module):
    def __init__(self, input_dim=80, output_dim=512, hidden_dim=512, num_layers=4, freeze=False):
        """
        log-Mel 입력을 받아 Conformer-style feature를 생성하는 임시 Audio Encoder
        (NeMo 모델을 직접 쓰는 게 아니라 log-Mel 기반 커스텀 encoder 구성)

        Args:
            input_dim: 입력 특성 차원 (log-Mel feature 수 = 80)
            output_dim: 최종 출력 차원 (모델 fusion에 쓰일 dim)
            hidden_dim: Transformer 내부 hidden dim
            num_layers: Transformer layer 수
            freeze: 파라미터 학습 여부
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

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
        """
        x: [B, T, 80]  log-Mel spectrogram
        return: [B, T, output_dim]
        """
        x = self.input_proj(x)           # [B, T, hidden_dim]
        x = self.encoder(x)              # [B, T, hidden_dim]
        x = self.output_proj(x)          # [B, T, output_dim]
        return x