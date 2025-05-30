import torch
import torch.nn as nn
import torchvision.models as models
from transformers import Wav2Vec2Model

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type='prelu'):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.PReLU(planes) if relu_type == 'prelu' else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class ResNet(nn.Module):
    def __init__(self, block, layers, relu_type='prelu'):
        super().__init__()
        self.inplanes = 64
        self.layer1 = self._make_layer(block, 64, layers[0], relu_type=relu_type)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, relu_type=relu_type)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, relu_type=relu_type)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, relu_type=relu_type)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, blocks, stride=1, relu_type='prelu'):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )
        layers = [block(self.inplanes, planes, stride, downsample, relu_type)]
        self.inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type=relu_type))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.avgpool(x).flatten(1)

# -------------------------------
# 📌 영상 인코더: VisualEncoder
# -------------------------------
class VisualEncoder(nn.Module):
    def __init__(self, relu_type='prelu'):
        super().__init__()
        self.frontend3D = nn.Sequential(
            nn.Conv3d(1, 64, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.PReLU(64) if relu_type == 'prelu' else nn.ReLU(True),
            nn.MaxPool3d((1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))
        )
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type=relu_type)
        self.output_dim = 512

    def forward(self, x):
        B, C, T, H, W = x.shape
        x = self.frontend3D(x)  # (B, 64, T', H', W')
        x = x.transpose(1, 2).contiguous().view(B * x.shape[2], 64, x.shape[3], x.shape[4])
        x = self.trunk(x)       # (B*T', 512)
        x = x.view(B, -1, 512)  # (B, T', 512)
        return x

# -------------------------------
# 🎧 음성 인코더: HuggingFaceAudioEncoder
# -------------------------------
class AudioEncoder(nn.Module):
    def __init__(self, model_name="kresnik/wav2vec2-large-xlsr-korean", freeze=True):
        super().__init__()
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.output_dim = self.model.config.hidden_size
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x, attention_mask=None):
        # x: [B, T], attention_mask: [B, T]
        if attention_mask is not None:
            attention_mask = attention_mask.long()  # Convert to long tensor if needed
        output = self.model(input_values=x, attention_mask=attention_mask, return_dict=True)
        return output.last_hidden_state  # [B, T', output_dim = 1024]
