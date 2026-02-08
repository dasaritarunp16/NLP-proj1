import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        return self.block(x)


class BallTrackerNet(nn.Module):
    def __init__(self, in_channels=9, out_channels=256):
        super().__init__()
        # Encoder (VGG-16 style)
        self.encoder = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.MaxPool2d(2, 2),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            nn.Upsample(scale_factor=2),
            ConvBlock(256, 128),
            ConvBlock(128, 128),
            nn.Upsample(scale_factor=2),
            ConvBlock(128, 64),
            ConvBlock(64, 64),
            nn.Conv2d(64, out_channels, 1),
            nn.Softmax(dim=1),
        )
        self._init_weights()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.05, 0.05)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
