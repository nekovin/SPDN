import torch.nn as nn

from spdn.models.components.components import (
    DoubleConv,
    Down,
    Up,
    OutConv,
    ChannelAttention,
)


class LargeUNetAtt(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(LargeUNetAtt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Encoder path
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)

        # Add attention at bottleneck (most critical for feature selection)
        self.bottleneck = nn.Sequential(
            DoubleConv(512, 512), ChannelAttention(512, reduction_ratio=16)
        )

        # Add attention to each decoder stage to enhance detail preservation
        self.ca1 = ChannelAttention(256, reduction_ratio=8)
        self.ca2 = ChannelAttention(128, reduction_ratio=8)
        self.ca3 = ChannelAttention(64, reduction_ratio=4)
        self.ca4 = ChannelAttention(32, reduction_ratio=4)

        # Decoder path
        self.up1 = Up(1024, 256, bilinear)
        self.up2 = Up(512, 128, bilinear)
        self.up3 = Up(256, 64, bilinear)
        self.up4 = Up(128, 32, bilinear)
        self.up5 = Up(64, 32, bilinear)

        self.outc = OutConv(32, out_channels)
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x1_drop = self.dropout(x1)

        x2 = self.down1(x1_drop)
        x2_drop = self.dropout(x2)

        x3 = self.down2(x2_drop)
        x3_drop = self.dropout(x3)

        x4 = self.down3(x3_drop)
        x4_drop = self.dropout(x4)

        x5 = self.down4(x4_drop)
        x5_drop = self.dropout(x5)

        # Bottleneck with attention
        x6 = self.bottleneck(x5_drop)
        x6_drop = self.dropout(x6)

        # Decoder path with attention on each level
        x = self.up1(x6_drop, x5)
        x = self.ca1(x)  # Apply attention after upsampling and concatenation
        x = self.dropout(x)

        x = self.up2(x, x4)
        x = self.ca2(x)
        x = self.dropout(x)

        x = self.up3(x, x3)
        x = self.ca3(x)
        x = self.dropout(x)

        x = self.up4(x, x2)
        x = self.ca4(x)
        x = self.dropout(x)

        x = self.up5(x, x1)
        x = self.outc(x)

        return x

    def __str__(self):
        return "LargeUNetAtt"
