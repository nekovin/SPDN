import torch.nn as nn

from spdn.models.components.components import DoubleConv, Down, Up, OutConv


class SimpleUNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(SimpleUNet2, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        self.dropout = nn.Dropout2d(0.1)
        self.bottleneck = DoubleConv(256, 256)

        self.up1 = Up(256 + 128, 128, bilinear)
        self.up2 = Up(128 + 64, 64, bilinear)

        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x1 = self.dropout(x1)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)

        x4 = self.bottleneck(x3)

        x = self.up1(x4, x2)
        x = self.dropout(x)
        x = self.up2(x, x1)
        x = self.dropout(x)
        x = self.outc(x)

        return x

    def __str__(self):
        return "SimpleUNet"
