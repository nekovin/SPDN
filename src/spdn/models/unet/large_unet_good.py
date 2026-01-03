import torch.nn as nn

from spdn.models.components.components import DoubleConv, Down, Up, OutConv


class LargeUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(LargeUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        # Same structure but larger filters
        self.inc = DoubleConv(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)  # Added to reach 512 channels

        self.bottleneck = DoubleConv(512, 512)

        # Matching upsampling path with skip connections
        self.up1 = Up(1024, 256, bilinear)  # 512 + 512 = 1024 input channels
        self.up2 = Up(512, 128, bilinear)  # 256 + 256 = 512 input channels
        self.up3 = Up(256, 64, bilinear)  # 128 + 128 = 256 input channels
        self.up4 = Up(128, 32, bilinear)  # 64 + 64 = 128 input channels
        self.up5 = Up(64, 32, bilinear)  # 32 + 32 = 64 input channels

        self.outc = OutConv(32, out_channels)

        self.dropout = nn.Dropout2d(0.1)

    def forward(self, x):
        # Encoder path
        x1 = self.inc(x)
        x1_drop = self.dropout(
            x1
        )  # Apply dropout but keep x1 intact for skip connection

        x2 = self.down1(x1_drop)
        x2_drop = self.dropout(x2)  # Apply dropout but keep x2 intact

        x3 = self.down2(x2_drop)
        x3_drop = self.dropout(x3)

        x4 = self.down3(x3_drop)
        x4_drop = self.dropout(x4)

        x5 = self.down4(x4_drop)

        # No dropout before bottleneck
        x6 = self.bottleneck(x5)

        # Decoder path with all skip connections using original (non-dropped) features
        x = self.up1(x6, x5)
        x = self.dropout(x)

        x = self.up2(x, x4)
        x = self.dropout(x)

        x = self.up3(x, x3)
        x = self.dropout(x)

        x = self.up4(x, x2)
        x = self.dropout(x)

        x = self.up5(x, x1)
        # No dropout before final output

        x = self.outc(x)

        return x

    def __str__(self):
        return "LargeUNet"
