import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """
    Channel attention module for OCT image processing
    Adapted from CBAM (Convolutional Block Attention Module)
    """

    def __init__(self, channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)


class ResidualBlock(nn.Module):
    def __init__(self, conv_path, shortcut):
        super(ResidualBlock, self).__init__()
        self.conv_path = conv_path
        self.shortcut = shortcut
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = self.shortcut(x)
        x = self.conv_path(x)
        x += residual
        return self.relu(x)


class UNet2(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet2, self).__init__()

        # Encoder (downsampling)
        self.enc1 = self._block(in_channels, 48, name="enc1")
        self.enc2 = self._block(48, 96, name="enc2")
        self.enc3 = self._block(96, 192, name="enc3")
        self.enc4 = self._block(192, 384, name="enc4")

        self.ca_enc1 = ChannelAttention(48)
        self.ca_enc2 = ChannelAttention(96)
        self.ca_enc3 = ChannelAttention(192)
        self.ca_enc4 = ChannelAttention(384)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )

        self.ca_bottleneck = ChannelAttention(384)

        self.dec1 = self._block(384 + 192, 192, name="dec1")
        self.dec2 = self._block(192 + 96, 96, name="dec2")
        self.dec3 = self._block(96 + 48, 48, name="dec3")

        self.ca_dec1 = ChannelAttention(192)
        self.ca_dec2 = ChannelAttention(96)
        self.ca_dec3 = ChannelAttention(48)

        # Final layer
        self.final = nn.Conv2d(48 + in_channels, out_channels, kernel_size=3, padding=1)

        # Max pooling
        self.pool = nn.MaxPool2d(2)

    def _block(self, in_channels, features, name):
        conv_path = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
        )

        shortcut = nn.Sequential()
        if in_channels != features:
            shortcut = nn.Sequential(
                nn.Conv2d(in_channels, features, kernel_size=1, bias=False),
                nn.BatchNorm2d(features),
            )

        return ResidualBlock(conv_path, shortcut)

    def forward(self, x):
        # Save input for skip connection
        inp = x

        # Encoder
        enc1 = self.enc1(x)
        enc1 = self.ca_enc1(enc1)
        enc2 = self.enc2(self.pool(enc1))
        enc2 = self.ca_enc2(enc2)
        enc3 = self.enc3(self.pool(enc2))
        enc3 = self.ca_enc3(enc3)
        enc4 = self.enc4(self.pool(enc3))
        enc4 = self.ca_enc4(enc4)

        bottleneck = self.bottleneck(enc4)
        bottleneck = self.ca_bottleneck(bottleneck)

        # Decoder with skip connections
        dec1 = self.dec1(
            torch.cat(
                [F.interpolate(bottleneck, scale_factor=2, mode="nearest"), enc3], dim=1
            )
        )
        dec1 = self.ca_dec1(dec1)
        dec2 = self.dec2(
            torch.cat(
                [F.interpolate(dec1, scale_factor=2, mode="nearest"), enc2], dim=1
            )
        )
        dec2 = self.ca_dec2(dec2)
        dec3 = self.dec3(
            torch.cat(
                [F.interpolate(dec2, scale_factor=2, mode="nearest"), enc1], dim=1
            )
        )
        dec3 = self.ca_dec3(dec3)

        # Final layer with skip connection to input
        final = torch.cat([dec3, inp], dim=1)

        return self.final(final)

    def __str__(self):
        return "UNet2"


def load_unet(
    config,
):  # checkpoint_path = r'C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\baselines\n2n\checkpoints\noise2noise_final.pth',device=None, load=False
    checkpoint_path = config["training"]["checkpoint_path"]
    load = config["training"]["load"]
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2(in_channels=1, out_channels=1).to(device)
    if load:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    return model
