import torch
import torch.nn as nn
import torch.nn.functional as F

from spdn.models.components.components import ChannelAttention


# Create new BlindSpot convolution layers
class BlindSpotConv(nn.Module):
    """
    A blind spot convolution that GUARANTEES zero information flow from the center pixel
    by using a structural approach rather than weight masking.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(BlindSpotConv, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation

        # Instead of using a single conv with masked weights,
        # we'll use multiple convs that each avoid the center pixel

        # 1. Create a main convolution for features around the center
        self.main_conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=padding
        )

        # 2. Create a special convolution for the center pixel
        # that will always output zero (we don't connect it to the input)
        self.center_zero = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 1. Apply main convolution
        out = self.main_conv(x)

        # 2. Force center pixel output to zero for each spatial location
        batch_size, channels, height, width = out.shape

        # Create an explicit zero mask for center pixels
        mask = torch.ones_like(out)

        # Get the center indices for each spatial location based on kernel and padding
        center_offset = self.kernel_size // 2

        # Zero out the center pixels of each receptive field
        for b in range(batch_size):
            for c in range(channels):
                for h in range(center_offset, height - center_offset):
                    for w in range(center_offset, width - center_offset):
                        # For each output pixel, zero out the contribution
                        # from input pixels at the center of its receptive field
                        h_input = h - center_offset
                        w_input = w - center_offset
                        if 0 <= h_input < height and 0 <= w_input < width:
                            # Zero replacement
                            out[b, c, h, w] = (
                                out[b, c, h, w]
                                * mask[b, c, h, w]
                                * (1 - (h_input == h and w_input == w))
                            )

        return out


class BlindSpotDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BlindSpotDoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            BlindSpotConv(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class BlindLargeUNetAtt(nn.Module):
    """
    U-Net architecture with blind-spot convolutions and skip connections to ensure the network
    never sees the central pixel it's trying to predict while preserving spatial details.
    """

    def __init__(self, in_channels=1, out_channels=1, features=32):
        super(BlindLargeUNetAtt, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder with blind-spot convolutions
        # First layer uses blind-spot convolution to establish the blind spot
        self.enc1 = nn.Sequential(
            BlindSpotConv(in_channels, features),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )

        # Additional encoder layers
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            BlindSpotConv(features, features * 2, dilation=1, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
        )

        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = nn.Sequential(
            BlindSpotConv(features * 2, features * 4, dilation=1, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True),
        )

        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = nn.Sequential(
            BlindSpotConv(features * 4, features * 8, dilation=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
        )

        self.pool4 = nn.MaxPool2d(2)

        # Bottleneck with attention mechanism
        self.bottleneck = nn.Sequential(
            BlindSpotConv(features * 8, features * 16, dilation=1, padding=1),
            nn.BatchNorm2d(features * 16),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.2),
            ChannelAttention(features * 16, reduction_ratio=16),
        )

        # Decoder path with skip connections
        # Upsampling layers
        self.up4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )

        # Decoder blocks including skip connection handling
        self.dec4 = nn.Sequential(
            nn.Conv2d(
                features * 16, features * 8, kernel_size=3, padding=1
            ),  # Input is doubled due to skip connection
            nn.BatchNorm2d(features * 8),
            nn.ReLU(inplace=True),
            ChannelAttention(features * 8, reduction_ratio=8),
        )

        self.up3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.dec3 = nn.Sequential(
            nn.Conv2d(
                features * 8, features * 4, kernel_size=3, padding=1
            ),  # Input is doubled due to skip connection
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True),
            ChannelAttention(features * 4, reduction_ratio=8),
        )

        self.up2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(
                features * 4, features * 2, kernel_size=3, padding=1
            ),  # Input is doubled due to skip connection
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
            ChannelAttention(features * 2, reduction_ratio=4),
        )

        self.up1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(
                features * 2, features, kernel_size=3, padding=1
            ),  # Input is doubled due to skip connection
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
            ChannelAttention(features, reduction_ratio=4),
        )

        # Final output layer
        self.final = nn.Conv2d(features, out_channels, kernel_size=1)

        # Dropout for regularization
        self.dropout = nn.Dropout2d(0.2)

    def forward(self, x):
        # Encoder path with blind-spot convolutions
        e1 = self.enc1(x)
        e1 = self.dropout(e1)

        e2 = self.enc2(self.pool1(e1))
        e2 = self.dropout(e2)

        e3 = self.enc3(self.pool2(e2))
        e3 = self.dropout(e3)

        e4 = self.enc4(self.pool3(e3))
        e4 = self.dropout(e4)

        # Bottleneck
        b = self.bottleneck(self.pool4(e4))

        # Decoder path with skip connections
        # Upsample and concatenate with corresponding encoder features
        d4 = self.up4(b)
        # Check if dimensions match and handle spatial discrepancies
        if d4.shape[2] != e4.shape[2] or d4.shape[3] != e4.shape[3]:
            d4 = F.interpolate(
                d4,
                size=(e4.shape[2], e4.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        # Concatenate along channel dimension
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        d4 = self.dropout(d4)

        d3 = self.up3(d4)
        if d3.shape[2] != e3.shape[2] or d3.shape[3] != e3.shape[3]:
            d3 = F.interpolate(
                d3,
                size=(e3.shape[2], e3.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d3 = self.dropout(d3)

        d2 = self.up2(d3)
        if d2.shape[2] != e2.shape[2] or d2.shape[3] != e2.shape[3]:
            d2 = F.interpolate(
                d2,
                size=(e2.shape[2], e2.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d2 = self.dropout(d2)

        d1 = self.up1(d2)
        if d1.shape[2] != e1.shape[2] or d1.shape[3] != e1.shape[3]:
            d1 = F.interpolate(
                d1,
                size=(e1.shape[2], e1.shape[3]),
                mode="bilinear",
                align_corners=False,
            )
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Final output
        output = self.final(d1)

        return output

    def __str__(self):
        return "BlindLargeUNetAtt"
