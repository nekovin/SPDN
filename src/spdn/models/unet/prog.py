import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """double conv with optional residual connection"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Down(nn.Module):
    """downscaling with maxpool then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """upscaling then double conv"""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(
            x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2]
        )
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ProgressiveFusionDynamicUNet(nn.Module):
    """progressive fusion u-net with dynamic outputs for OCT denoising"""

    def __init__(
        self, n_channels: int = 1, base_features: int = 32, use_fusion: bool = True
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.use_fusion = use_fusion

        # Encoder pathway
        self.inc = DoubleConv(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)

        # Fusion attention mechanism (optional)
        if use_fusion:
            self.fusion_attention = nn.Sequential(
                nn.Conv2d(base_features * 8, base_features * 8, kernel_size=1),
                nn.Sigmoid(),
            )

        # Decoder pathway
        self.up1 = Up(base_features * 8, base_features * 4)
        self.up2 = Up(base_features * 4, base_features * 2)
        self.up3 = Up(base_features * 2, base_features)

        # Output layer
        self.outc = nn.Conv2d(base_features, n_channels, kernel_size=1)

        # Learnable parameters
        self.residual_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, x: torch.Tensor, n_targets: int, target_size: torch.Size):
        # Store input for residual connection
        input_image = x

        # Encoder pathway with skip connections
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Apply fusion attention if enabled
        if self.use_fusion:
            x4 = x4 * self.fusion_attention(x4)

        # Decoder pathway with skip connections
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Dynamically generate outputs for each target
        outputs = [
            (1 - self.residual_weight)
            * F.interpolate(
                self.outc(x),
                size=target_size[-2:],
                mode="bilinear",
                align_corners=False,
            )
            + self.residual_weight * input_image
            for _ in range(n_targets)
        ]
        return outputs


def create_progressive_fusion_dynamic_unet(
    base_features: int = 32, use_fusion: bool = True
) -> ProgressiveFusionDynamicUNet:
    """Factory function to create progressive fusion dynamic u-net"""
    return ProgressiveFusionDynamicUNet(
        n_channels=1, base_features=base_features, use_fusion=use_fusion
    )
