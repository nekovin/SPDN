import torch.nn as nn
import torch.nn.functional as F
import torch


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None):
        super(NonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        # Define transformations for query, key, and value
        self.g = nn.Conv2d(
            in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.theta = nn.Conv2d(
            in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0
        )
        self.phi = nn.Conv2d(
            in_channels, self.inter_channels, kernel_size=1, stride=1, padding=0
        )

        # Output transformation
        self.W = nn.Conv2d(
            self.inter_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        batch_size = x.size(0)

        # g(x): [B, C, H, W] -> [B, C//2, H, W]
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)  # [B, C//2, H*W]
        g_x = g_x.permute(0, 2, 1)  # [B, H*W, C//2]

        # theta(x): [B, C, H, W] -> [B, C//2, H, W]
        theta_x = self.theta(x).view(
            batch_size, self.inter_channels, -1
        )  # [B, C//2, H*W]
        theta_x = theta_x.permute(0, 2, 1)  # [B, H*W, C//2]

        # phi(x): [B, C, H, W] -> [B, C//2, H, W]
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)  # [B, C//2, H*W]

        # Calculate attention map
        f = torch.matmul(theta_x, phi_x)  # [B, H*W, H*W]
        f_div_C = F.softmax(f, dim=-1)

        # Weighted sum using the attention map
        y = torch.matmul(f_div_C, g_x)  # [B, H*W, C//2]
        y = y.permute(0, 2, 1).contiguous()  # [B, C//2, H*W]
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])  # [B, C//2, H, W]

        # Final transformation and residual connection
        W_y = self.W(y)  # [B, C, H, W]
        W_y = self.bn(W_y)  # Apply batch normalization
        z = W_y + x  # Residual connection

        return z


# Generator Network (with Nonlocal blocks)
class Generator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, features=64):
        super(Generator, self).__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )  # [B, 64, H/2, W/2]

        self.enc2 = nn.Sequential(
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
        )  # [B, 128, H/4, W/4]

        self.enc3 = nn.Sequential(
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
        )  # [B, 256, H/8, W/8]

        self.enc4 = nn.Sequential(
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )  # [B, 512, H/16, W/16]

        # NonLocal blocks
        self.nonlocal1 = NonLocalBlock(features * 8)
        self.nonlocal2 = NonLocalBlock(features * 4)

        # Decoder
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 8, features * 4, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(inplace=True),
        )  # [B, 256, H/8, W/8]

        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 8, features * 2, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(inplace=True),
        )  # [B, 128, H/4, W/4]

        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 4, features, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True),
        )  # [B, 64, H/2, W/2]

        self.dec4 = nn.Sequential(
            nn.ConvTranspose2d(
                features * 2, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),  # Output range [-1, 1]
        )  # [B, 1, H, W]

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Apply NonLocal block at bottleneck
        e4 = self.nonlocal1(e4)

        # Decoder with skip connections
        d1 = self.dec1(e4)
        d1 = self.nonlocal2(d1)
        d1 = torch.cat([d1, e3], dim=1)

        d2 = self.dec2(d1)
        d2 = torch.cat([d2, e2], dim=1)

        d3 = self.dec3(d2)
        d3 = torch.cat([d3, e1], dim=1)

        d4 = self.dec4(d3)

        return d4


# PatchGAN Discriminator
class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=64):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # Layer 1: No batch norm
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 2
            nn.Conv2d(features, features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 3
            nn.Conv2d(features * 2, features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # Layer 4
            nn.Conv2d(features * 4, features * 8, kernel_size=4, stride=1, padding=1),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # Output layer
            nn.Conv2d(features * 8, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, x):
        return self.model(x)
