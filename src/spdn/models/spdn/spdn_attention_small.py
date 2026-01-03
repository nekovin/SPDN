import torch
import torch.nn as nn

from .components import ChannelAttention, SpatialAttention


def normalize_image_torch(img):
    # Get min and max values
    min_val = torch.min(img)
    max_val = torch.max(img)

    # Normalize to [0,1] range
    if max_val > min_val:
        normalized = (img - min_val) / (max_val - min_val)
    else:
        normalized = torch.zeros_like(img)

    return normalized


class SmallSPDNAttention(nn.Module):
    """
    Smaller version of SPDN U-Net with attention mechanisms
    """

    def __init__(self, input_channels=1, feature_dim=16, depth=3, block_depth=2):
        """
        Initialize the Small SPDN Attention Module

        Args:
            input_channels: Number of input image channels (default: 1 for grayscale OCT)
            feature_dim: Initial dimension of feature maps (reduced from 32 to 16)
            depth: Depth of the U-Net (reduced from 5 to 3)
            block_depth: Number of convolution layers in each block (reduced from 3 to 2)
        """
        super(SmallSPDNAttention, self).__init__()

        self.encoder_blocks = nn.ModuleList()
        self.encoder_attentions = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attentions = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.depth = depth
        self.input_channels = input_channels
        self.feature_dim = feature_dim
        self.block_depth = block_depth

        # Encoder path with attention
        in_channels = input_channels
        for i in range(depth):
            out_channels = feature_dim * (2 ** min(i, 2))  # Cap at 2 instead of 3
            encoder_block = []

            # First conv in the block
            encoder_block.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            )
            encoder_block.append(nn.BatchNorm2d(out_channels))
            encoder_block.append(nn.ReLU(inplace=True))

            # Additional conv layers in each block
            for _ in range(block_depth - 1):
                encoder_block.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                )
                encoder_block.append(nn.BatchNorm2d(out_channels))
                encoder_block.append(nn.ReLU(inplace=True))

            self.encoder_blocks.append(nn.Sequential(*encoder_block))

            # Add attention modules after each encoder block
            self.encoder_attentions.append(
                nn.Sequential(ChannelAttention(out_channels), SpatialAttention())
            )

            in_channels = out_channels

        # Smaller bottleneck
        bottleneck_channels = feature_dim * (2 ** min(depth, 2))
        bottleneck = []
        for _ in range(block_depth):  # Reduced bottleneck depth
            bottleneck.append(
                nn.Conv2d(in_channels, bottleneck_channels, kernel_size=3, padding=1)
            )
            bottleneck.append(nn.BatchNorm2d(bottleneck_channels))
            bottleneck.append(nn.ReLU(inplace=True))
            in_channels = bottleneck_channels

        self.bottleneck = nn.Sequential(*bottleneck)

        # Bottleneck attention
        self.bottleneck_attention = nn.Sequential(
            ChannelAttention(bottleneck_channels), SpatialAttention()
        )

        # Decoder path with attention
        in_channels = bottleneck_channels
        for i in range(depth):
            out_channels = feature_dim * (2 ** min(depth - i - 1, 2))
            decoder_block = []

            # First conv after the skip connection
            decoder_block.append(
                nn.Conv2d(
                    in_channels + out_channels, out_channels, kernel_size=3, padding=1
                )
            )
            decoder_block.append(nn.BatchNorm2d(out_channels))
            decoder_block.append(nn.ReLU(inplace=True))

            # Additional conv layers in each block
            for _ in range(block_depth - 1):
                decoder_block.append(
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                )
                decoder_block.append(nn.BatchNorm2d(out_channels))
                decoder_block.append(nn.ReLU(inplace=True))

            self.decoder_blocks.append(nn.Sequential(*decoder_block))

            # Add attention modules after each decoder block
            self.decoder_attentions.append(
                nn.Sequential(ChannelAttention(out_channels), SpatialAttention())
            )

            in_channels = out_channels

        # Simplified dilated convolutions (fewer dilations)
        self.dilation_block = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
        )

        # Final attention after dilation
        self.final_attention = nn.Sequential(
            ChannelAttention(feature_dim),
            nn.Dropout(0.2),
            SpatialAttention(),
            nn.Dropout(0.2),
        )

        # Simplified output layers
        self.flow_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        self.noise_branch = nn.Sequential(
            nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(feature_dim, input_channels, kernel_size=1),
            nn.Sigmoid(),
        )

        # Upsampling layer
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def __repr__(self):
        return f"SmallSPDNAttention(input_channels={self.input_channels}, feature_dim={self.feature_dim}, depth={self.depth}, block_depth={self.block_depth})"

    def forward(self, x):
        encoder_features = []

        # Encoder path with attention
        for i in range(self.depth):
            x = self.encoder_blocks[i](x)
            x = self.encoder_attentions[i](x)
            encoder_features.append(x)
            if i < self.depth - 1:
                x = self.pool(x)

        # Bottleneck with attention
        x = self.bottleneck(x)
        x = self.bottleneck_attention(x)

        # Decoder path with attention
        for i in range(self.depth):
            x = self.up(x)
            encoder_feature = encoder_features[self.depth - i - 1]
            if x.size() != encoder_feature.size():
                x = nn.functional.interpolate(
                    x,
                    size=encoder_feature.size()[2:],
                    mode="bilinear",
                    align_corners=True,
                )
            x = torch.cat([x, encoder_feature], dim=1)
            x = self.decoder_blocks[i](x)
            x = self.decoder_attentions[i](x)

        # Dilated convolutions and final attention
        x = self.dilation_block(x)
        x = self.final_attention(x)

        # Output branches
        flow_component = self.flow_branch(x)
        noise_component = self.noise_branch(x)

        flow_component = normalize_image_torch(flow_component)

        return {"flow_component": flow_component, "noise_component": noise_component}


def get_spdn_model_simple(checkpoint_path):
    print("Loading simplified SSM model...")

    model = SmallSPDNAttention()

    checkpoint = None

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])

    return model, checkpoint
