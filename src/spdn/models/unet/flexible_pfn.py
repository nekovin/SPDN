import torch
import torch.nn as nn
import torch.nn.functional as F


class FlexibleProgressiveUNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        base_features=64,
        bilinear=True,
        max_levels=5,
    ):
        super(FlexibleProgressiveUNet, self).__init__()
        self.max_levels = max_levels

        # Shared encoder
        self.inc = self._double_conv(in_channels, base_features)
        self.down1 = self._down(base_features, base_features * 2)
        self.down2 = self._down(base_features * 2, base_features * 4)
        self.down3 = self._down(base_features * 4, base_features * 8)

        # Bottleneck
        factor = 2 if bilinear else 1
        self.down4 = self._down(base_features * 8, base_features * 16 // factor)
        self.bottleneck = self._double_conv(
            base_features * 16 // factor, base_features * 16 // factor
        )

        # Create decoders for each level
        self.decoders = nn.ModuleList()
        self.level_inc = nn.ModuleList()
        self.outc = nn.ModuleList()

        # Initialize the first level differently (doesn't take previous output)
        first_decoder = nn.ModuleDict(
            {
                "up1": self._up(
                    base_features * 16 // factor, base_features * 8 // factor, bilinear
                ),
                "up2": self._up(
                    base_features * 8, base_features * 4 // factor, bilinear
                ),
                "up3": self._up(
                    base_features * 4, base_features * 2 // factor, bilinear
                ),
                "up4": self._up(base_features * 2, base_features, bilinear),
            }
        )
        self.decoders.append(first_decoder)
        self.outc.append(nn.Conv2d(base_features, out_channels, kernel_size=1))

        # Initialize subsequent levels that take previous output as input
        for _ in range(1, max_levels):
            # Feature extractor for previous level output
            self.level_inc.append(self._double_conv(out_channels, base_features))

            # Decoder stack for this level
            decoder = nn.ModuleDict(
                {
                    "up1": self._up(
                        base_features * 16 // factor,
                        base_features * 8 // factor,
                        bilinear,
                    ),
                    "up2": self._up(
                        base_features * 8, base_features * 4 // factor, bilinear
                    ),
                    "up3": self._up(
                        base_features * 4, base_features * 2 // factor, bilinear
                    ),
                    "up4": self._up(base_features * 2, base_features, bilinear),
                }
            )
            self.decoders.append(decoder)

            # Output layer for this level (takes concatenated features)
            self.outc.append(nn.Conv2d(base_features * 2, out_channels, kernel_size=1))

        # Learnable residual weights for each level
        self.residual_weights = nn.ParameterList(
            [
                nn.Parameter(
                    torch.tensor(0.3 - 0.05 * i)
                )  # Decreasing residual influence
                for i in range(max_levels)
            ]
        )

    def _double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def _down(self, in_channels, out_channels):
        return nn.Sequential(
            nn.MaxPool2d(2), self._double_conv(in_channels, out_channels)
        )

    def _up(self, in_channels, out_channels, bilinear=True):
        if bilinear:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                self._double_conv(in_channels, out_channels),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
                self._double_conv(out_channels, out_channels),
            )

    def forward(self, x, n_targets=None, target_size=None):
        # Default to using all available levels if n_targets not specified
        if n_targets is None:
            n_targets = self.max_levels
        else:
            n_targets = min(n_targets, self.max_levels)

        # Handle target size
        if target_size is None or (
            isinstance(target_size, torch.Size) and len(target_size) == 0
        ):
            target_size = x.shape

        input_image = x

        # Shared encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5_bottleneck = self.bottleneck(x5)

        # Process each level sequentially
        outputs = []
        prev_output = None

        for level in range(n_targets):
            decoder = self.decoders[level]

            # First level decoder path
            x = decoder["up1"](x5_bottleneck)
            x = self._handle_skip(x, x4)
            x = decoder["up2"](x)
            x = self._handle_skip(x, x3)
            x = decoder["up3"](x)
            x = self._handle_skip(x, x2)
            x = decoder["up4"](x)
            x = self._handle_skip(x, x1)

            level_features = x

            # For first level, just use the features directly
            if level == 0:
                output = self.outc[level](level_features)
                residual_source = input_image
            else:
                # For subsequent levels, incorporate features from previous level
                prev_features = self.level_inc[level - 1](prev_output)
                # Concatenate features from this level with processed features from previous level
                combined_features = torch.cat([level_features, prev_features], dim=1)
                output = self.outc[level](combined_features)
                residual_source = prev_output

            # Apply residual connection with appropriate weight
            output = (
                1 - self.residual_weights[level]
            ) * output + self.residual_weights[level] * residual_source

            outputs.append(output)
            prev_output = output

        return outputs

    def _handle_skip(self, x, skip_features):
        # Handle the case where sizes don't match
        if x.shape[2:] != skip_features.shape[2:]:
            x = F.interpolate(
                x, size=skip_features.shape[2:], mode="bilinear", align_corners=True
            )

        return torch.cat([x, skip_features], dim=1)
