import torch
import torch.nn as nn
import torch.nn.functional as F


class ProgUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ProgUNet, self).__init__()

        self.enc1_features = 64
        self.enc2_features = 128
        self.enc3_features = 256
        self.enc4_features = 512
        self.bottleneck_features = 512

        self.enc1 = self._block(in_channels, self.enc1_features, name="enc1")
        self.enc2 = self._block(self.enc1_features, self.enc2_features, name="enc2")
        self.enc3 = self._block(self.enc2_features, self.enc3_features, name="enc3")
        self.enc4 = self._block(self.enc3_features, self.enc4_features, name="enc4")

        self.bottleneck = self._block_dilated(
            self.enc4_features, self.bottleneck_features, name="bottleneck"
        )

        self.dec1 = self._block(
            self.bottleneck_features + self.enc4_features,
            self.enc3_features,
            name="dec1",
        )
        self.dec2 = self._block(
            self.enc3_features + self.enc3_features, self.enc2_features, name="dec2"
        )
        self.dec3 = self._block(
            self.enc2_features + self.enc2_features, self.enc1_features, name="dec3"
        )
        self.dec4 = self._block(
            self.enc1_features + self.enc1_features, self.enc1_features, name="dec4"
        )

        self.final = nn.Conv2d(
            self.enc1_features, out_channels, kernel_size=1, padding=0
        )

        self.pool = nn.MaxPool2d(2)

        self.residual_weight = nn.Parameter(torch.tensor(0.2))

    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=3, padding=1, bias=True),
            # nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
        )

    def _block_dilated(self, in_channels, features, name):
        return nn.Sequential(
            nn.ReflectionPad2d(2),  # Padding of 2 for dilation of 2
            nn.Conv2d(in_channels, features, kernel_size=3, dilation=2, bias=True),
            # nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(2),  # Padding of 2 for dilation of 2
            nn.Conv2d(features, features, kernel_size=3, dilation=2, bias=True),
            # nn.BatchNorm2d(features),
            nn.InstanceNorm2d(features),  # trying this out
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, n_targets: int = 1, target_size=None):
        input_image = x

        if target_size is None or (
            isinstance(target_size, torch.Size) and len(target_size) == 0
        ):
            print("target_size is None or empty, using input image size")
            target_size = x.shape

        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        bottleneck_up = F.interpolate(
            bottleneck, size=enc4.shape[2:], mode="bilinear", align_corners=False
        )
        dec1 = self.dec1(torch.cat([bottleneck_up, enc4], dim=1))

        dec1_up = F.interpolate(
            dec1, size=enc3.shape[2:], mode="bilinear", align_corners=False
        )
        dec2 = self.dec2(torch.cat([dec1_up, enc3], dim=1))

        dec2_up = F.interpolate(
            dec2, size=enc2.shape[2:], mode="bilinear", align_corners=False
        )
        dec3 = self.dec3(torch.cat([dec2_up, enc2], dim=1))

        dec3_up = F.interpolate(
            dec3, size=enc1.shape[2:], mode="bilinear", align_corners=False
        )
        dec4 = self.dec4(torch.cat([dec3_up, enc1], dim=1))

        # final_output = self.final(dec_final)
        # final_upscaled = F.interpolate(final_output, size=x.shape[2:], mode='bilinear', align_corners=False)

        outputs = [
            (1 - self.residual_weight) * dec4 + self.residual_weight * input_image
            for _ in range(n_targets)
        ]
        return outputs

    def __str__(self):
        return "ProgUNet"


def load_prog_unet(config):
    checkpoint_path = config["training"]["checkpoint_path"]
    load = config["training"]["load"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProgUNet(in_channels=1, out_channels=1).to(device)
    if load:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    return model
