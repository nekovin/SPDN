import torch
import torch.nn as nn


from spdn.models.unet.large_unet_old import LargeUNet


class ProgLargeUNet(LargeUNet):
    def __init__(self, in_channels=1, out_channels=1, bilinear=True):
        super(ProgLargeUNet, self).__init__(in_channels, out_channels, bilinear)

        self.residual_weight = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, n_targets=1, target_size=None):
        # Store input image for residual connection
        input_image = x

        # Handle target_size similar to ProgUNet
        if target_size is None or (
            isinstance(target_size, torch.Size) and len(target_size) == 0
        ):
            print("target_size is None or empty, using input image size")
            target_size = x.shape

        # Original LargeUNet forward path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x6 = self.bottleneck(x5)

        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)

        x = self.outc(x)

        outputs = [
            (1 - self.residual_weight) * x + self.residual_weight * input_image
            for _ in range(n_targets)
        ]

        return outputs


def load_prog_unet(config):
    checkpoint_path = config["training"]["checkpoint_path"]
    load = config["training"]["load"]
    config.get("model", {}).get("use_instance_norm", False)
    bilinear = config.get("model", {}).get("bilinear", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProgLargeUNet(in_channels=1, out_channels=1, bilinear=bilinear).to(device)

    if load:
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.to(device)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e

    return model
