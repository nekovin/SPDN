from spdn.trainers import train_spdn
from spdn.losses.ssm_loss import custom_loss
from spdn.utils.config import get_config
import os
import torch


def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def train_small():
    config_path = os.environ.get("SPDN_CONFIG_PATH")

    override_dict = {
        "training": {
            "ablation": "spdn_architecture_depth/small",
            "model": "SPDNNoAttention",
        }
    }

    config = get_config(config_path, override_dict)

    loss_name = config["training"]["criterion"]

    if loss_name == "mse":
        loss_fn = mse_loss
    else:
        loss_fn = custom_loss

    print(config["training"])

    train_spdn(config["training"], loss_fn, loss_name)


if __name__ == "__main__":
    train_small()
