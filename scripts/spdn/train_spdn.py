from spdn.trainers import train_spdn
from spdn.losses.ssm_loss import custom_loss
from spdn.utils.config import get_config
import os
import torch
import torch.nn.functional as F


def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)


def dice_loss(y_true, y_pred, smooth=1e-6):
    """Dice - Standard for medical segmentation"""
    intersection = (y_true * y_pred).sum()
    return 1 - (2.0 * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)


def dice_bce_loss(y_true, y_pred, dice_weight=0.5):
    """Dice + BCE - Most common for OCT segmentation"""
    dice = dice_loss(y_true, y_pred)
    bce = F.binary_cross_entropy_with_logits(y_pred, y_true)
    return dice_weight * dice + (1 - dice_weight) * bce


def main():
    config_path = os.environ.get("SPDN_CONFIG_PATH")

    config = get_config(config_path)

    loss_name = config["training"]["criterion"]

    if loss_name == "mse":
        loss_fn = mse_loss
    elif loss_name == "dice":
        loss_fn = dice_loss
    elif loss_name == "dice_bce":
        loss_fn = dice_bce_loss
    else:
        print(f"Using custom loss function: {loss_name}")
        loss_fn = custom_loss

    print(config["training"])

    train_spdn(config["training"], loss_fn, loss_name)


if __name__ == "__main__":
    main()
