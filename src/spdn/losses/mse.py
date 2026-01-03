import torch


def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)
