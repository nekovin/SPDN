from spdn.utils.eval_utils.evaluate import evaluate
import torch
from spdn.models import UNet, UNet2, LargeUNetAttention, LargeUNet2, LargeUNet3
from spdn.models.unet.large_unet_attention import LargeUNetAtt
from spdn.models.unet.large_unet_good import LargeUNet
from spdn.models.unet.small_unet import SmallUNet
from spdn.models.unet.small_unet_att import SmallUNetAtt
from spdn.models.unet.blind_large_unet_attention import BlindLargeUNetAtt
from spdn.models.unet.simple_unet import SimpleUNet


def load_model(config, verbose=False, last=False, best=False):
    use_speckle = config["speckle_module"]["use"]
    eval_config = config["training"]
    base_checkpoint_path = eval_config["baselines_checkpoint_path"]
    method = eval_config["method"]
    model = eval_config["model"]
    print(base_checkpoint_path)
    if use_speckle:
        best_loss = config["speckle_module"]["best"]
        if best_loss:
            checkpoint_path = (
                base_checkpoint_path
                + rf"{method}_{model}_ssm_patched_best_checkpoint.pth"
            )
        else:
            checkpoint_path = (
                base_checkpoint_path
                + rf"{method}_{model}_ssm_patched_last_checkpoint.pth"
            )
    else:
        checkpoint_path = (
            base_checkpoint_path + rf"{method}_{model}_patched_best_checkpoint.pth"
        )

    if best:
        checkpoint_path = (
            base_checkpoint_path
            + rf"{method}_{model}_patched_best_metrics_checkpoint.pth"
        )

    device = eval_config["device"]

    if model == "UNet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif model == "UNet2":
        model = UNet2(in_channels=1, out_channels=1).to(device)
    elif model == "LargeUNet":
        model = LargeUNet(in_channels=1, out_channels=1).to(device)
    elif model == "LargeUNetAttention":
        model = LargeUNetAttention(in_channels=1, out_channels=1).to(device)
    elif model == "LargeUNetNoAttention" or model == "LargeUNet2":
        model = LargeUNet2(in_channels=1, out_channels=1).to(device)
    elif model == "LargeUNet3":
        model = LargeUNet3(in_channels=1, out_channels=1).to(device)
    elif model == "SmallUNet":
        model = SmallUNet(in_channels=1, out_channels=1).to(device)
    elif model == "SmallUNetAtt":
        model = SmallUNetAtt().to(device)
    elif model == "LargeUNetAtt":
        model = LargeUNetAtt(in_channels=1, out_channels=1).to(device)
    elif model == "BlindLargeUNetAtt":
        model = BlindLargeUNetAtt(in_channels=1, out_channels=1).to(device)
    elif model == "SimpleUNet":
        model = SimpleUNet(in_channels=1, out_channels=1).to(device)
        print("Using SimpleUNet model")
    elif model == "SimpleUNet2":
        from spdn.models.unet.simple_unet2 import SimpleUNet2

        model = SimpleUNet2(in_channels=1, out_channels=1).to(device)
    else:
        raise ValueError(f"Model {model} not supported")

    checkpoint = load_checkpoint(config, last)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model loaded from {checkpoint_path}")
    print(checkpoint["model_state_dict"])

    if verbose:
        print(f"Loading {model} model...")
        print(f"Checkpoint path: {checkpoint_path}")
        for key, value in checkpoint.items():
            # if key != 'model_state_dict' or key != 'optimizer_state_dict':
            # print(f"{key}: {value}")
            if key == "epoch":
                print(f"Epoch: {value}")
            if key == "best_val_loss":
                print(f"Loss: {value}")
        print("Model loaded successfully")
    return model, checkpoint


def _load_checkpoint(config, last=False):
    eval_config = config["training"]
    base_checkpoint_path = eval_config["baselines_checkpoint_path"]
    ablation = eval_config["ablation"].format(
        n=config["training"]["n_patients"],
        n_images=config["training"]["n_images_per_patient"],
    )
    method = eval_config["method"]
    model = eval_config["model"]
    if config["speckle_module"]["use"]:
        best = config["speckle_module"]["best"]
        if best:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_ssm_patched_best_checkpoint.pth"
            )
        else:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_ssm_patched_last_checkpoint.pth"
            )
        if last:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_ssm_patched_last_checkpoint.pth"
            )
    else:
        if last:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_patched_last_checkpoint.pth"
            )
        else:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_patched_best_checkpoint.pth"
            )
    print(f"Checkpoint path: {checkpoint_path}")

    # print(f"Checkpoint path: {checkpoint_path}")

    device = eval_config["device"]

    checkpoint = torch.load(checkpoint_path, map_location=device)

    return checkpoint


def load_checkpoint(config, last=False, best=False):
    eval_config = config["training"]
    base_checkpoint_path = eval_config["baselines_checkpoint_path"]
    ablation = eval_config["ablation"].format(
        n=config["training"]["n_patients"],
        n_images=config["training"]["n_images_per_patient"],
    )
    method = eval_config["method"]
    model = eval_config["model"]

    if best:
        if config["speckle_module"]["use"]:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_ssm_patched_best_metrics_checkpoint.pth"
            )
        else:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_patched_best_metrics_checkpoint.pth"
            )
    elif config["speckle_module"]["use"]:
        best_loss = config["speckle_module"]["best"]
        if best_loss and not last:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_ssm_patched_best_checkpoint.pth"
            )
        else:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_ssm_patched_last_checkpoint.pth"
            )
    else:
        if last:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_patched_last_checkpoint.pth"
            )
        else:
            checkpoint_path = (
                base_checkpoint_path
                + ablation
                + rf"/{method}_{model}_patched_best_checkpoint.pth"
            )

    print(f"Checkpoint path: {checkpoint_path}")

    device = eval_config["device"]

    checkpoint = torch.load(checkpoint_path, map_location=device)

    return checkpoint


def evaluate_baseline(image, reference, method, model):
    metrics, denoised = evaluate(image, reference, model, method)

    metrics["model"] = str(model)

    return metrics, denoised


def evaluate_ssm_constraint(image, reference, method, model):
    metrics, denoised = evaluate(image, reference, model, method)

    metrics["model"] = str(model)

    return metrics, denoised
