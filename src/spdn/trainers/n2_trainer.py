from spdn.data import get_paired_loaders
from spdn.utils.config import get_config
from spdn.utils.paths import get_sdoct_path
from spdn.models.unet.unet import UNet
from spdn.models.unet.unet_2 import UNet2
from spdn.models.unet.large_unet import LargeUNetAttention, LargeUNet2, LargeUNet3
from spdn.models.unet.large_unet_good import LargeUNet
from spdn.models.unet.large_unet_attention import LargeUNetAtt
from spdn.models.spdn.spdn_attention import SPDNAttention
from spdn.models.unet.small_unet import SmallUNet
from spdn.models.unet.small_unet_att import SmallUNetAtt
from spdn.models.unet.simple_unet import SimpleUNet
from spdn.models.unet.simple_unet2 import SimpleUNet2

from spdn.schemas.baselines.n2n import train_n2n
from spdn.schemas.baselines.n2v import train_n2v
from spdn.schemas.baselines.n2s import train_n2s, train_n2s_patch

from spdn.schemas.baselines.n2n_patch import train_n2n_patch
from spdn.schemas.baselines.n2v_patch import train_n2v_patch


import os
import torch.optim as optim
import torch

import random
from spdn.utils import load_sdoct_dataset, normalize_image_np


def train_n2(config_path=None, schema=None, ssm=False, override_config=None):
    if config_path is None:
        raise ValueError("Config path must be specified.")

    if schema is None:
        raise ValueError("Model must be specified.")

    config = get_config(config_path, override_config)

    train(config, schema, ssm)


def train(config, method, ssm):
    train_config = config["training"]
    if method is None:
        raise ValueError("Method must be specified either in config or function.")

    print(f"Training method: {method}")

    n_patients = train_config["n_patients"]
    n_images_per_patient = train_config["n_images_per_patient"]
    batch_size = train_config["batch_size"]
    start = train_config["start_patient"] if train_config["start_patient"] else 1
    ablation = train_config["ablation"].format(
        n=n_patients, n_images=n_images_per_patient
    )

    train_loader, val_loader = get_paired_loaders(
        start, n_patients, n_images_per_patient, batch_size
    )
    print(f"Train loader size: {len(train_loader.dataset)}")
    sample = next(iter(train_loader))[0].shape
    print(f"Sample shape: {sample}")
    print(f"Validation loader size: {len(val_loader.dataset)}")

    baselines_checkpoint_path = train_config["baselines_checkpoint_path"] + ablation

    model = train_config["model"]

    checkpoint_path = baselines_checkpoint_path + rf"/{method}_"

    if not os.path.exists(baselines_checkpoint_path):
        os.makedirs(baselines_checkpoint_path)

    if config["speckle_module"]["use"] is True or ssm:
        checkpoint_path = checkpoint_path + rf"{model}_ssm"
        print("Checkpoint path: ", checkpoint_path)
    else:
        checkpoint_path = checkpoint_path + rf"{model}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if train_config["model"] == "UNet":
        model = UNet(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "UNet2":
        model = UNet2(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "LargeUNet":
        model = LargeUNet(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "LargeUNetAttention":
        model = LargeUNetAttention(in_channels=1, out_channels=1).to(device)
    elif (
        train_config["model"] == "LargeUNetNoAttention"
        or train_config["model"] == "LargeUNet2"
    ):
        model = LargeUNet2(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "LargeUNet3":
        model = LargeUNet3(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "SmallUNet":
        model = SmallUNet(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "SmallUNetAtt":
        model = SmallUNetAtt(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "LargeUNetAtt":
        model = LargeUNetAtt(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "SimpleUNet":
        model = SimpleUNet(in_channels=1, out_channels=1).to(device)
    elif train_config["model"] == "SimpleUNet2":
        model = SimpleUNet2(in_channels=1, out_channels=1).to(device)
    else:
        raise ValueError("Model not found")

    sdoct_path = get_sdoct_path()
    dataset = load_sdoct_dataset(sdoct_path)

    import cv2
    import numpy as np

    sample = random.choice(list(dataset.keys()))
    raw_image = dataset[sample]["raw"][0][0]
    raw_image = raw_image.cpu().numpy()
    print(f"Raw image shape: {raw_image.shape}")
    if raw_image.max() > 1.0:
        raw_image = raw_image / 255.0
    print(f"Raw image max: {raw_image.max()}, min: {raw_image.min()}")
    resized = cv2.resize(raw_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    print(f"Resized image shape: {resized.shape}")
    resized = normalize_image_np(resized)
    print(f"Raw image max: {raw_image.max()}, min: {raw_image.min()}")
    resized = resized[:, :, np.newaxis]
    raw_image = torch.from_numpy(
        resized.transpose(2, 0, 1)
    ).float()  # Transpose channels
    raw_image = raw_image.unsqueeze(0)
    raw_image = raw_image.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=train_config["learning_rate"], weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=5, factor=0.5
    )

    visualise = train_config["visualise"]

    alpha = 1
    starting_epoch = 0
    best_val_loss = float("inf")
    best_metrics_score = float("-inf")

    save = train_config["save"]

    # Ablations
    adaptive_loss = train_config["adaptive_loss"]

    # save config as text
    config_checkpoint_path = baselines_checkpoint_path + "/config_text/"
    print("Config checkpoint path: ", config_checkpoint_path)
    if not os.path.exists(config_checkpoint_path):
        os.makedirs(config_checkpoint_path)
    with open(config_checkpoint_path + "config.txt", "w") as f:
        f.write(f"Training method: {method}\n")
        f.write(f"Model: {model}\n")
        f.write(f"Batch size: {batch_size}\n")
        f.write(f"Learning rate: {train_config['learning_rate']}\n")
        f.write(f"Epochs: {train_config['epochs']}\n")
        f.write(f"Start patient: {start}\n")
        f.write(f"Ablation: {ablation}\n")
        f.write(f"Number of patients: {n_patients}\n")
        f.write(f"Number of images per patient: {n_images_per_patient}\n")

    if config["speckle_module"]["use"] is True or ssm:
        speckle_module = SPDNAttention(input_channels=1, feature_dim=32).to(device)
        try:
            print("Loading ssm model from checkpoint...")
            ssm_checkpoint_path = train_config["ssm_checkpoint_path"]
            print(f"SSM checkpoint path: {ssm_checkpoint_path}")
            ssm_checkpoint = torch.load(ssm_checkpoint_path, map_location=device)
            speckle_module.load_state_dict(ssm_checkpoint["model_state_dict"])
            speckle_module.to(device)
            alpha = config["speckle_module"]["alpha"]
            print("SPDN model loaded successfully")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")
            raise e
    else:
        speckle_module = None

    if train_config["load"]:
        try:
            last_checkpoint = torch.load(
                checkpoint_path + "_patched_last_checkpoint.pth", map_location=device
            )
            best_checkpoint = torch.load(
                checkpoint_path + "_patched_best_checkpoint.pth", map_location=device
            )
            print("Loading model from checkpoint...")
            model.load_state_dict(last_checkpoint["model_state_dict"])
            optimizer.load_state_dict(last_checkpoint["optimizer_state_dict"])
            print("Model loaded successfully")
            print(
                f"Epoch: {last_checkpoint['epoch']}, Loss: {best_checkpoint['best_val_loss']}"
            )
            best_metrics_score = best_checkpoint["metrics_score"]
            starting_epoch = last_checkpoint["epoch"]
            best_val_loss = best_checkpoint["val_loss"]
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Starting training from scratch.")

    print("Alpha: ", alpha)

    if train_config["train"]:
        patch = train_config["patch"]
        if method == "n2n":
            if patch:
                train_n2n_patch(
                    model,
                    train_loader,
                    val_loader,
                    optimizer=optimizer,
                    criterion=train_config["criterion"],
                    starting_epoch=starting_epoch,
                    epochs=train_config["epochs"],
                    batch_size=train_config["batch_size"],
                    lr=train_config["learning_rate"],
                    best_val_loss=best_val_loss,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    visualise=visualise,
                    speckle_module=speckle_module,
                    alpha=alpha,
                    save=save,
                    scheduler=scheduler,
                    best_metrics_score=best_metrics_score,
                    train_config=train_config,
                    sample=raw_image,
                    patch_size=train_config["patch_size"],
                    stride=train_config["stride"],
                    adaptive_loss=adaptive_loss,
                )
            else:
                model = train_n2n(
                    model,
                    train_loader,
                    val_loader,
                    optimizer=optimizer,
                    criterion=train_config["criterion"],
                    starting_epoch=starting_epoch,
                    epochs=train_config["epochs"],
                    batch_size=train_config["batch_size"],
                    lr=train_config["learning_rate"],
                    best_val_loss=best_val_loss,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    visualise=visualise,
                    speckle_module=speckle_module,
                    alpha=alpha,
                    save=save,
                    scheduler=scheduler,
                    best_metrics_score=best_metrics_score,
                    train_config=train_config,
                )

        elif method == "n2v":
            if patch:
                print("Training n2v with patch")

                model = train_n2v_patch(
                    model,
                    train_loader,
                    val_loader,
                    optimizer=optimizer,
                    criterion=train_config["criterion"],
                    starting_epoch=starting_epoch,
                    epochs=train_config["epochs"],
                    batch_size=train_config["batch_size"],
                    lr=train_config["learning_rate"],
                    best_val_loss=best_val_loss,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    visualise=visualise,
                    speckle_module=speckle_module,
                    alpha=alpha,
                    save=save,
                    method=method,
                    octa_criterion=False,
                    threshold=train_config["threshold"],
                    mask_ratio=train_config["mask_ratio"],
                    best_metrics_score=best_metrics_score,
                    scheduler=scheduler,
                    train_config=train_config,
                    sample=raw_image,
                    patch_size=train_config["patch_size"],
                    stride=train_config["stride"],
                    patience_count=train_config["patience"],
                    adaptive_loss=adaptive_loss,
                )
            else:
                model = train_n2v(
                    model,
                    train_loader,
                    val_loader,
                    optimizer=optimizer,
                    criterion=train_config["criterion"],
                    starting_epoch=starting_epoch,
                    epochs=train_config["epochs"],
                    batch_size=train_config["batch_size"],
                    lr=train_config["learning_rate"],
                    best_val_loss=best_val_loss,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    visualise=visualise,
                    speckle_module=speckle_module,
                    alpha=alpha,
                    save=save,
                    method=method,
                    octa_criterion=False,
                    threshold=train_config["threshold"],
                    mask_ratio=train_config["mask_ratio"],
                    best_metrics_score=best_metrics_score,
                    scheduler=scheduler,
                )
        elif method == "n2s":
            if patch:
                model = train_n2s_patch(
                    model,
                    train_loader,
                    val_loader,
                    optimizer=optimizer,
                    criterion=train_config["criterion"],
                    starting_epoch=starting_epoch,
                    epochs=train_config["epochs"],
                    batch_size=train_config["batch_size"],
                    lr=train_config["learning_rate"],
                    best_val_loss=best_val_loss,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    visualise=visualise,
                    speckle_module=speckle_module,
                    alpha=alpha,
                    save=save,
                    scheduler=scheduler,
                    sample=raw_image,
                    train_config=train_config,
                    best_metrics_score=best_metrics_score,
                    patch_size=train_config["patch_size"],
                    stride=train_config["stride"],
                    n_partitions=train_config["n_partitions"],
                    adaptive_loss=adaptive_loss,
                )

            else:
                model = train_n2s(
                    model,
                    train_loader,
                    val_loader,
                    optimizer=optimizer,
                    criterion=train_config["criterion"],
                    starting_epoch=starting_epoch,
                    epochs=train_config["epochs"],
                    batch_size=train_config["batch_size"],
                    lr=train_config["learning_rate"],
                    best_val_loss=best_val_loss,
                    checkpoint_path=checkpoint_path,
                    device=device,
                    visualise=visualise,
                    speckle_module=speckle_module,
                    alpha=alpha,
                    save=save,
                )

    return model
