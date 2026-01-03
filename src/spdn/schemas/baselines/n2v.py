import time
import torch
from IPython.display import clear_output

import sys

from torch.utils.data import Dataset, DataLoader

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

from contextlib import nullcontext

from tqdm import tqdm
from spdn.utils.eval_utils.visualise import plot_images


def create_blind_spot_input_fast(
    image, mask
):  # This creates an artificial situation: your network learns to reconstruct pixels from surrounding context, but the masking pattern (black dots) doesn't match the actual noise distribution in OCT images.
    blind_input = image.clone()
    # noise = torch.randn_like(image) * image.std() + image.mean()
    # blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    # blind_input = torch.where(mask.bool(), torch.zeros_like(image), image)
    blind_input[mask.bool()] = 0
    return blind_input


def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    # noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    return blind_input


def create_blind_spot_input_with_realistic_noise(image, mask):
    blind_input = image.clone()

    # Parameters for OCT-like noise (speckle)
    mean_level = image.mean()
    std_level = image.std() * 0.9  # Adjust based on your OCT noise level

    # Generate noise with speckle-like characteristics
    noise = torch.randn_like(image) * std_level + mean_level

    # Apply mask
    blind_input[mask.bool()] = noise[mask.bool()]
    return blind_input


def compute_octa(oct1, oct2):
    numerator = (oct1 - oct2) ** 2
    denominator = oct1**2 + oct2**2

    epsilon = 1e-10
    octa = numerator / (denominator + epsilon)

    return octa


def threshold_octa(octa, oct, threshold):
    oct = oct.cpu().numpy()
    octa = octa.detach().cpu().numpy()
    background_mask = oct > np.percentile(oct, threshold)  # Bottom 20% of OCT values

    if np.sum(background_mask) > 0:  # Ensure we have background pixels
        background_mean = np.mean(oct[background_mask])
        background_std = np.std(oct[background_mask])

        threshold = background_mean + 2 * background_std
    else:
        # If no background pixels, use a fixed threshold
        print("No background pixels found, using fixed threshold")
        threshold = np.percentile(oct, 1)

    signal_mask = oct > threshold

    thresholded_octa = octa * signal_mask

    return thresholded_octa


def threshold_octa_torch(
    octa: torch.Tensor, oct: torch.Tensor, threshold_percent: float
):
    """
    Apply a threshold to the OCTA image based on the statistics of the OCT image.

    Args:
        octa (torch.Tensor): The OCTA image tensor (should require gradients).
        oct (torch.Tensor): The corresponding OCT image tensor.
        threshold_percent (float): Percentile value (0-100) for initial thresholding.

    Returns:
        torch.Tensor: The thresholded OCTA image.
    """
    # Compute the percentile value using torch.quantile (percentile_percent/100.0)
    t_percentile = torch.quantile(oct, threshold_percent / 100.0)

    # Create the background mask for OCT pixels
    background_mask = oct > t_percentile

    # If we have background pixels, compute the mean and std of those pixels.
    if background_mask.sum() > 0:
        background_mean = torch.mean(oct[background_mask])
        background_std = torch.std(oct[background_mask])
        threshold_val = background_mean + 2 * background_std
    else:
        # If no background pixels, default to a fixed low percentile (e.g., 1st percentile)
        threshold_val = torch.quantile(oct, 0.01)

    # Create a signal mask based on the computed threshold_val.
    signal_mask = oct > threshold_val

    # Multiply the OCTA image with the signal mask; cast the boolean mask to float.
    thresholded_octa = octa * signal_mask.float()

    return thresholded_octa


def enhanced_differentiable_threshold_octa_torch(
    octa, oct, threshold_percentile=80, smoothness=3.0, enhancement_factor=1.2
):
    """
    Enhanced differentiable thresholding for OCTA images with vessel structure preservation.

    Args:
        octa: The OCTA image tensor (computed from OCT differences)
        oct: The OCT tensor used for thresholding reference
        threshold_percentile: Percentile (0-100) to determine foreground/background
        smoothness: Controls the transition sharpness in the sigmoid (lower = smoother)
        enhancement_factor: Factor to enhance vessel structures (higher enhances vessels)

    Returns:
        Thresholded OCTA image with preserved vessel structures
    """
    # Get foreground/background threshold using percentile
    sorted_values, _ = torch.sort(oct.reshape(-1))
    threshold_idx = int(sorted_values.shape[0] * threshold_percentile / 100)
    threshold_value = sorted_values[threshold_idx]

    # Use pixels ABOVE threshold as foreground (vessels + tissue)
    # This is more appropriate for OCT where vessels appear bright
    foreground_mask = oct > threshold_value

    # Compute statistics for better thresholding
    if torch.sum(foreground_mask) > 0:
        foreground_pixels = oct[foreground_mask]
        foreground_mean = torch.mean(foreground_pixels)
        foreground_std = torch.std(foreground_pixels)

        # Set threshold at mean - std to include most vessel structures
        signal_threshold = foreground_mean - foreground_std
    else:
        # Fallback: use high percentile threshold
        fallback_idx = int(sorted_values.shape[0] * 0.95)  # 95th percentile
        signal_threshold = sorted_values[fallback_idx]

    # Create gradient-aware mask using softplus instead of sigmoid
    # Softplus provides a smoother transition and better preserves fine structures
    soft_mask = torch.log(1 + torch.exp(smoothness * (oct - signal_threshold)))

    # Normalize soft mask to [0,1] range
    if soft_mask.max() > soft_mask.min():
        soft_mask = (soft_mask - soft_mask.min()) / (soft_mask.max() - soft_mask.min())

    # Apply vessel enhancement by emphasizing high OCTA values
    enhanced_octa = octa * (1.0 + (octa * enhancement_factor))

    # Apply soft mask to the enhanced OCTA
    thresholded_octa = enhanced_octa * soft_mask

    # Final normalization to ensure output is well-scaled
    if thresholded_octa.max() > 0:
        thresholded_octa = thresholded_octa / thresholded_octa.max()

    return thresholded_octa


from matplotlib.colors import NoNorm


def visualise_n2v(
    raw1,
    oct1,
    oct2,
    output1,
    output2,
    normalize_output1,
    octa_from_outputs,
    thresholded_octa,
    stage1_output,
):
    """
    Visualize the N2V process with mask overlay
    """
    clear_output(wait=True)

    # Create figure with specific axis layout
    fig = plt.figure(figsize=(24, 6))

    # Create a grid with specific positions
    grid = plt.GridSpec(3, 5, figure=fig)

    # Create only the axes you need
    ax1 = fig.add_subplot(grid[0, 0])
    axraw = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0])

    ax3 = fig.add_subplot(grid[0, 1])
    ax4 = fig.add_subplot(grid[2, 1])
    axnorm = fig.add_subplot(grid[1, 1])

    ax5 = fig.add_subplot(grid[1, 2])
    ax6 = fig.add_subplot(grid[1, 3])
    ax7 = fig.add_subplot(grid[1, 4])
    ax8 = fig.add_subplot(grid[0, 3])
    ax9 = fig.add_subplot(grid[0, 4])

    # Plot on individual axes
    ax1.imshow(oct1.squeeze(), cmap="gray", norm=NoNorm())
    ax1.axis("off")
    ax1.set_title("OCT1")

    axraw.imshow(raw1.squeeze(), cmap="gray")
    axraw.axis("off")
    axraw.set_title("Raw1 (norm)")

    ax2.imshow(oct2.squeeze(), cmap="gray", norm=NoNorm())
    ax2.axis("off")
    ax2.set_title("OCT2")

    ax3.imshow(output1.squeeze(), cmap="gray")
    ax3.axis("off")
    ax3.set_title("Output1")

    ax4.imshow(output2.squeeze(), cmap="gray", norm=NoNorm())
    ax4.axis("off")
    ax4.set_title("Output2")

    axnorm.imshow(normalize_output1.squeeze(), cmap="gray")
    axnorm.axis("off")
    axnorm.set_title("Normalized Output1")

    ax5.imshow(octa_from_outputs.squeeze(), cmap="gray", norm=NoNorm())
    ax5.axis("off")
    ax5.set_title("Octa from outputs")

    ax6.imshow(thresholded_octa.squeeze(), cmap="gray")
    ax6.axis("off")
    ax6.set_title("Thresholded OCTA (normalised)")

    ax7.imshow(stage1_output.squeeze(), cmap="gray")
    ax7.axis("off")
    ax7.set_title("Stage 1 Output (normalised)")

    ax8.imshow(stage1_output.squeeze(), cmap="gray", norm=NoNorm())
    ax8.axis("off")
    ax8.set_title("Stage 1 Output")

    ax9.imshow(thresholded_octa.squeeze(), cmap="gray", norm=NoNorm())
    ax9.axis("off")
    ax9.set_title("Thresholded OCTA")

    plt.tight_layout()
    plt.show()


def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


import torch

sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")


def normalize_image_torch(t_img: torch.Tensor) -> torch.Tensor:
    """
    Normalise the input image tensor.

    For pixels above 0.01, computes the min and max (foreground) and scales
    those pixels to the [0, 1] range. Pixels below 0.01 are forced to 0.

    Args:
        t_img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: The normalized image tensor.
    """
    if t_img.max() > 0:
        # Create mask for non-background (foreground) pixels.
        foreground_mask = t_img > 0.01

        # Check if any foreground pixel is found.
        if torch.any(foreground_mask):
            fg_values = t_img[foreground_mask]
            fg_min = fg_values.min()
            fg_max = fg_values.max()

            # Normalize only if there is a valid range.
            if fg_max > fg_min:
                # Use torch.where to selectively update foreground pixels.
                t_img = torch.where(
                    foreground_mask, (t_img - fg_min) / (fg_max - fg_min), t_img
                )

        # Force background (pixels < 0.01) to be 0
        t_img = torch.where(t_img < 0.01, torch.zeros_like(t_img), t_img)
    return t_img


def freeze():
    """
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    """
    pass


def load(model, optimizer, save_path, scratch, device):
    print(f"Received save_path: {save_path}")
    print(f"Current working directory: {os.getcwd()}")

    checkpoints_dir = os.path.dirname(save_path)
    if not os.path.exists(checkpoints_dir) and not scratch:
        os.makedirs(checkpoints_dir, exist_ok=True)

    if scratch:
        try:
            checkpoint = torch.load(
                r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v\checkpoints\stage1.pth"
            )
            print(checkpoint.keys())
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            old_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            print(
                f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch + 1}"
            )
            return model, optimizer, old_epoch, history
        except:
            raise ValueError(
                "Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path."
            )
    else:
        try:
            checkpoint = torch.load(save_path)
            print(checkpoint.keys())
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            old_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            print(
                f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch + 1}"
            )
            return model, optimizer, old_epoch, history
        except:
            raise ValueError(
                "Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path."
            )


def check_performance(
    val_loss,
    best_val_loss,
    model,
    optimizer,
    epoch,
    save_path,
    history,
    old_epoch,
    avg_train_loss,
):
    if val_loss < best_val_loss:
        print(f"Saving model with val loss: {val_loss:.6f} from epoch {epoch + 1}")
        best_val_loss = val_loss
        try:
            torch.save(
                {
                    "epoch": epoch + old_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "history": history,
                },
                save_path,
            )

        except:
            print("Err")
        print(
            f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss:.6f}"
        )
        print("-" * 50)


def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    # noise = torch.randn_like(image) * image.std() + image.mean()
    # blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    # blind_input = torch.where(mask.bool(), torch.zeros_like(image), image)
    blind_input[mask.bool()] = 0
    return blind_input


def create_blind_spot_input_fast(image, mask):
    blind_input = image.clone()
    # noise = torch.randn_like(image) * image.std() + image.mean()
    blind_input = torch.where(mask > 0, torch.zeros_like(image), blind_input)
    return blind_input


def create_blind_spot_input_with_realistic_noise(image, mask):
    blind_input = image.clone()

    # Parameters for OCT-like noise (speckle)
    mean_level = image.mean()
    std_level = image.std() * 0.9  # Adjust based on your OCT noise level

    # Generate noise with speckle-like characteristics
    noise = torch.randn_like(image) * std_level + mean_level

    # Apply mask
    blind_input[mask.bool()] = noise[mask.bool()]
    return blind_input


def compute_octa(oct1, oct2):
    numerator = (oct1 - oct2) ** 2
    denominator = oct1**2 + oct2**2

    epsilon = 1e-10
    octa = numerator / (denominator + epsilon)

    return octa


import torch


def threshold_octa(octa, oct, threshold):
    oct = oct.cpu().numpy()
    octa = octa.detach().cpu().numpy()
    background_mask = oct > np.percentile(oct, threshold)  # Bottom 20% of OCT values

    if np.sum(background_mask) > 0:  # Ensure we have background pixels
        background_mean = np.mean(oct[background_mask])
        background_std = np.std(oct[background_mask])

        threshold = background_mean + 2 * background_std
    else:
        # If no background pixels, use a fixed threshold
        print("No background pixels found, using fixed threshold")
        threshold = np.percentile(oct, 1)

    signal_mask = oct > threshold

    thresholded_octa = octa * signal_mask

    return thresholded_octa


def threshold_octa_torch(
    octa: torch.Tensor, oct: torch.Tensor, threshold_percent: float
):
    """
    Apply a threshold to the OCTA image based on the statistics of the OCT image.

    Args:
        octa (torch.Tensor): The OCTA image tensor (should require gradients).
        oct (torch.Tensor): The corresponding OCT image tensor.
        threshold_percent (float): Percentile value (0-100) for initial thresholding.

    Returns:
        torch.Tensor: The thresholded OCTA image.
    """
    # Compute the percentile value using torch.quantile (percentile_percent/100.0)
    t_percentile = torch.quantile(oct, threshold_percent / 100.0)

    # Create the background mask for OCT pixels
    background_mask = oct > t_percentile

    # If we have background pixels, compute the mean and std of those pixels.
    if background_mask.sum() > 0:
        background_mean = torch.mean(oct[background_mask])
        background_std = torch.std(oct[background_mask])
        threshold_val = background_mean + 2 * background_std
    else:
        # If no background pixels, default to a fixed low percentile (e.g., 1st percentile)
        threshold_val = torch.quantile(oct, 0.01)

    # Create a signal mask based on the computed threshold_val.
    signal_mask = oct > threshold_val

    # Multiply the OCTA image with the signal mask; cast the boolean mask to float.
    thresholded_octa = octa * signal_mask.float()

    return thresholded_octa


def enhanced_differentiable_threshold_octa_torch(
    octa, oct, threshold_percentile=80, smoothness=3.0, enhancement_factor=1.2
):
    """
    Enhanced differentiable thresholding for OCTA images with vessel structure preservation.

    Args:
        octa: The OCTA image tensor (computed from OCT differences)
        oct: The OCT tensor used for thresholding reference
        threshold_percentile: Percentile (0-100) to determine foreground/background
        smoothness: Controls the transition sharpness in the sigmoid (lower = smoother)
        enhancement_factor: Factor to enhance vessel structures (higher enhances vessels)

    Returns:
        Thresholded OCTA image with preserved vessel structures
    """
    # Get foreground/background threshold using percentile
    sorted_values, _ = torch.sort(oct.reshape(-1))
    threshold_idx = int(sorted_values.shape[0] * threshold_percentile / 100)
    threshold_value = sorted_values[threshold_idx]

    # Use pixels ABOVE threshold as foreground (vessels + tissue)
    # This is more appropriate for OCT where vessels appear bright
    foreground_mask = oct > threshold_value

    # Compute statistics for better thresholding
    if torch.sum(foreground_mask) > 0:
        foreground_pixels = oct[foreground_mask]
        foreground_mean = torch.mean(foreground_pixels)
        foreground_std = torch.std(foreground_pixels)

        # Set threshold at mean - std to include most vessel structures
        signal_threshold = foreground_mean - foreground_std
    else:
        # Fallback: use high percentile threshold
        fallback_idx = int(sorted_values.shape[0] * 0.95)  # 95th percentile
        signal_threshold = sorted_values[fallback_idx]

    # Create gradient-aware mask using softplus instead of sigmoid
    # Softplus provides a smoother transition and better preserves fine structures
    soft_mask = torch.log(1 + torch.exp(smoothness * (oct - signal_threshold)))

    # Normalize soft mask to [0,1] range
    if soft_mask.max() > soft_mask.min():
        soft_mask = (soft_mask - soft_mask.min()) / (soft_mask.max() - soft_mask.min())

    # Apply vessel enhancement by emphasizing high OCTA values
    enhanced_octa = octa * (1.0 + (octa * enhancement_factor))

    # Apply soft mask to the enhanced OCTA
    thresholded_octa = enhanced_octa * soft_mask

    # Final normalization to ensure output is well-scaled
    if thresholded_octa.max() > 0:
        thresholded_octa = thresholded_octa / thresholded_octa.max()

    return thresholded_octa


def visualise_n2v(
    raw1,
    oct1,
    oct2,
    output1,
    output2,
    normalize_output1,
    octa_from_outputs,
    thresholded_octa,
    stage1_output,
):
    """
    Visualize the N2V process with mask overlay
    """
    clear_output(wait=True)

    # Create figure with specific axis layout
    fig = plt.figure(figsize=(24, 6))

    # Create a grid with specific positions
    grid = plt.GridSpec(3, 5, figure=fig)

    # Create only the axes you need
    ax1 = fig.add_subplot(grid[0, 0])
    axraw = fig.add_subplot(grid[1, 0])
    ax2 = fig.add_subplot(grid[2, 0])

    ax3 = fig.add_subplot(grid[0, 1])
    ax4 = fig.add_subplot(grid[2, 1])
    axnorm = fig.add_subplot(grid[1, 1])

    ax5 = fig.add_subplot(grid[1, 2])
    ax6 = fig.add_subplot(grid[1, 3])
    ax7 = fig.add_subplot(grid[1, 4])
    ax8 = fig.add_subplot(grid[0, 3])
    ax9 = fig.add_subplot(grid[0, 4])

    # Plot on individual axes
    ax1.imshow(oct1.squeeze(), cmap="gray", norm=NoNorm())
    ax1.axis("off")
    ax1.set_title("OCT1")

    axraw.imshow(raw1.squeeze(), cmap="gray")
    axraw.axis("off")
    axraw.set_title("Raw1 (norm)")

    ax2.imshow(oct2.squeeze(), cmap="gray", norm=NoNorm())
    ax2.axis("off")
    ax2.set_title("OCT2")

    ax3.imshow(output1.squeeze(), cmap="gray")
    ax3.axis("off")
    ax3.set_title("Output1")

    ax4.imshow(output2.squeeze(), cmap="gray", norm=NoNorm())
    ax4.axis("off")
    ax4.set_title("Output2")

    axnorm.imshow(normalize_output1.squeeze(), cmap="gray")
    axnorm.axis("off")
    axnorm.set_title("Normalized Output1")

    ax5.imshow(octa_from_outputs.squeeze(), cmap="gray", norm=NoNorm())
    ax5.axis("off")
    ax5.set_title("Octa from outputs")

    ax6.imshow(thresholded_octa.squeeze(), cmap="gray")
    ax6.axis("off")
    ax6.set_title("Thresholded OCTA (normalised)")

    ax7.imshow(stage1_output.squeeze(), cmap="gray")
    ax7.axis("off")
    ax7.set_title("Stage 1 Output (normalised)")

    ax8.imshow(stage1_output.squeeze(), cmap="gray", norm=NoNorm())
    ax8.axis("off")
    ax8.set_title("Stage 1 Output")

    ax9.imshow(thresholded_octa.squeeze(), cmap="gray", norm=NoNorm())
    ax9.axis("off")
    ax9.set_title("Thresholded OCTA")

    plt.tight_layout()
    plt.show()


def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(10, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


import torch
import sys

sys.path.append(r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v")


def normalize_image_torch(t_img: torch.Tensor) -> torch.Tensor:
    """
    Normalise the input image tensor.

    For pixels above 0.01, computes the min and max (foreground) and scales
    those pixels to the [0, 1] range. Pixels below 0.01 are forced to 0.

    Args:
        t_img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: The normalized image tensor.
    """
    if t_img.max() > 0:
        # Create mask for non-background (foreground) pixels.
        foreground_mask = t_img > 0.01

        # Check if any foreground pixel is found.
        if torch.any(foreground_mask):
            fg_values = t_img[foreground_mask]
            fg_min = fg_values.min()
            fg_max = fg_values.max()

            # Normalize only if there is a valid range.
            if fg_max > fg_min:
                # Use torch.where to selectively update foreground pixels.
                t_img = torch.where(
                    foreground_mask, (t_img - fg_min) / (fg_max - fg_min), t_img
                )

        # Force background (pixels < 0.01) to be 0
        t_img = torch.where(t_img < 0.01, torch.zeros_like(t_img), t_img)
    return t_img


def freeze():
    """
    for name, param in model.named_parameters():
        if 'encoder' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-3
    )
    """
    pass


def load(model, optimizer, save_path, scratch, device):
    print(f"Received save_path: {save_path}")
    print(f"Current working directory: {os.getcwd()}")

    checkpoints_dir = os.path.dirname(save_path)
    if not os.path.exists(checkpoints_dir) and not scratch:
        os.makedirs(checkpoints_dir, exist_ok=True)

    if scratch:
        try:
            checkpoint = torch.load(
                r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\ssn2v\checkpoints\stage1.pth"
            )
            print(checkpoint.keys())
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            old_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            print(
                f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch + 1}"
            )
            return model, optimizer, old_epoch, history
        except:
            raise ValueError(
                "Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path."
            )
    else:
        try:
            checkpoint = torch.load(save_path)
            print(checkpoint.keys())
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            old_epoch = checkpoint["epoch"]
            history = checkpoint["history"]
            print(
                f"Loaded model with val loss: {checkpoint['val_loss']:.6f} from epoch {old_epoch + 1}"
            )
            return model, optimizer, old_epoch, history
        except:
            raise ValueError(
                "Checkpoint not found. Please train the model from scratch or provide a valid checkpoint path."
            )


def check_performance(
    val_loss,
    best_val_loss,
    model,
    optimizer,
    epoch,
    save_path,
    history,
    old_epoch,
    avg_train_loss,
):
    if val_loss < best_val_loss:
        print(f"Saving model with val loss: {val_loss:.6f} from epoch {epoch + 1}")
        best_val_loss = val_loss
        try:
            torch.save(
                {
                    "epoch": epoch + old_epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": avg_train_loss,
                    "val_loss": val_loss,
                    "history": history,
                },
                save_path,
            )

        except:
            print("Err")
        print(
            f"Epoch {epoch + 1}, Training Loss: {avg_train_loss:.6f}, Validation Loss: {val_loss:.6f}"
        )
        print("-" * 50)


class Stage2(Dataset):
    def __init__(self, data_by_patient, transform=None):
        """
        Args:
            data_by_patient: Dictionary where keys are patient IDs and values are lists of samples
            transform: Optional transforms to apply
        """
        self.data_by_patient = data_by_patient
        self.transform = transform

        # Create an index mapping to locate samples
        self.index_mapping = []
        for patient_id, patient_data in self.data_by_patient.items():
            for i in range(len(patient_data) - 1):  # Ensure we can get pairs
                self.index_mapping.append((patient_id, i))

    def __len__(self):
        return len(self.index_mapping)

    def __getitem__(self, idx):
        # Get patient ID and sample index from our mapping
        patient_id, sample_idx = self.index_mapping[idx]

        # Get two consecutive samples from the same patient
        sample1 = self.data_by_patient[patient_id][sample_idx]
        sample2 = self.data_by_patient[patient_id][sample_idx + 1]

        # Extract the images from both samples
        preprocessed1 = sample1[0]  # First preprocessed image
        preprocessed2 = sample2[0]  # Second preprocessed image
        octa_calc = sample1[1]  # OCTA calculation
        stage1_output = sample1[2]  # Stage 1 output

        # Convert to tensors if they aren't already
        if not torch.is_tensor(preprocessed1):
            preprocessed1 = torch.tensor(preprocessed1, dtype=torch.float32).unsqueeze(
                0
            )
        if not torch.is_tensor(preprocessed2):
            preprocessed2 = torch.tensor(preprocessed2, dtype=torch.float32).unsqueeze(
                0
            )
        if not torch.is_tensor(octa_calc):
            octa_calc = torch.tensor(octa_calc, dtype=torch.float32).unsqueeze(0)
        if not torch.is_tensor(stage1_output):
            stage1_output = torch.tensor(stage1_output, dtype=torch.float32).unsqueeze(
                0
            )

        # Apply transforms
        if self.transform:
            preprocessed1 = self.transform(preprocessed1)
            preprocessed2 = self.transform(preprocessed2)
            octa_calc = self.transform(octa_calc)
            stage1_output = self.transform(stage1_output)

        # Return both preprocessed images along with other data
        return preprocessed1, preprocessed2, octa_calc, stage1_output


def get_stage2_loaders(dataset, img_size, test_split=0.2, val_split=0.15):
    # Get list of patient IDs
    patients = list(dataset.keys())
    print(f"Available patients: {patients}")

    # Prepare data by split type while maintaining patient separation
    train_data_by_patient = {}
    val_data_by_patient = {}
    test_data_by_patient = {}

    # Define transforms
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )

    # Process each patient separately
    for patient in patients:
        # Get patient's data
        patient_data = dataset[patient]

        # Shuffle patient's data
        random.shuffle(patient_data)

        # Calculate split sizes
        total_samples = len(patient_data)
        test_size = int(total_samples * test_split)
        val_size = int(total_samples * val_split)
        train_size = total_samples - test_size - val_size

        # Split patient's data
        train_data = patient_data[:train_size]
        val_data = patient_data[train_size : train_size + val_size]
        test_data = patient_data[train_size + val_size :]

        # Add to respective patient dictionaries
        if train_data:  # Only add if there's data
            train_data_by_patient[patient] = train_data
        if val_data:
            val_data_by_patient[patient] = val_data
        if test_data:
            test_data_by_patient[patient] = test_data

    # Create datasets with patient-specific organization
    train_dataset = Stage2(data_by_patient=train_data_by_patient, transform=transform)
    val_dataset = Stage2(data_by_patient=val_data_by_patient, transform=transform)
    test_dataset = Stage2(data_by_patient=test_data_by_patient, transform=transform)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True
    )

    print(
        f"Train size: {len(train_loader)}, Validation size: {len(val_loader)}, Test size: {len(test_loader)}"
    )

    return train_loader, val_loader, test_loader


def load_data(stage2_config):
    # n_patients = 4, background_thresh=0.01
    stage2_config["data_config"]["img_size"]
    n_patients = stage2_config["data_config"]["num_patients"]
    background_thresh = stage2_config["data_config"]["background_threshold"]
    stage1_data = r"C:\Datasets\OCTData\stage1_outputs"

    patients = range(1, n_patients)  # Adjust this to the number of patients you have

    def normalize_image(np_img):
        if np_img.max() > 0:
            # Create mask of non-background pixels
            foreground_mask = np_img > background_thresh
            if foreground_mask.any():
                # Get min/max of only foreground pixels
                fg_min = np_img[foreground_mask].min()
                fg_max = np_img[foreground_mask].max()

                # Normalize only foreground pixels to [0,1] range
                if fg_max > fg_min:
                    np_img[foreground_mask] = (np_img[foreground_mask] - fg_min) / (
                        fg_max - fg_min
                    )

        # Force background to be true black
        np_img[np_img < background_thresh] = 0
        return np_img

    stage1_dataset = {}

    for patient in patients:
        stage1_dataset[patient] = {}
        patient_path = os.path.join(stage1_data, f"{patient}")
        patient_data_len = len(
            [img for img in os.listdir(patient_path) if "raw" in img]
        )

        for i in range(1, patient_data_len + 1):
            # list images with raw in name

            stage1_dataset[patient][i] = {}

            raw_img = plt.imread(os.path.join(stage1_data, f"{patient}", f"raw{i}.png"))
            octa_img = plt.imread(
                os.path.join(stage1_data, f"{patient}", f"octa{i}.png")
            )

            # norm the octa
            octa_img = normalize_image(octa_img)

            stage1_dataset[patient][i]["raw"] = raw_img
            stage1_dataset[patient][i]["octa"] = octa_img[:, :, 1]

    n_patients = len(patients)
    n = 50
    n_images_per_patient = max(10, n)

    regular = stage2_config["data_config"]["regular"]

    if regular:
        dataset = preprocessing(
            n_patients, n_images_per_patient, n_neighbours=2, threshold=0
        )  # n neighbours must be 2
    else:
        dataset = preprocessing_v2(
            n_patients,
            n_images_per_patient,
            n_neighbours=stage2_config["data_config"]["n_neighbours"],
            threshold=background_thresh,
            post_process_size=stage2_config["data_config"]["post_process_size"],
        )
    for patient in dataset.keys():
        for i in range(len(dataset[patient])):
            dataset[patient][i].append(stage1_dataset[patient][i + 1]["octa"])

    return dataset


def normalize_image_torch(t_img: torch.Tensor) -> torch.Tensor:
    if t_img.max() > 0:
        foreground_mask = t_img > 0.01

        if torch.any(foreground_mask):
            fg_values = t_img[foreground_mask]
            fg_min = fg_values.min()
            fg_max = fg_values.max()
            if fg_max > fg_min:
                t_img = torch.where(
                    foreground_mask, (t_img - fg_min) / (fg_max - fg_min), t_img
                )

        # Force background (pixels < 0.01) to be 0
        t_img = torch.where(t_img < 0.01, torch.zeros_like(t_img), t_img)
    return t_img


def normalize_image_torch(t_img: torch.Tensor) -> torch.Tensor:
    """
    Normalise the input image tensor to [0, 1] range.

    Args:
        t_img (torch.Tensor): Input image tensor.

    Returns:
        torch.Tensor: The normalized image tensor.
    """
    min_val = t_img.min()
    max_val = t_img.max()

    if max_val > min_val:
        return (t_img - min_val) / (max_val - min_val)
    else:
        # If all values are the same, return zeros
        return torch.zeros_like(t_img)


def create_blind_spot_mask(
    batch_size, channels, height, width, device, blind_spot_ratio=0.1
):
    mask = torch.ones((batch_size, channels, height, width), device=device)

    # Determine number of pixels to mask per image
    num_pixels = int(height * width * blind_spot_ratio)

    for b in range(batch_size):
        for c in range(channels):
            # Generate random pixel coordinates
            coords = torch.randperm(height * width, device=device)[:num_pixels]
            y_coords = coords // width
            x_coords = coords % width

            # Set mask to 0 at blind spot locations
            mask[b, c, y_coords, x_coords] = 0

    return mask


def create_blind_spot_input_with_realistic_noise(image, mask):
    blind_input = image.clone()

    # Parameters for OCT-like noise (speckle)
    mean_level = image.mean()
    std_level = image.std() * 0.9  # Adjust based on your OCT noise level

    # Generate noise with speckle-like characteristics
    noise = torch.randn_like(image) * std_level + mean_level

    # Apply mask
    blind_input[mask.bool()] = noise[mask.bool()]
    return blind_input


def process_batch_n2v(
    model,
    loader,
    criterion,
    mask_ratio,
    optimizer=None,  # Optional parameter - present for training, None for evaluation
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            raw1, raw2 = batch

            raw1 = raw1.to(device)
            raw2 = raw2.to(device)

            mask = torch.bernoulli(
                torch.full(
                    (raw1.size(0), 1, raw1.size(2), raw1.size(3)),
                    mask_ratio,
                    device=device,
                )
            )

            blind1 = create_blind_spot_input_with_realistic_noise(
                raw1, mask
            ).requires_grad_(True)
            blind2 = create_blind_spot_input_with_realistic_noise(
                raw2, mask
            ).requires_grad_(True)

            if optimizer:
                optimizer.zero_grad()

            if speckle_module is not None:
                flow_inputs = speckle_module(raw1)
                flow_inputs = flow_inputs["flow_component"].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                outputs1 = model(blind1)

                # outputs1 = model(blind1)
                # outputs2 = model(blind2)
                flow_outputs = speckle_module(outputs1)
                flow_outputs = flow_outputs["flow_component"].detach()
                flow_outputs = normalize_image_torch(flow_outputs)
                flow_loss1 = torch.mean(torch.abs(flow_outputs - flow_inputs))

                flow_inputs = speckle_module(raw2)
                flow_inputs = flow_inputs["flow_component"].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                outputs2 = model(blind2)
                flow_outputs = speckle_module(outputs2)
                flow_outputs = flow_outputs["flow_component"].detach()
                flow_outputs = normalize_image_torch(flow_outputs)
                flow_loss2 = torch.mean(torch.abs(flow_outputs - flow_inputs))

                n2v_loss1 = criterion(outputs1[mask > 0], raw1[mask > 0])
                n2v_loss2 = criterion(outputs2[mask > 0], raw2[mask > 0])

                loss = n2v_loss1 + n2v_loss2 + flow_loss1 * alpha + flow_loss2 * alpha

            else:
                # outputs = model(input_imgs)
                # loss = criterion(outputs, target_imgs)

                outputs1 = model(blind1)
                outputs2 = model(blind2)

                n2v_loss1 = criterion(outputs1[mask > 0], raw1[mask > 0])
                n2v_loss2 = criterion(outputs2[mask > 0], raw2[mask > 0])

                loss = n2v_loss1 + n2v_loss2

            if optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

            if visualize and batch_idx == 0:
                assert raw1[0][0].shape == (256, 256)
                assert blind1[0][0].shape == (256, 256)
                assert outputs1[0][0].shape == (256, 256)

                if speckle_module is not None:
                    titles = [
                        "Input Image",
                        "Flow Input",
                        "Flow Output",
                        "Target Image",
                        "Output Image",
                    ]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs[0][0].cpu().detach().numpy(),
                        flow_outputs[0][0].cpu().detach().numpy(),
                        blind1[0][0].cpu().numpy(),
                        outputs1[0][0].cpu().detach().numpy(),
                    ]
                    losses = {
                        "Flow Loss": flow_loss1.item() + flow_loss2.item(),
                        "Total Loss": loss.item(),
                    }
                else:
                    titles = ["Input Image", "Target Image", "Output Image"]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        blind1[0][0].cpu().numpy(),
                        outputs1[0][0].cpu().detach().numpy(),
                    ]
                    losses = {"Total Loss": loss.item()}

                plot_images(images, titles, losses)

    return total_loss / len(loader)


def create_blind_spot_input(input_imgs, mask):
    """
    Create inputs for Noise2Void by replacing masked pixels with neighborhood means.

    Args:
        input_imgs (torch.Tensor): Original input images
        mask (torch.Tensor): Binary mask with 0s at blind spot locations

    Returns:
        torch.Tensor: Input with masked pixels replaced by neighborhood means
    """
    # Create a copy of the input to modify
    masked_input = input_imgs.clone()
    batch_size, channels, height, width = input_imgs.shape

    # For each image in the batch
    for b in range(batch_size):
        for c in range(channels):
            # Find the masked pixel coordinates
            y_coords, x_coords = torch.where(mask[b, c] == 0)

            # For each masked pixel
            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]

                # Calculate the neighborhood mean (3x3 window excluding center)
                neighborhood = []
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue  # Skip the center pixel

                        ny, nx = y + dy, x + dx
                        if 0 <= ny < height and 0 <= nx < width:
                            neighborhood.append(input_imgs[b, c, ny, nx].item())

                # Replace the masked pixel with the neighborhood mean
                if neighborhood:
                    masked_input[b, c, y, x] = sum(neighborhood) / len(neighborhood)

    return masked_input


def visualise_n2v(raw1, blind1, blind2, output1, output2):
    """
    Visualization function for Noise2Void results.
    This should be replaced with your actual visualization code.

    Args:
        raw1: Original images
        blind1: First masked input
        blind2: Second masked input
        output1: First model output
        output2: Second model output
    """
    import matplotlib.pyplot as plt

    # Create a figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Plot the images
    axes[0, 0].imshow(raw1[0, 0], cmap="gray")
    axes[0, 0].set_title("Original Image")

    axes[0, 1].imshow(blind1[0, 0], cmap="gray")
    axes[0, 1].set_title("Masked Input 1")

    axes[0, 2].imshow(output1[0, 0], cmap="gray")
    axes[0, 2].set_title("Output 1")

    axes[1, 0].imshow(raw1[0, 0] - output1[0, 0], cmap="gray")
    axes[1, 0].set_title("Residual 1")

    axes[1, 1].imshow(blind2[0, 0], cmap="gray")
    axes[1, 1].set_title("Masked Input 2")

    axes[1, 2].imshow(output2[0, 0], cmap="gray")
    axes[1, 2].set_title("Output 2")

    plt.tight_layout()
    plt.show()


def train_n2v(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    starting_epoch,
    epochs,
    batch_size,
    lr,
    best_val_loss,
    checkpoint_path=None,
    device="cuda",
    visualise=False,
    speckle_module=None,
    alpha=1,
    save=False,
    method="n2v",
    octa_criterion=None,
    threshold=0.0,
    mask_ratio=0.1,
):
    """
    Train function that handles both Noise2Void and Noise2Self approaches.

    Args:
        method (str): 'n2v' for Noise2Void or 'n2s' for Noise2Self
    """

    last_checkpoint_path = checkpoint_path + "_last_checkpoint.pth"
    best_checkpoint_path = checkpoint_path + "_best_checkpoint.pth"

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        model.train()

        # print(model)

        train_loss = process_batch_n2v(
            model,
            train_loader,
            criterion,
            mask_ratio,
            optimizer=optimizer,
            device="cuda",
            speckle_module=speckle_module,
            visualize=False,
        )

        model.eval()
        with torch.no_grad():
            val_loss = process_batch_n2v(
                model,
                val_loader,
                criterion,
                mask_ratio,
                optimizer=None,
                device="cuda",
                speckle_module=speckle_module,
                visualize=True,
            )

        print(
            f"Epoch [{epoch + 1}/{starting_epoch + epochs}], Average Loss: {train_loss:.6f}"
        )

        if val_loss < best_val_loss and save:
            best_val_loss = val_loss
            print(f"Saving best model with val loss: {val_loss:.6f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                best_checkpoint_path,
            )

    if save:
        print(f"Saving last model with val loss: {val_loss:.6f}")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
            },
            last_checkpoint_path,
        )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")

    return model
