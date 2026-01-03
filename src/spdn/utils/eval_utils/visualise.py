import matplotlib.pyplot as plt
from IPython.display import clear_output
from torchviz import make_dot
import torch
import numpy as np
from spdn.utils.data_utils.standard_preprocessing import normalize_image
from spdn.models.spdn.spdn_attention import SpatialAttention


def visualize_progress(model, input_tensor, target_tensor, masked_tensor, epoch):
    """
    Visualize the current model's output during training

    Args:
        model: Current model
        input_tensor: Input tensor [1, 1, H, W]
        target_tensor: Target tensor [1, 1, H, W]
        epoch: Current epoch number
    """
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    def adaptive_threshold(denoised_img, sensitivity=0.02):
        # Get an estimate of noise level from background
        bg_mask = denoised_img < np.percentile(denoised_img, 50)
        noise_std = np.std(denoised_img[bg_mask])

        # Set threshold as a multiple of background noise
        threshold = sensitivity * noise_std
        return np.maximum(denoised_img - threshold, 0)

    # Get components
    input_np = input_tensor[0, 0].cpu().numpy()
    target_np = target_tensor[0, 0].cpu().numpy()
    flow_np = output["flow_component"][0, 0].cpu().numpy()
    noise_np = output["noise_component"][0, 0].cpu().numpy()
    denoised_np = input_np - noise_np
    denoised_np = adaptive_threshold(denoised_np, sensitivity=0.02)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original data
    #
    axes[0, 0].imshow(input_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(target_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("Target")
    axes[0, 1].axis("off")

    # Row 2: Model outputs
    # normalise
    # axes[1, 0].imshow(flow_np, cmap='gray')
    #
    axes[1, 0].imshow(flow_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 0].set_title("Flow Component")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(noise_np, cmap="gray", vmin=0, vmax=1)
    axes[1, 1].set_title("Noise Component")
    axes[1, 1].axis("off")

    # axes[1, 2].imshow(denoised_np, cmap='gray')
    # axes[1, 2].set_title("Denoised (Input - Noise)")
    # axes[1, 2].axis('off')

    # axes[1, 2].imshow(masked_tensor, cmap='gray')
    # axes[1, 2].set_title("Masked Tensor")
    # axes[1, 2].axis('off')

    if masked_tensor is not None:
        axes[1, 2].imshow(masked_tensor, cmap="gray", vmin=0, vmax=1)
        axes[1, 2].set_title("Masked Tensor")
    else:
        axes[1, 2].set_title("No Masked Tensor")
    axes[1, 2].axis("off")

    flow_np = normalize_image(flow_np)
    axes[0, 2].imshow(flow_np, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("Flow Component (Normalized)")
    axes[0, 2].axis("off")

    plt.suptitle(f"Training Progress - Epoch {epoch}")
    plt.tight_layout()
    plt.show()


def visualize_attention_maps(model, input_tensor):
    """
    Visualize attention maps from a SpeckleSeparationUNetAttention model

    Args:
        model: Trained SpeckleSeparationUNetAttention model
        input_tensor: Input OCT image tensor/array
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # Set model to eval mode
    model.eval()

    # Ensure input is a proper torch tensor
    if isinstance(input_tensor, np.ndarray):
        # If input is numpy array, convert to tensor and add batch/channel dims if needed
        if input_tensor.ndim == 2:
            input_tensor = torch.tensor(
                input_tensor[np.newaxis, np.newaxis, ...], dtype=torch.float32
            ).to(next(model.parameters()).device)
        elif input_tensor.ndim == 3:
            input_tensor = torch.tensor(
                input_tensor[np.newaxis, ...], dtype=torch.float32
            ).to(next(model.parameters()).device)
        else:
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(
                next(model.parameters()).device
            )

    # Register hooks to capture attention maps
    attention_maps = []

    def hook_fn(module, input, output):
        # For spatial attention, the output is the product of input and attention map
        # We can get the attention map by dividing output by input
        if isinstance(module, SpatialAttention):
            # Extract just the attention map
            # The map is applied as x * attention_map, so we can derive it
            if input[0].sum() > 0:  # Avoid division by zero
                # Get the average over channels to visualize
                attention_map = output / (input[0] + 1e-10)
                attention_map = attention_map.mean(dim=1, keepdim=True)
                attention_maps.append(attention_map.detach().cpu())

    # Register hooks on spatial attention modules
    hooks = []
    # Encoder attention
    for i, attn_block in enumerate(model.encoder_attentions):
        # Access the spatial attention (second item in sequential)
        spatial_attn = attn_block[1]
        hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Bottleneck attention
    spatial_attn = model.bottleneck_attention[1]
    hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Decoder attention
    for i, attn_block in enumerate(model.decoder_attentions):
        spatial_attn = attn_block[1]
        hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Final attention
    spatial_attn = model.final_attention[1]
    hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Visualize attention maps
    fig, axes = plt.subplots(
        2, len(attention_maps) // 2 + len(attention_maps) % 2, figsize=(15, 6)
    )
    axes = axes.flatten()

    # Original input image
    if isinstance(input_tensor, torch.Tensor):
        input_np = input_tensor[0, 0].cpu().numpy()
    else:
        input_np = input_tensor

    axes[0].imshow(input_np, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot attention maps
    for i, attn_map in enumerate(attention_maps):
        if i + 1 >= len(axes):
            break
        im = axes[i + 1].imshow(attn_map[0, 0].numpy(), cmap="hot")
        axes[i + 1].set_title(f"Attention {i + 1}")
        axes[i + 1].axis("off")
        fig.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    plt.tight_layout()

    # Plot problematic area zoom (top right corner)
    if len(attention_maps) > 0:
        fig2, axes2 = plt.subplots(1, min(4, len(attention_maps)), figsize=(15, 3))
        if len(attention_maps) <= 4:
            axes2 = [axes2]

        # Get the upper right corner of each attention map
        for i, attn_map in enumerate(attention_maps[-min(4, len(attention_maps)) :]):
            if not isinstance(axes2, list):  # Handle case of single subplot
                ax = axes2
            else:
                ax = axes2[i]

            # Extract top right corner (25% of width, 25% of height)
            h, w = attn_map.shape[2], attn_map.shape[3]
            corner = attn_map[0, 0, : h // 4, 3 * w // 4 :].numpy()

            im = ax.imshow(corner, cmap="hot")
            ax.set_title(
                f"Corner Attention {len(attention_maps) - min(4, len(attention_maps)) + i + 1}"
            )
            fig2.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig, fig2, attention_maps

    return fig, None, attention_maps


def visualize_attention_maps(model, input_tensor):
    """
    Visualize attention maps from a SpeckleSeparationUNetAttention model

    Args:
        model: Trained SpeckleSeparationUNetAttention model
        input_tensor: Input OCT image tensor/array
    """
    import matplotlib.pyplot as plt
    import torch
    import numpy as np

    # Set model to eval mode
    model.eval()

    # Ensure input is a proper torch tensor
    if isinstance(input_tensor, np.ndarray):
        # If input is numpy array, convert to tensor and add batch/channel dims if needed
        if input_tensor.ndim == 2:
            input_tensor = torch.tensor(
                input_tensor[np.newaxis, np.newaxis, ...], dtype=torch.float32
            ).to(next(model.parameters()).device)
        elif input_tensor.ndim == 3:
            input_tensor = torch.tensor(
                input_tensor[np.newaxis, ...], dtype=torch.float32
            ).to(next(model.parameters()).device)
        else:
            input_tensor = torch.tensor(input_tensor, dtype=torch.float32).to(
                next(model.parameters()).device
            )

    # Register hooks to capture attention maps
    attention_maps = []

    def hook_fn(module, input, output):
        # For spatial attention modules
        if hasattr(
            module, "sigmoid"
        ):  # A simple way to identify spatial attention modules
            # Get attention map (rough approximation)
            attention_map = output / (input[0] + 1e-10)
            attention_map = attention_map.mean(dim=1, keepdim=True)
            attention_maps.append(attention_map.detach().cpu())

    # Register hooks on spatial attention modules
    hooks = []

    # Encoder attention
    for i, attn_block in enumerate(model.encoder_attentions):
        # Access the spatial attention (second item in sequential)
        spatial_attn = attn_block[1]
        hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Bottleneck attention
    spatial_attn = model.bottleneck_attention[1]
    hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Decoder attention
    for i, attn_block in enumerate(model.decoder_attentions):
        spatial_attn = attn_block[1]
        hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Final attention
    spatial_attn = model.final_attention[1]
    hooks.append(spatial_attn.register_forward_hook(hook_fn))

    # Forward pass
    with torch.no_grad():
        _ = model(input_tensor)

    # Remove hooks
    for hook in hooks:
        hook.remove()

    # Check if we have attention maps to visualize
    if len(attention_maps) == 0:
        print(
            "No attention maps were captured. Check if model contains attention modules."
        )
        return None, None, []

    # Visualize attention maps
    rows = 2
    cols = max(
        1, len(attention_maps) // rows + (1 if len(attention_maps) % rows else 0)
    )
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6))

    # Handle single axis case
    if rows * cols == 1:
        axes = np.array([axes])

    # Flatten axes for easier indexing
    axes = axes.flatten()

    # Original input image
    if isinstance(input_tensor, torch.Tensor):
        input_np = input_tensor[0, 0].cpu().numpy()
    else:
        input_np = input_tensor

    # Show input image in first subplot
    axes[0].imshow(input_np, cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot attention maps
    for i, attn_map in enumerate(attention_maps):
        if i + 1 >= len(axes):
            break
        im = axes[i + 1].imshow(attn_map[0, 0].numpy(), cmap="hot")
        axes[i + 1].set_title(f"Attention {i + 1}")
        axes[i + 1].axis("off")
        fig.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)

    # Hide unused subplots
    for i in range(len(attention_maps) + 1, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()

    # Plot problematic area zoom (top right corner)
    if len(attention_maps) > 0:
        n_corner_plots = min(4, len(attention_maps))
        fig2, corner_axes = plt.subplots(1, n_corner_plots, figsize=(12, 3))

        # Make sure corner_axes is iterable
        if n_corner_plots == 1:
            corner_axes = [corner_axes]

        # Get the upper right corner of each attention map
        for i in range(n_corner_plots):
            attn_map = attention_maps[-(i + 1)]  # Get maps from the end (deeper layers)

            # Extract top right corner (25% of width, 25% of height)
            h, w = attn_map.shape[2], attn_map.shape[3]
            corner = attn_map[0, 0, : h // 4, 3 * w // 4 :].numpy()

            im = corner_axes[i].imshow(corner, cmap="hot")
            corner_axes[i].set_title(f"Corner Attn Layer {len(attention_maps) - i}")
            fig2.colorbar(im, ax=corner_axes[i], fraction=0.046, pad=0.04)

        plt.tight_layout()
        return fig, fig2, attention_maps

    return fig, None, attention_maps


def plot_images(images, titles, losses):
    # clear output
    clear_output(wait=True)
    cols = len(images)
    n_images = len(images)
    rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5))
    loss_titles = []
    for loss_name, loss_value in losses.items():
        loss_titles.append(f"{loss_name}: {loss_value:.4f}")
    combined_title = " | ".join(loss_titles)
    fig.suptitle(combined_title, fontsize=16)
    for i, img in enumerate(images):
        axes[i].imshow(img, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(titles[i])
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()


def plot_computation_graph(model, loss, speckle_module):
    # Create visualization
    dot = make_dot(
        loss,
        params=dict(
            list(model.named_parameters()) + list(speckle_module.named_parameters())
        ),
    )
    dot.render("results/computation_graph", format="png")


#######

from spdn.utils.data_utils.patch_processing import (
    extract_patches,
    reconstruct_from_patches,
)


def visualize_progress_patch(
    model,
    input_tensor,
    target_tensor,
    masked_tensor,
    epoch,
    predicted_flow=None,
    predicted_noise=None,
):
    """
    Visualize the current model's output during training

    Args:
        model: Current model
        input_tensor: Input tensor [1, 1, H, W]
        target_tensor: Target tensor [1, 1, H, W]
        masked_tensor: Optional masked tensor
        epoch: Current epoch number
        predicted_flow: Pre-computed flow component (optional)
        predicted_noise: Pre-computed noise component (optional)
    """
    model.eval()

    # If we don't have pre-computed outputs, compute them
    if predicted_flow is None or predicted_noise is None:
        with torch.no_grad():
            # For full images, we need to use the patching approach
            patches, locations = extract_patches(input_tensor, patch_size=64)

            # Process patches in batches to avoid memory issues
            flow_patches = []
            noise_patches = []
            patch_batch_size = 8  # Adjust based on your GPU memory

            for i in range(0, len(patches), patch_batch_size):
                batch_patches = patches[i : i + patch_batch_size]
                outputs = model(batch_patches)
                flow_patches.append(outputs["flow_component"])
                noise_patches.append(outputs["noise_component"])

            # Concatenate results
            flow_patches = torch.cat(flow_patches, dim=0)
            noise_patches = torch.cat(noise_patches, dim=0)

            # Reconstruct full images
            predicted_flow = reconstruct_from_patches(
                flow_patches, locations, input_tensor.shape, patch_size=64
            )
            predicted_noise = reconstruct_from_patches(
                noise_patches, locations, input_tensor.shape, patch_size=64
            )

    def adaptive_threshold(denoised_img, sensitivity=0.02):
        # Get an estimate of noise level from background
        bg_mask = denoised_img < np.percentile(denoised_img, 50)
        noise_std = np.std(denoised_img[bg_mask])

        # Set threshold as a multiple of background noise
        threshold = sensitivity * noise_std
        return np.maximum(denoised_img - threshold, 0)

    # Get components
    input_np = input_tensor[0, 0].cpu().numpy()
    target_np = target_tensor[0, 0].cpu().numpy()
    flow_np = predicted_flow[0, 0].cpu().numpy()
    noise_np = predicted_noise[0, 0].cpu().numpy()
    denoised_np = input_np - noise_np
    denoised_np = adaptive_threshold(denoised_np, sensitivity=0.02)

    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Original data
    axes[0, 0].imshow(input_np, cmap="gray")
    axes[0, 0].set_title("Input")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(target_np, cmap="gray")
    axes[0, 1].set_title("Target")
    axes[0, 1].axis("off")

    # Row 2: Model outputs
    axes[1, 0].imshow(flow_np, cmap="gray")
    axes[1, 0].set_title("Flow Component")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(noise_np, cmap="gray")
    axes[1, 1].set_title("Noise Component")
    axes[1, 1].axis("off")

    if masked_tensor is not None:
        masked_np = (
            masked_tensor[0, 0].cpu().numpy()
            if isinstance(masked_tensor, torch.Tensor)
            else masked_tensor
        )
        axes[1, 2].imshow(masked_np, cmap="gray")
        axes[1, 2].set_title("Masked Tensor")
    else:
        axes[1, 2].set_title("No Masked Tensor")
    axes[1, 2].axis("off")

    # Normalize flow component for visualization
    normalized_flow = normalize_image(flow_np)
    axes[0, 2].imshow(normalized_flow, cmap="gray")
    axes[0, 2].set_title("Flow Component (Normalized)")
    axes[0, 2].axis("off")

    plt.suptitle(f"Training Progress - Epoch {epoch}")
    plt.tight_layout()
    plt.show()
