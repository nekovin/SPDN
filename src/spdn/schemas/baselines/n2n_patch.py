import time
import torch
from spdn.utils.eval_utils.visualise import plot_images
from spdn.utils.data_utils.patch_processing import (
    extract_patches,
    reconstruct_from_patches,
)

from spdn.utils import evaluate_oct_denoising


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


def threshold_flow_component(
    t_img: torch.Tensor, threshold: float = 0.01, bottom_percent: float = 0.2
) -> torch.Tensor:
    """
    Creates a binary mask of the flow component with bottom portion blacked out.

    Args:
        t_img (torch.Tensor): Input flow component tensor
        threshold (float): Threshold value for binarization
        bottom_percent (float): Percentage of image height to black out from bottom

    Returns:
        torch.Tensor: Binary tensor with bottom portion set to zero
    """
    # Create binary threshold
    binary_mask = (t_img > threshold).float()

    # Create mask for bottom portion
    batch_size, channels, height, width = binary_mask.shape
    bottom_pixels = int(height * bottom_percent)

    # Create a mask where bottom 20% is zero and rest is one
    bottom_mask = torch.ones_like(binary_mask)
    bottom_mask[:, :, -bottom_pixels:, :] = 0

    # Apply both masks
    return binary_mask * bottom_mask


def process_batch(
    data_loader,
    model,
    criterion,
    optimizer,
    epoch,
    epochs,
    device,
    visualise,
    speckle_module,
    alpha,
    scheduler,
    sample,
    patch_size,
    stride,
    adaptive_loss,
):
    mode = "train" if model.training else "val"

    epoch_loss = 0
    metrics = None

    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        # Extract patches
        input_patches, patch_locations = extract_patches(input_imgs, patch_size, stride)
        target_patches, _ = extract_patches(target_imgs, patch_size, stride)

        sub_batch_size = 16
        total_loss = 0
        all_output_patches = []

        if optimizer is not None and mode == "train":
            optimizer.zero_grad()

        for i in range(0, len(input_patches), sub_batch_size):
            input_sub_batch = input_patches[i : i + sub_batch_size]
            target_sub_batch = target_patches[i : i + sub_batch_size]

            if speckle_module is not None:
                # Only input flow computation should be no_grad
                with torch.no_grad():
                    flow_inputs = speckle_module(input_sub_batch)
                    flow_inputs = flow_inputs["flow_component"]
                    flow_inputs = normalize_image_torch(flow_inputs)

                # Model inference and loss computation need gradients
                outputs = model(input_sub_batch)
                for j in range(outputs.size(0)):
                    all_output_patches.append(outputs[j].detach().clone())

                flow_outputs = speckle_module(outputs)
                flow_outputs = flow_outputs["flow_component"]
                flow_outputs = normalize_image_torch(flow_outputs)

                # flow_loss_abs = torch.mean(torch.abs(flow_outputs - flow_inputs))
                # flow_loss_mse = F.mse_loss(flow_outputs, flow_inputs)
                flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                # patch_loss = criterion(outputs, target_sub_batch) + flow_loss_abs * alpha + flow_loss_mse * alpha

                if adaptive_loss:
                    n2s_loss = criterion(outputs, target_sub_batch)
                    alpha_adaptive = n2s_loss.detach() / (flow_loss.detach() + 1e-8)
                    # sub_loss = n2s_loss + flow_loss * alpha_adaptive
                    patch_loss = n2s_loss + flow_loss * alpha_adaptive
                else:
                    patch_loss = (
                        criterion(outputs, target_sub_batch) + flow_loss * alpha
                    )

            else:
                outputs = model(input_sub_batch)
                for j in range(outputs.size(0)):
                    all_output_patches.append(outputs[j].detach().clone())
                patch_loss = criterion(outputs, target_sub_batch)

            if optimizer is not None and mode == "train":
                patch_loss.backward()

            total_loss += patch_loss.item() * len(input_sub_batch)

        if optimizer is not None and mode == "train":
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Reconstruct full images from patches for visualization
        if visualise and batch_idx % 10 == 0:
            sample_input = sample
            print(f"Sample input shape: {sample_input.shape}")
            sample_output = model(sample_input).cpu().numpy()
            output_patches = torch.stack(all_output_patches)
            reconstructed_outputs = reconstruct_from_patches(
                output_patches, patch_locations, input_imgs.shape, patch_size
            )

            if speckle_module is not None:
                flow_inputs_full = speckle_module(input_imgs)["flow_component"].detach()
                flow_outputs_full = speckle_module(reconstructed_outputs)[
                    "flow_component"
                ].detach()

                titles = [
                    "Input Image",
                    "Flow Input",
                    "Flow Output",
                    "Target Image",
                    "Output Image",
                    "Sample Input",
                    "Sample Output",
                ]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    flow_inputs_full[0][0].cpu().numpy(),
                    flow_outputs_full[0][0].cpu().numpy(),
                    target_imgs[0][0].cpu().numpy(),
                    reconstructed_outputs[0][0].cpu().numpy(),
                    sample_input.cpu().numpy()[0][0],
                    sample_output[0][0],
                ]
                losses = {
                    "Flow Loss": flow_loss.item(),
                    "Total Loss": total_loss / len(input_patches),
                }
            else:
                titles = [
                    "Input Image",
                    "Target Image",
                    "Output Image",
                    "Sample Input",
                    "Sample Output",
                ]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    target_imgs[0][0].cpu().numpy(),
                    reconstructed_outputs[0][0].cpu().numpy(),
                    sample_input.cpu().numpy()[0][0],
                    sample_output[0][0],
                ]
                losses = {"Total Loss": total_loss / len(input_patches)}

            plot_images(images, titles, losses)

            metrics = evaluate_oct_denoising(
                input_imgs[0][0].cpu().numpy(),
                reconstructed_outputs[0][0].cpu().numpy(),
            )

        loss_value = total_loss / len(input_patches)
        epoch_loss += loss_value

    if mode != "train":
        scheduler.step(loss_value)

    if metrics is not None:
        return epoch_loss / len(data_loader), metrics
    else:
        return epoch_loss / len(data_loader)


def train_n2n_patch(
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
    scheduler=None,
    best_metrics_score=None,
    train_config=None,
    sample=None,
    patch_size=128,
    stride=48,
    adaptive_loss=False,
):
    last_checkpoint_path = checkpoint_path + "_patched_last_checkpoint.pth"
    best_checkpoint_path = checkpoint_path + "_patched_best_checkpoint.pth"
    best_metrics_checkpoint_path = (
        checkpoint_path + "_patched_best_metrics_checkpoint.pth"
    )

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        model.train()
        visualise = False
        train_loss = process_batch_n2n_patch(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch,
            starting_epoch + epochs,
            device,
            visualise,
            speckle_module,
            alpha,
            scheduler,
            sample,
            patch_size,
            stride,
            adaptive_loss,
        )

        model.eval()
        visualise = True
        with torch.no_grad():
            val_loss, val_metrics = process_batch_n2n_patch(
                val_loader,
                model,
                criterion,
                optimizer,
                epoch,
                starting_epoch + epochs,
                device,
                visualise,
                speckle_module,
                alpha,
                scheduler,
                sample,
                patch_size,
                stride,
                adaptive_loss,
            )

            val_metrics_score = (
                val_metrics.get("snr", 0) * 0.3
                + val_metrics.get("cnr", 0) * 0.3
                + val_metrics.get("enl", 0) * 0.2
                + val_metrics.get("epi", 0) * 0.2
            )

        print(
            f"Epoch [{epoch + 1}/{starting_epoch + epochs}], Average Loss: {train_loss:.6f}"
        )

        if val_loss < best_val_loss and save:
            best_val_loss = val_loss
            print(f"Saving best model with val loss: {val_loss:.6f}")
            print(f"Best checkpoint path: {best_checkpoint_path}")
            print(f"Epoch: {epoch}, Best val loss: {best_val_loss:.6f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "train_config": train_config,
                    "metrics": val_metrics,
                    "metrics_score": val_metrics_score,
                },
                best_checkpoint_path,
            )

        if val_metrics_score > best_metrics_score and save:
            best_metrics_score = val_metrics_score
            print(f"Saving best metrics model with score: {val_metrics_score:.4f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "metrics": val_metrics,
                    "metrics_score": val_metrics_score,
                    "train_config": train_config,
                },
                best_metrics_checkpoint_path,
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
                    "train_config": train_config,
                    "metrics": val_metrics,
                    "metrics_score": val_metrics_score,
                },
                last_checkpoint_path,
            )

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")

    return model


def process_batch_n2n_patch(
    data_loader,
    model,
    criterion,
    optimizer,
    epoch,
    epochs,
    device,
    visualise,
    speckle_module,
    alpha,
    scheduler,
    sample,
    patch_size,
    stride,
    adaptive_loss,
):
    mode = "train" if model.training else "val"

    epoch_loss = 0
    metrics = None

    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        # Pre-compute flow input once per batch (computational optimization)
        flow_inputs_full = None
        if speckle_module is not None:
            with torch.no_grad():
                flow_inputs_full = speckle_module(input_imgs)["flow_component"]
                flow_inputs_full = normalize_image_torch(flow_inputs_full)

        # Extract patches
        input_patches, patch_locations = extract_patches(input_imgs, patch_size, stride)
        target_patches, _ = extract_patches(target_imgs, patch_size, stride)

        sub_batch_size = 16
        total_n2n_loss = 0.0
        all_output_patches = []
        final_outputs = None
        final_targets = None

        if optimizer is not None and mode == "train":
            optimizer.zero_grad()

        for i in range(0, len(input_patches), sub_batch_size):
            input_sub_batch = input_patches[i : i + sub_batch_size]
            target_sub_batch = target_patches[i : i + sub_batch_size]

            # Model inference
            outputs = model(input_sub_batch)

            # Store for reconstruction
            for j in range(outputs.size(0)):
                all_output_patches.append(outputs[j].detach().clone())

            # Keep final outputs for gradient computation if training
            if optimizer is not None and mode == "train":
                final_outputs = outputs
                final_targets = target_sub_batch

            # Accumulate N2N loss (scalar computation for efficiency)
            n2n_loss = criterion(outputs, target_sub_batch)
            total_n2n_loss += n2n_loss.item() * len(input_sub_batch)

        # Compute average N2N loss
        avg_n2n_loss = total_n2n_loss / len(input_patches)

        # Flow computation (only if speckle module exists)
        flow_loss_value = 0.0
        if speckle_module is not None:
            # Reconstruct full image from patches
            output_patches = torch.stack(all_output_patches)
            reconstructed_outputs = reconstruct_from_patches(
                output_patches, patch_locations, input_imgs.shape, patch_size
            )

            # Single flow computation on reconstructed image
            with torch.no_grad():
                flow_outputs_full = speckle_module(reconstructed_outputs)[
                    "flow_component"
                ]
                flow_outputs_full = normalize_image_torch(flow_outputs_full)
                flow_loss_value = torch.mean(
                    torch.abs(flow_outputs_full - flow_inputs_full)
                ).item()

        # Compute total loss
        if adaptive_loss and speckle_module is not None:
            alpha_adaptive = avg_n2n_loss / (flow_loss_value + 1e-8)
            total_batch_loss = avg_n2n_loss + flow_loss_value * alpha_adaptive
        else:
            total_batch_loss = avg_n2n_loss + flow_loss_value * alpha

        # Gradient computation (only during training)
        if optimizer is not None and mode == "train" and final_outputs is not None:
            # Compute loss with gradients for backprop
            if speckle_module is not None:
                # Need to recompute flow loss with gradients for final batch
                with torch.no_grad():
                    flow_inputs_patch = speckle_module(final_outputs.detach())[
                        "flow_component"
                    ]
                    flow_inputs_patch = normalize_image_torch(flow_inputs_patch)

                flow_outputs_patch = speckle_module(final_outputs)["flow_component"]
                flow_outputs_patch = normalize_image_torch(flow_outputs_patch)
                flow_loss_grad = torch.mean(
                    torch.abs(flow_outputs_patch - flow_inputs_patch)
                )

                if adaptive_loss:
                    n2n_loss_grad = criterion(final_outputs, final_targets)
                    alpha_adaptive = n2n_loss_grad.detach() / (
                        flow_loss_grad.detach() + 1e-8
                    )
                    patch_loss = n2n_loss_grad + flow_loss_grad * alpha_adaptive
                else:
                    patch_loss = (
                        criterion(final_outputs, final_targets) + flow_loss_grad * alpha
                    )
            else:
                patch_loss = criterion(final_outputs, final_targets)

            patch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Visualization
        if visualise and batch_idx % 10 == 0:
            # Ensure reconstruction exists
            if "reconstructed_outputs" not in locals():
                output_patches = torch.stack(all_output_patches)
                reconstructed_outputs = reconstruct_from_patches(
                    output_patches, patch_locations, input_imgs.shape, patch_size
                )

            sample_input = sample
            print(f"Sample input shape: {sample_input.shape}")
            sample_output = model(sample_input).cpu().numpy()

            if speckle_module is not None:
                titles = [
                    "Input Image",
                    "Flow Input",
                    "Flow Output",
                    "Target Image",
                    "Output Image",
                    "Sample Input",
                    "Sample Output",
                ]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    flow_inputs_full[0][0].cpu().numpy(),
                    flow_outputs_full[0][0].cpu().numpy(),
                    target_imgs[0][0].cpu().numpy(),
                    reconstructed_outputs[0][0].cpu().numpy(),
                    sample_input.cpu().numpy()[0][0],
                    sample_output[0][0],
                ]
                losses = {"Flow Loss": flow_loss_value, "Total Loss": total_batch_loss}
            else:
                titles = [
                    "Input Image",
                    "Target Image",
                    "Output Image",
                    "Sample Input",
                    "Sample Output",
                ]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    target_imgs[0][0].cpu().numpy(),
                    reconstructed_outputs[0][0].cpu().numpy(),
                    sample_input.cpu().numpy()[0][0],
                    sample_output[0][0],
                ]
                losses = {"Total Loss": total_batch_loss}

            plot_images(images, titles, losses)

            from spdn.utils import evaluate_oct_denoising

            metrics = evaluate_oct_denoising(
                input_imgs[0][0].cpu().numpy(),
                reconstructed_outputs[0][0].cpu().numpy(),
            )

        epoch_loss += total_batch_loss

    if mode != "train":
        scheduler.step(epoch_loss / len(data_loader))

    if metrics is not None:
        return epoch_loss / len(data_loader), metrics
    else:
        return epoch_loss / len(data_loader)
