import time
import torch

from spdn.utils.eval_utils.visualise import plot_images

from spdn.utils import evaluate_oct_denoising
import torch.nn.functional as F

from spdn.utils.data_utils.patch_processing import (
    extract_patches,
    reconstruct_from_patches,
    threshold_patches,
)


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


def train_n2v_patch(
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
    best_metrics_score=float("-inf"),
    scheduler=None,
    train_config=None,
    sample=None,
    patch_size=64,
    stride=32,
    patience_count=10,
    adaptive_loss=False,
):
    """
    Train function that handles both Noise2Void and Noise2Self approaches.

    Args:
        method (str): 'n2v' for Noise2Void or 'n2s' for Noise2Self
    """

    last_checkpoint_path = checkpoint_path + "_patched_last_checkpoint.pth"
    best_checkpoint_path = checkpoint_path + "_patched_best_checkpoint.pth"
    best_metrics_checkpoint_path = (
        checkpoint_path + "_patched_best_metrics_checkpoint.pth"
    )

    print(f"Saving checkpoints to {best_checkpoint_path}")

    patience = 0

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        model.train()

        # print(model)
        print(f"Epoch [{epoch + 1}/{starting_epoch + epochs}]")

        train_loss = process_batch_n2v_patch(
            model,
            train_loader,
            criterion,
            mask_ratio,
            optimizer=optimizer,
            device="cuda",
            speckle_module=speckle_module,
            visualize=False,
            alpha=alpha,
            scheduler=scheduler,
            sample=sample,
            patch_size=patch_size,
            stride=stride,
            adaptive_loss=adaptive_loss,
        )

        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = process_batch_n2v_patch(
                model,
                val_loader,
                criterion,
                mask_ratio,
                optimizer=None,
                device="cuda",
                speckle_module=speckle_module,
                visualize=True,
                alpha=alpha,
                scheduler=scheduler,
                sample=sample,
                patch_size=patch_size,
                stride=stride,
                adaptive_loss=adaptive_loss,
            )

            val_metrics_score = (
                val_metrics.get("snr", 0) * 0.3
                + val_metrics.get("cnr", 0) * 0.3
                + val_metrics.get("enl", 0) * 0.2
                + val_metrics.get("epi", 0) * 0.2
            )

        if scheduler is not None:
            scheduler.step(val_loss)

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
                    "metrics": val_metrics,
                    "metrics_score": val_metrics_score,
                    "train_config": train_config,
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
                    "metrics": val_metrics,
                    "metrics_score": val_metrics_score,
                    "train_config": train_config,
                },
                last_checkpoint_path,
            )

        # Early stopping based on validation loss
        if val_loss < best_val_loss:
            patience = 0
        else:
            patience += 1
            if patience >= patience_count:
                print(
                    f"Early stopping at epoch {epoch + 1} due to no improvement in validation loss."
                )
                break

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")

    return model


def create_blind_spot_input(input_tensor, mask):
    expanded_mask = mask.expand(-1, input_tensor.size(1), -1, -1)

    output = input_tensor.clone()

    shifted_down = torch.roll(input_tensor, shifts=1, dims=2)
    shifted_down[:, :, 0, :] = input_tensor[:, :, 0, :]

    output = torch.where(expanded_mask > 0, shifted_down, output)

    return output


def split_spectrum_for_n2v(image, num_bands=8, structure_noise_ratio=None):
    """
    Split image spectrum into bands with adaptive parameters based on image statistics.

    Parameters:
    -----------
    image : 2D numpy array
        OCT image to process
    num_bands : int
        Number of frequency bands to split into (default: 4)
    structure_noise_ratio : float or None
        If None, will be calculated adaptively based on image characteristics

    Returns:
    --------
    dict
        Dictionary containing structure, noise, and blended components
    """
    import numpy as np
    from scipy import fftpack

    # Get image dimensions
    h, w = image.shape

    # Calculate image statistics for adaptive parameters
    img_mean = np.mean(image)
    img_std = np.std(image)
    img_entropy = -np.sum(
        np.histogram(image, bins=256, density=True)[0]
        * np.log2(np.histogram(image, bins=256, density=True)[0] + 1e-10)
    )

    # Adaptively set structure_noise_ratio based on image complexity
    # Higher entropy (more complex images) -> preserve more structure
    if structure_noise_ratio is None:
        # Normalize entropy to 0-1 range (typical OCT entropy ranges from 3-7)
        norm_entropy = np.clip((img_entropy - 3) / 4, 0, 1)
        # More complex images (higher entropy) get higher structure ratio
        structure_noise_ratio = 0.6 + 0.3 * norm_entropy

    # Adaptively set decay factor based on signal-to-noise characteristics
    # Higher SNR -> faster decay (more emphasis on low frequencies)
    snr_estimate = img_mean / (img_std + 1e-8)
    decay_factor = np.clip(2.0 + snr_estimate / 2.0, 2.0, 5.0)

    # Apply FFT
    fft_image = fftpack.fft2(image)
    fft_shifted = fftpack.fftshift(fft_image)

    # Center coordinates
    y_center, x_center = h // 2, w // 2

    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    distance_map = np.sqrt((y - y_center) ** 2 + (x - x_center) ** 2)

    # Maximum radius (from center to corner)
    max_radius = np.sqrt((h / 2) ** 2 + (w / 2) ** 2)

    # Create bands with exponential spacing
    bands = []
    structure_weights = []
    noise_weights = []

    # Generate exponentially spaced radii
    radii = []
    for i in range(num_bands + 1):
        radius = (
            max_radius
            * (np.exp(i * np.log(2.0) / num_bands) - 1)
            / (np.exp(np.log(2.0)) - 1)
        )
        radii.append(radius)

    # Extract bands and assign weights
    for i in range(num_bands):
        inner_radius = radii[i]
        outer_radius = radii[i + 1]

        # Create band mask
        mask = (distance_map > inner_radius) & (distance_map <= outer_radius)

        # Apply mask
        band_fft = fft_shifted.copy()
        band_fft[~mask] = 0

        # Inverse FFT to get band
        band_ifft = fftpack.ifftshift(band_fft)
        band = np.abs(fftpack.ifft2(band_ifft))
        bands.append(band)

        # Assign weights with adaptive decay
        structure_weight = np.exp(-decay_factor * i / (num_bands - 1))
        structure_weights.append(structure_weight)
        noise_weights.append(1.0 - structure_weight)

    # Create structure and noise components
    structure = np.zeros_like(image)
    noise = np.zeros_like(image)

    for i, band in enumerate(bands):
        structure += structure_weights[i] * band
        noise += noise_weights[i] * band

    # Normalize components using robust scaling to handle outliers
    def robust_normalize(img):
        p_low, p_high = np.percentile(img, [1, 99])
        normalized = np.clip((img - p_low) / (p_high - p_low + 1e-8), 0, 1)
        return normalized

    structure_norm = robust_normalize(structure)
    noise_norm = robust_normalize(noise)

    # Create blended output based on structure_noise_ratio
    blended = (
        structure_noise_ratio * structure_norm
        + (1 - structure_noise_ratio) * noise_norm
    )

    return {
        "structure": structure_norm,
        "noise": noise_norm,
        "blended": blended,
        "params": {
            "structure_noise_ratio": structure_noise_ratio,
            "decay_factor": decay_factor,
            "entropy": img_entropy,
            "snr_estimate": snr_estimate,
        },
    }


def _old_process_batch_n2v_patch(
    model,
    loader,
    criterion,
    mask_ratio,
    optimizer=None,  # Optional parameter - present for training, None for evaluation
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
):
    if optimizer:
        model.train()
        mode = "TRAINING"
    else:
        model.eval()
        mode = "EVALUATION"

    print(
        f"\n===== STARTING {mode} WITH {'SPECKLE MODULE' if speckle_module else 'NO SPECKLE MODULE'} ====="
    )
    print(f"Patch size: {patch_size}, Stride: {stride}, Mask ratio: {mask_ratio}")

    total_loss = 0.0
    metrics = None

    for batch_idx, batch in enumerate(loader):
        print(f"\n----- Processing batch {batch_idx + 1}/{len(loader)} -----")

        # Load data to device
        raw1, _ = batch  # Ignore second image
        raw1 = raw1.to(device)

        print("Extracting patches...")
        raw1_patches, patch_locations = extract_patches(raw1, patch_size, stride)
        print(f"Created {len(raw1_patches)} patches")

        # Initialize storage for visualization only if needed
        reconstructed_outputs = None
        if visualize and batch_idx == 0:
            all_outputs = torch.zeros_like(raw1_patches)

        # Initialize batch loss
        batch_loss = 0.0

        # Zero gradients once per batch if training
        if optimizer:
            optimizer.zero_grad()
            print("Zeroed gradients for batch")

        # Create masks for all patches at once (more efficient)
        # Using broadcasting to create masks
        print("Creating masks for loss calculation...")
        masks = torch.bernoulli(
            torch.ones(len(raw1_patches), 1, patch_size, patch_size, device=device)
            * mask_ratio
        )

        # Process patches in sub-batches to avoid OOM
        sub_batch_size = 32  # Increased from 16
        print(
            f"Processing {len(raw1_patches)} patches in sub-batches of {sub_batch_size}..."
        )

        # Track flow loss for visualization
        flow_loss_value = 0

        for i in range(0, len(raw1_patches), sub_batch_size):
            end_idx = min(i + sub_batch_size, len(raw1_patches))

            # Get current sub-batch of patches and masks
            inputs = raw1_patches[i:end_idx]
            current_masks = masks[i:end_idx]

            # Forward pass through model
            outputs = model(inputs)

            # Store outputs for visualization if needed
            if visualize and batch_idx == 0:
                all_outputs[i:end_idx] = outputs.detach()

            # Calculate loss on masked pixels more efficiently
            # Element-wise multiplication with mask for masked inputs/outputs
            masked_outputs = outputs * current_masks
            masked_targets = inputs * current_masks

            # Calculate MSE loss on masked pixels only
            # Sum and normalize by number of masked pixels
            num_masked_pixels = current_masks.sum() + 1e-8
            mse_loss = (
                torch.sum((masked_outputs - masked_targets) ** 2) / num_masked_pixels
            )

            # Calculate flow loss if speckle module exists
            if speckle_module is not None:
                flow_inputs = speckle_module(inputs)["flow_component"].detach()
                flow_outputs = speckle_module(outputs)["flow_component"].detach()

                # flow_loss = F.l1_loss(flow_outputs, flow_inputs)
                flow_loss = F.mse_loss(flow_outputs, flow_inputs)
                flow_loss_value = flow_loss.item()

                # Combine losses with scaling factor
                sub_loss = mse_loss + alpha * flow_loss
            else:
                sub_loss = mse_loss

            # Scale loss for gradient accumulation
            # Use a fraction based on current sub-batch size
            sub_batch_fraction = (end_idx - i) / len(raw1_patches)
            sub_loss = sub_loss * sub_batch_fraction

            # Backward pass if training
            if optimizer:
                sub_loss.backward()

            # Add to batch loss (unnormalized)
            batch_loss += (
                sub_loss.item() / sub_batch_fraction
            )  # Undo scaling for reporting

        # End of patch processing

        # Update weights after processing all sub-batches
        if optimizer:
            print("Updating model weights...")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Only reconstruct full images for visualization to save memory
        if visualize and batch_idx == 0:
            print("Generating visualizations...")

            # Process sample image if provided
            if sample is not None:
                print(f"Processing sample input (shape: {sample.shape})...")
                sample_output = model(sample).cpu().numpy()

            # Reconstruct full images from patches for visualization
            print("  Reconstructing full images from patches...")
            reconstructed_outputs = reconstruct_from_patches(
                all_outputs, patch_locations, raw1.shape, patch_size
            )

            # Visualize results
            if speckle_module is not None:
                print("  Calculating flow components...")
                flow_inputs_full = speckle_module(raw1)["flow_component"].detach()
                flow_outputs_full = speckle_module(reconstructed_outputs)[
                    "flow_component"
                ].detach()

                titles = ["Input Image", "Flow Input", "Flow Output", "Output Image"]
                images = [
                    raw1[0][0].cpu().numpy(),
                    flow_inputs_full[0][0].cpu().numpy(),
                    flow_outputs_full[0][0].cpu().numpy(),
                    reconstructed_outputs[0][0].cpu().numpy(),
                ]

                # Add sample images if available
                if sample is not None:
                    titles.extend(["Sample Input", "Sample Output"])
                    images.extend([sample.cpu().numpy()[0][0], sample_output[0][0]])

                losses = {"Flow Loss": flow_loss_value, "Total Loss": batch_loss}
            else:
                titles = ["Input Image", "Output Image"]
                images = [
                    raw1[0][0].cpu().numpy(),
                    reconstructed_outputs[0][0].cpu().numpy(),
                ]

                # Add sample images if available
                if sample is not None:
                    titles.extend(["Sample Input", "Sample Output"])
                    images.extend([sample.cpu().numpy()[0][0], sample_output[0][0]])

                losses = {"Total Loss": batch_loss}

            print("  Plotting images...")
            plot_images(images, titles, losses)

            print("  Calculating evaluation metrics...")
            metrics = evaluate_oct_denoising(
                raw1[0][0].cpu().numpy(), reconstructed_outputs[0][0].cpu().numpy()
            )

            # Free memory
            del all_outputs
            if reconstructed_outputs is not None:
                del reconstructed_outputs
            torch.cuda.empty_cache()

        # Track average loss
        total_loss += batch_loss

        print(f"\nBatch {batch_idx + 1} completed")
        print(f"  Batch loss: {batch_loss:.6f}")

    # Calculate average loss across all batches
    avg_loss = total_loss / len(loader)

    # Print final summary
    print("\n===== SUMMARY =====")
    print(f"Average loss: {avg_loss:.6f}")
    print("===================\n")

    # Return metrics if available
    if metrics is not None:
        return avg_loss, metrics
    else:
        return avg_loss


from contextlib import nullcontext
from tqdm import tqdm
from spdn.utils.data_utils.standard_preprocessing import normalize_image_torch
from spdn.utils.noise import create_blind_spot_input_with_realistic_noise


def _process_batch_n2v_patch(
    model,
    loader,
    criterion,
    mask_ratio,
    optimizer=None,  # Optional parameter - present for training, None for evaluation
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,  # Choose appropriate patch size
    stride=32,
    adaptive_loss=False,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    threshold = 0.5

    total_loss = 0.0

    metrics = None

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            raw1, _ = batch

            raw1 = raw1.to(device)
            # raw2 = raw2.to(device)

            # Extract patches
            raw1_patches, patch_locations1 = extract_patches(raw1, patch_size, stride)
            # raw2_patches, patch_locations2 = extract_patches(raw2, patch_size, stride)

            print(f"Raw1 patches shape: {raw1_patches.shape}")
            # print(f"Raw2 patches shape: {raw2_patches.shape}")

            # Process patches in sub-batches to avoid memory issues
            sub_batch_size = 32  # Adjust based on your GPU memory
            batch_loss = 0.0
            all_output1_patches = []

            for i in range(0, len(raw1_patches), sub_batch_size):
                raw1_sub_batch = raw1_patches[i : i + sub_batch_size]
                # raw2_sub_batch = raw2_patches[i:i+sub_batch_size]

                mask = torch.bernoulli(
                    torch.full(
                        (
                            raw1_sub_batch.size(0),
                            1,
                            raw1_sub_batch.size(2),
                            raw1_sub_batch.size(3),
                        ),
                        mask_ratio,
                        device=device,
                    )
                )

                blind1 = create_blind_spot_input_with_realistic_noise(
                    raw1_sub_batch, mask
                ).requires_grad_(True)
                # blind2 = create_blind_spot_input_with_realistic_noise(raw2_sub_batch, mask).requires_grad_(True)

                if speckle_module is not None:
                    flow_inputs = speckle_module(raw1_sub_batch)
                    flow_inputs = flow_inputs["flow_component"].detach()
                    flow_inputs = normalize_image_torch(flow_inputs)

                    outputs1 = model(blind1)
                    all_output1_patches.append(outputs1.detach())

                    flow_outputs = speckle_module(outputs1)
                    flow_outputs = flow_outputs["flow_component"].detach()
                    flow_outputs = normalize_image_torch(flow_outputs)

                    flow_inputs = threshold_patches(flow_inputs, threshold=threshold)
                    flow_outputs = threshold_patches(flow_outputs, threshold=threshold)

                    flow_loss1 = torch.mean(torch.abs(flow_inputs - flow_outputs))

                    n2v_loss1 = criterion(outputs1[mask > 0], raw1_sub_batch[mask > 0])

                    if adaptive_loss:
                        alpha_adaptive = n2v_loss1.detach() / (
                            flow_loss1.detach() + 1e-8
                        )
                        sub_loss = n2v_loss1 + flow_loss1 * alpha_adaptive
                    else:
                        sub_loss = n2v_loss1 + flow_loss1 * alpha

                else:
                    outputs1 = model(blind1)
                    # outputs2 = model(blind2)
                    all_output1_patches.append(outputs1.detach())
                    # all_output2_patches.append(outputs2.detach())

                    n2v_loss1 = criterion(outputs1[mask > 0], raw1_sub_batch[mask > 0])
                    # n2v_loss2 = criterion(outputs2[mask > 0], raw2_sub_batch[mask > 0])

                    sub_loss = n2v_loss1

                batch_loss += sub_loss.item() * len(raw1_sub_batch)

            if optimizer:
                optimizer.zero_grad()
                sub_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += batch_loss / len(raw1_patches)

            if visualize and batch_idx == 0:
                # Flatten all output patches
                output1_patches = torch.cat(all_output1_patches, dim=0)
                # output2_patches = torch.cat(all_output2_patches, dim=0)

                reconstructed_outputs1 = reconstruct_from_patches(
                    output1_patches, patch_locations1, raw1.shape, patch_size
                )

                sample_output = model(sample)
                sample_output = sample_output.cpu().numpy()

                if speckle_module is not None:
                    # Create flow components for visualization
                    flow_inputs_full = speckle_module(raw1)["flow_component"].detach()
                    flow_outputs_full = speckle_module(reconstructed_outputs1)[
                        "flow_component"
                    ].detach()

                    flow_inputs_full = threshold_patches(
                        flow_inputs_full, threshold=threshold
                    )
                    flow_outputs_full = threshold_patches(
                        flow_outputs_full, threshold=threshold
                    )

                    titles = [
                        "Input Image",
                        "Flow Input",
                        "Flow Output",
                        "Blind Spot Input",
                        "Output Image",
                        "Sample Input",
                        "Sample Output",
                    ]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs_full[0][0].cpu().numpy(),
                        flow_outputs_full[0][0].cpu().numpy(),
                        blind1[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                        sample.cpu().numpy()[0][0] if sample is not None else None,
                        sample_output[0][0] if sample is not None else None,
                    ]
                    losses = {
                        "Flow Loss": (flow_loss1.item()),
                        "Total Loss": batch_loss / len(raw1_patches),
                    }
                else:
                    titles = [
                        "Input Image",
                        "Blind Spot Input",
                        "Output Image",
                        "Sample Input",
                        "Sample Output",
                    ]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        blind1[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                        sample.cpu().numpy()[0][0] if sample is not None else None,
                        sample_output[0][0] if sample is not None else None,
                    ]
                    losses = {"Total Loss": batch_loss / len(raw1_patches)}

                plot_images(images, titles, losses)

                from spdn.utils import evaluate_oct_denoising

                metrics = evaluate_oct_denoising(
                    raw1[0][0].cpu().numpy(), reconstructed_outputs1[0][0].cpu().numpy()
                )

    if metrics is not None:
        return total_loss / len(loader), metrics
    else:
        return total_loss / len(loader)


def process_batch_n2v_patch(
    model,
    loader,
    criterion,
    mask_ratio,
    optimizer=None,  # Optional parameter - present for training, None for evaluation
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,  # Choose appropriate patch size
    stride=32,
    adaptive_loss=False,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    threshold = 0.5
    total_loss = 0.0
    metrics = None

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    for batch_idx, batch in enumerate(tqdm(loader)):
        raw1, _ = batch
        raw1 = raw1.to(device)

        # Extract patches
        raw1_patches, patch_locations1 = extract_patches(raw1, patch_size, stride)
        print(f"Raw1 patches shape: {raw1_patches.shape}")

        # Process patches in sub-batches
        sub_batch_size = 32
        n2v_losses = []
        all_output1_patches = []
        all_masks = []
        all_raw_patches = []

        # Forward pass through patches
        with context_manager:
            for i in range(0, len(raw1_patches), sub_batch_size):
                raw1_sub_batch = raw1_patches[i : i + sub_batch_size]

                mask = torch.bernoulli(
                    torch.full(
                        (
                            raw1_sub_batch.size(0),
                            1,
                            raw1_sub_batch.size(2),
                            raw1_sub_batch.size(3),
                        ),
                        mask_ratio,
                        device=device,
                    )
                )

                blind1 = create_blind_spot_input_with_realistic_noise(
                    raw1_sub_batch, mask
                )
                if optimizer:
                    blind1 = blind1.requires_grad_(True)

                outputs1 = model(blind1)

                if optimizer:
                    # Store for gradient computation
                    all_output1_patches.append(outputs1)
                    all_masks.append(mask)
                    all_raw_patches.append(raw1_sub_batch)
                else:
                    all_output1_patches.append(outputs1.detach())

                # Compute N2V loss for this sub-batch
                n2v_loss = criterion(outputs1[mask > 0], raw1_sub_batch[mask > 0])
                n2v_losses.append(n2v_loss if optimizer else n2v_loss.item())

        # Reconstruct full images
        if optimizer:
            output1_patches = torch.cat(all_output1_patches, dim=0)
        else:
            output1_patches = torch.cat(all_output1_patches, dim=0)

        reconstructed_outputs1 = reconstruct_from_patches(
            output1_patches, patch_locations1, raw1.shape, patch_size
        )

        # Compute total loss
        if optimizer:
            # Training mode - maintain gradients
            total_n2v_loss = sum(n2v_losses)

            if speckle_module is not None:
                # Flow computation with gradients
                flow_inputs = speckle_module(raw1)["flow_component"].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                flow_inputs = threshold_patches(flow_inputs, threshold=threshold)

                flow_outputs = speckle_module(reconstructed_outputs1)["flow_component"]
                flow_outputs = normalize_image_torch(flow_outputs)
                flow_outputs = threshold_patches(flow_outputs, threshold=threshold)

                flow_loss = torch.mean(torch.abs(flow_inputs - flow_outputs))

                if adaptive_loss:
                    alpha_adaptive = total_n2v_loss.detach() / (
                        flow_loss.detach() + 1e-8
                    )
                    total_batch_loss = total_n2v_loss + flow_loss * alpha_adaptive
                else:
                    total_batch_loss = total_n2v_loss + flow_loss * alpha
            else:
                total_batch_loss = total_n2v_loss
                flow_loss = None

            # Single backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            batch_loss_value = total_batch_loss.item()

        else:
            # Evaluation mode
            batch_n2v_loss = sum(n2v_losses)

            if speckle_module is not None:
                flow_inputs = speckle_module(raw1)["flow_component"].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                flow_inputs = threshold_patches(flow_inputs, threshold=threshold)

                flow_outputs = speckle_module(reconstructed_outputs1.detach())[
                    "flow_component"
                ].detach()
                flow_outputs = normalize_image_torch(flow_outputs)
                flow_outputs = threshold_patches(flow_outputs, threshold=threshold)

                flow_loss = torch.mean(torch.abs(flow_inputs - flow_outputs))

                if adaptive_loss:
                    alpha_adaptive = batch_n2v_loss / (flow_loss.item() + 1e-8)
                    batch_loss_value = (
                        batch_n2v_loss + flow_loss.item() * alpha_adaptive
                    )
                else:
                    batch_loss_value = batch_n2v_loss + flow_loss.item() * alpha
            else:
                batch_loss_value = batch_n2v_loss
                flow_loss = None

        total_loss += batch_loss_value

        # Visualization
        if visualize and batch_idx == 0:
            sample_output = model(sample)
            sample_output = sample_output.cpu().numpy()

            if speckle_module is not None:
                titles = [
                    "Input Image",
                    "Flow Input",
                    "Flow Output",
                    "Blind Spot Input",
                    "Output Image",
                    "Sample Input",
                    "Sample Output",
                ]
                images = [
                    raw1[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().numpy(),
                    flow_outputs[0][0].cpu().numpy(),
                    blind1[0][0].cpu().numpy(),
                    reconstructed_outputs1[0][0].cpu().numpy(),
                    sample.cpu().numpy()[0][0] if sample is not None else None,
                    sample_output[0][0] if sample is not None else None,
                ]
                losses = {
                    "Flow Loss": flow_loss.item() if flow_loss is not None else 0,
                    "Total Loss": batch_loss_value,
                }
            else:
                titles = [
                    "Input Image",
                    "Blind Spot Input",
                    "Output Image",
                    "Sample Input",
                    "Sample Output",
                ]
                images = [
                    raw1[0][0].cpu().numpy(),
                    blind1[0][0].cpu().numpy(),
                    reconstructed_outputs1[0][0].cpu().numpy(),
                    sample.cpu().numpy()[0][0] if sample is not None else None,
                    sample_output[0][0] if sample is not None else None,
                ]
                losses = {"Total Loss": batch_loss_value}

            plot_images(images, titles, losses)

            from spdn.utils import evaluate_oct_denoising

            metrics = evaluate_oct_denoising(
                raw1[0][0].cpu().numpy(), reconstructed_outputs1[0][0].cpu().numpy()
            )

    if metrics is not None:
        return total_loss / len(loader), metrics
    else:
        return total_loss / len(loader)


def process_batch_n2v_patch(
    model,
    loader,
    criterion,
    mask_ratio,
    optimizer=None,
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
    adaptive_loss=False,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    threshold = 0.5
    total_loss = 0.0
    metrics = None

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            raw1, _ = batch
            raw1 = raw1.to(device)

            # Extract patches once
            raw1_patches, patch_locations1 = extract_patches(raw1, patch_size, stride)
            print(f"Raw1 patches shape: {raw1_patches.shape}")

            # Pre-compute flow input once per batch (computational optimization)
            flow_inputs = None
            if speckle_module is not None:
                with torch.no_grad():
                    flow_inputs = speckle_module(raw1)["flow_component"]
                    flow_inputs = normalize_image_torch(flow_inputs)
                    flow_inputs = threshold_patches(flow_inputs, threshold=threshold)

            # Process patches
            sub_batch_size = 32
            batch_n2v_loss = 0.0
            all_output_patches = []
            final_outputs = None
            final_mask = None
            final_raw = None

            for i in range(0, len(raw1_patches), sub_batch_size):
                raw1_sub_batch = raw1_patches[i : i + sub_batch_size]

                mask = torch.bernoulli(
                    torch.full(
                        (
                            raw1_sub_batch.size(0),
                            1,
                            raw1_sub_batch.size(2),
                            raw1_sub_batch.size(3),
                        ),
                        mask_ratio,
                        device=device,
                    )
                )

                blind1 = create_blind_spot_input_with_realistic_noise(
                    raw1_sub_batch, mask
                )
                if optimizer:
                    blind1 = blind1.requires_grad_(True)

                outputs1 = model(blind1)

                # Store for reconstruction
                all_output_patches.append(outputs1.detach())

                # Keep final outputs for gradient computation if training
                if optimizer:
                    final_outputs = outputs1
                    final_mask = mask
                    final_raw = raw1_sub_batch

                # Accumulate N2V loss
                n2v_loss = criterion(outputs1[mask > 0], raw1_sub_batch[mask > 0])
                batch_n2v_loss += n2v_loss.item() * len(raw1_sub_batch)

            # Compute average N2V loss
            avg_n2v_loss = batch_n2v_loss / len(raw1_patches)

            # Flow computation (only if speckle module exists)
            flow_loss_value = 0.0
            if speckle_module is not None:
                # Reconstruct full image from patches
                output_patches_cat = torch.cat(all_output_patches, dim=0)
                reconstructed_outputs = reconstruct_from_patches(
                    output_patches_cat, patch_locations1, raw1.shape, patch_size
                )

                # Single flow computation on reconstructed image
                with torch.no_grad():
                    flow_outputs = speckle_module(reconstructed_outputs)[
                        "flow_component"
                    ]
                    flow_outputs = normalize_image_torch(flow_outputs)
                    flow_outputs = threshold_patches(flow_outputs, threshold=threshold)

                    flow_loss_value = torch.mean(
                        torch.abs(flow_inputs - flow_outputs)
                    ).item()

            # Compute total loss
            if adaptive_loss and speckle_module is not None:
                alpha_adaptive = avg_n2v_loss / (flow_loss_value + 1e-8)
                total_batch_loss = avg_n2v_loss + flow_loss_value * alpha_adaptive
            else:
                total_batch_loss = avg_n2v_loss + flow_loss_value * alpha

            # Gradient computation (only during training)
            if optimizer and final_outputs is not None:
                optimizer.zero_grad()
                train_loss = criterion(
                    final_outputs[final_mask > 0], final_raw[final_mask > 0]
                )
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += total_batch_loss

            # Visualization (only first batch)
            if visualize and batch_idx == 0:
                # Ensure reconstruction exists
                if "reconstructed_outputs" not in locals():
                    output_patches_cat = torch.cat(all_output_patches, dim=0)
                    reconstructed_outputs = reconstruct_from_patches(
                        output_patches_cat, patch_locations1, raw1.shape, patch_size
                    )

                sample_output = model(sample) if sample is not None else None

                if speckle_module is not None:
                    titles = [
                        "Input Image",
                        "Flow Input",
                        "Flow Output",
                        "Blind Spot Input",
                        "Output Image",
                        "Sample Input",
                        "Sample Output",
                    ]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs[0][0].cpu().numpy(),
                        flow_outputs[0][0].cpu().numpy(),
                        blind1[0][0].cpu().numpy(),
                        reconstructed_outputs[0][0].cpu().numpy(),
                        sample.cpu().numpy()[0][0] if sample is not None else None,
                        sample_output.cpu().numpy()[0][0]
                        if sample_output is not None
                        else None,
                    ]
                    losses = {
                        "Flow Loss": flow_loss_value,
                        "Total Loss": total_batch_loss,
                    }
                else:
                    titles = [
                        "Input Image",
                        "Blind Spot Input",
                        "Output Image",
                        "Sample Input",
                        "Sample Output",
                    ]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        blind1[0][0].cpu().numpy(),
                        reconstructed_outputs[0][0].cpu().numpy(),
                        sample.cpu().numpy()[0][0] if sample is not None else None,
                        sample_output.cpu().numpy()[0][0]
                        if sample_output is not None
                        else None,
                    ]
                    losses = {"Total Loss": total_batch_loss}

                plot_images(images, titles, losses)

                from spdn.utils import evaluate_oct_denoising

                metrics = evaluate_oct_denoising(
                    raw1[0][0].cpu().numpy(), reconstructed_outputs[0][0].cpu().numpy()
                )

    return (
        total_loss / len(loader)
        if metrics is None
        else (total_loss / len(loader), metrics)
    )
