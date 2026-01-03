import time
import torch
from spdn.utils.eval_utils.visualise import plot_images
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook


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


def create_partitioning_function(shape, n_partitions=2):
    height, width = shape

    def partition_function(i, j):
        return (i + j) % n_partitions

    return partition_function


def create_partition_masks(shape, n_partitions=2, device="cuda"):
    height, width = shape

    y_coords = torch.arange(height, device=device).view(-1, 1).repeat(1, width)
    x_coords = torch.arange(width, device=device).repeat(height, 1)

    coord_sum = (y_coords + x_coords) % n_partitions

    partition_masks = []
    for p in range(n_partitions):
        mask = (coord_sum == p).float()
        partition_masks.append(mask)

    return partition_masks


def create_random_partition_masks(shape, n_partitions=4, device="cuda"):
    height, width = shape
    all_indices = torch.arange(height * width, device=device)
    shuffled = all_indices[torch.randperm(len(all_indices))]

    partition_size = len(all_indices) // n_partitions
    masks = []

    for i in range(n_partitions):
        start_idx = i * partition_size
        end_idx = (
            start_idx + partition_size if i < n_partitions - 1 else len(all_indices)
        )

        mask = torch.zeros(height * width, device=device)
        mask[shuffled[start_idx:end_idx]] = 1
        mask = mask.reshape(height, width)
        masks.append(mask)

    return masks


def create_partition_masks_batch(shape, n_partitions, batch_size, device="cuda"):
    """Pre-compute masks for entire batch"""
    height, width = shape
    total_pixels = height * width

    # Create masks for entire batch at once
    masks = torch.zeros(batch_size, n_partitions, height, width, device=device)

    for b in range(batch_size):
        indices = torch.randperm(total_pixels, device=device)
        partition_size = total_pixels // n_partitions

        for p in range(n_partitions):
            start_idx = p * partition_size
            end_idx = (
                start_idx + partition_size if p < n_partitions - 1 else total_pixels
            )

            flat_mask = torch.zeros(total_pixels, device=device)
            flat_mask[indices[start_idx:end_idx]] = 1
            masks[b, p] = flat_mask.view(height, width)

    return masks


def _process_batch_n2s(
    data_loader,
    model,
    criterion,
    optimizer,
    epoch,
    epochs,
    device,
    visualise,
    speckle_module=None,
    alpha=1.0,
):
    mode = "train" if model.training else "val"

    epoch_loss = 0

    partition_masks = create_partition_masks((256, 256), n_partitions=2, device=device)

    for batch_idx, (input_imgs, _) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        batch_size, channels, height, width = input_imgs.shape

        total_loss = 0
        outputs = None

        for p in range(len(partition_masks)):
            curr_mask = (
                partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
            )

            comp_mask = 1 - curr_mask

            masked_input = input_imgs * comp_mask

            curr_outputs = model(masked_input)

            if p == 0:
                outputs = curr_outputs

            pred = curr_outputs * curr_mask
            target = input_imgs * curr_mask
            loss = criterion(pred, target)
            total_loss += loss

        loss = total_loss / len(partition_masks)

        full_output = model(input_imgs)

        if speckle_module is not None and outputs is not None:
            flow_inputs = speckle_module(input_imgs)
            flow_inputs = flow_inputs["flow_component"].detach()
            flow_inputs = normalize_image_torch(flow_inputs)

            flow_outputs = speckle_module(full_output)
            flow_outputs = flow_outputs["flow_component"].detach()
            flow_outputs = normalize_image_torch(flow_outputs)

            flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
            loss = loss + flow_loss * alpha

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(
                f"N2S {mode.capitalize()} Epoch [{epoch + 1}/{epochs}], Batch [{batch_idx + 1}/{len(data_loader)}], Loss: {loss.item():.6f}"
            )

        if visualise and batch_idx == 0 and outputs is not None:
            if speckle_module is not None:
                titles = ["Input Image", "Flow Input", "Flow Output", "Output Image"]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().detach().numpy(),
                    flow_outputs[0][0].cpu().detach().numpy(),
                    full_output[0][0].cpu().detach().numpy(),
                ]
                losses = {
                    "N2S Loss": loss.item()
                    - (flow_loss.item() * alpha if speckle_module else 0),
                    "Flow Loss": flow_loss.item() if speckle_module else 0,
                    "Total Loss": loss.item(),
                }
            else:
                titles = ["Input Image", "Output Image"]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    # outputs[0][0].cpu().detach().numpy()
                    full_output[0][0].cpu().detach().numpy(),
                ]
                losses = {"Total Loss": loss.item()}

            plot_images(images, titles, losses)

    return epoch_loss / len(data_loader)


def process_batch_n2s(
    data_loader,
    model,
    criterion,
    optimizer,
    epoch,
    epochs,
    device,
    visualise,
    speckle_module=None,
    alpha=1.0,
):
    from torch.cuda.amp import autocast, GradScaler

    mode = "train" if model.training else "val"
    epoch_loss = 0

    scaler = GradScaler() if mode == "train" else None

    # Standard N2S uses 2 partitions
    # partition_masks = create_partition_masks((256, 256), n_partitions=4, device=device)
    partition_masks = create_random_partition_masks(
        (256, 256), n_partitions=8, device=device
    )

    for batch_idx, (input_imgs, _) in enumerate(tqdm(data_loader)):
        input_imgs = input_imgs.to(device)

        with autocast():
            total_loss = 0
            final_output = torch.zeros_like(input_imgs)

            for p in range(len(partition_masks)):
                curr_mask = (
                    partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
                )
                comp_mask = 1 - curr_mask

                # Mask current partition pixels for input
                masked_input = input_imgs * comp_mask

                # Predict only the masked pixels
                curr_outputs = model(masked_input)

                # Accumulate predictions for final output
                final_output += curr_outputs * curr_mask

                # Calculate loss only for current partition
                pred = curr_outputs * curr_mask
                target = input_imgs * curr_mask
                loss = criterion(pred, target)
                total_loss += loss

            # Average loss across partitions
            loss = total_loss / len(partition_masks)

            # SSM loss if enabled
            if speckle_module is not None:
                flow_inputs = speckle_module(input_imgs)
                flow_inputs = flow_inputs["flow_component"].detach()
                flow_inputs = normalize_image_torch(flow_inputs)

                flow_outputs = speckle_module(final_output)
                flow_outputs = flow_outputs["flow_component"].detach()
                flow_outputs = normalize_image_torch(flow_outputs)

                flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                loss = loss + flow_loss * alpha

        if mode == "train":
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item()

        if visualise and batch_idx == 0:
            # Use final_output for visualization, not the partial outputs
            if speckle_module is not None:
                titles = ["Input Image", "Flow Input", "Flow Output", "Output Image"]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().detach().numpy(),
                    flow_outputs[0][0].cpu().detach().numpy(),
                    final_output[0][0].cpu().detach().numpy(),
                ]
                losses = {
                    "N2S Loss": loss.item()
                    - (flow_loss.item() * alpha if speckle_module else 0),
                    "Flow Loss": flow_loss.item() if speckle_module else 0,
                    "Total Loss": loss.item(),
                }
            else:
                titles = ["Input Image", "Output Image"]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    final_output[0][0].cpu().detach().numpy(),
                ]
                losses = {"Total Loss": loss.item()}

            plot_images(images, titles, losses)

    return epoch_loss / len(data_loader)


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


def process_batch_n2s_with_clean_inference(
    data_loader,
    model,
    criterion,
    optimizer,
    epoch,
    epochs,
    device,
    visualise,
    speckle_module=None,
    alpha=1.0,
):
    """
    N2S training with periodic clean inference training
    """
    from torch.cuda.amp import autocast, GradScaler

    mode = "train" if model.training else "val"
    epoch_loss = 0

    scaler = GradScaler() if mode == "train" else None

    partition_masks = create_random_partition_masks(
        (256, 256), n_partitions=8, device=device
    )

    for batch_idx, (input_imgs, _) in tqdm_notebook(enumerate(data_loader)):
        input_imgs = input_imgs.to(device)

        if mode == "train" and batch_idx % 10 == 0:
            with autocast():
                clean_output = model(input_imgs)

                final_output = torch.zeros_like(input_imgs)
                for p in range(len(partition_masks)):
                    curr_mask = (
                        partition_masks[p]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand_as(input_imgs)
                    )
                    comp_mask = 1 - curr_mask
                    masked_input = input_imgs * comp_mask
                    curr_outputs = model(masked_input)
                    final_output += curr_outputs * curr_mask

                consistency_loss = criterion(clean_output, final_output.detach())

                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(consistency_loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    consistency_loss.backward()
                    optimizer.step()

        with autocast():
            total_loss = 0
            final_output = torch.zeros_like(input_imgs)

            for p in range(len(partition_masks)):
                curr_mask = (
                    partition_masks[p].unsqueeze(0).unsqueeze(0).expand_as(input_imgs)
                )
                comp_mask = 1 - curr_mask
                masked_input = input_imgs * comp_mask
                curr_outputs = model(masked_input)
                final_output += curr_outputs * curr_mask

                pred = curr_outputs * curr_mask
                target = input_imgs * curr_mask
                loss = criterion(pred, target)
                total_loss += loss

            loss = total_loss / len(partition_masks)

            if speckle_module is not None:
                flow_inputs = speckle_module(input_imgs)["flow_component"].detach()
                flow_inputs = normalize_image_torch(flow_inputs)
                flow_outputs = speckle_module(final_output)["flow_component"].detach()
                flow_outputs = normalize_image_torch(flow_outputs)
                flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                loss = loss + flow_loss * alpha

        if mode == "train":
            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

        epoch_loss += loss.item()

        # Visualization with clean output comparison
        if visualise and batch_idx == 0:
            clean_output = model(input_imgs)

            titles = ["Input", "N2S Output", "Clean Output", "Clean vs N2S"]
            images = [
                input_imgs[0][0].cpu().numpy(),
                final_output[0][0].cpu().detach().numpy(),
                clean_output[0][0].cpu().detach().numpy(),
                (clean_output[0][0] - final_output[0][0]).abs().cpu().detach().numpy(),
            ]

            losses = {"Total Loss": loss.item()}
            plot_images(images, titles, losses)

    return epoch_loss / len(data_loader)


def train_n2s(
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
):
    last_checkpoint_path = checkpoint_path + "_last_checkpoint.pth"
    best_checkpoint_path = checkpoint_path + "_best_checkpoint.pth"

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()

    for epoch in tqdm_notebook(range(starting_epoch, starting_epoch + epochs)):
        model.train()
        # train_loss = process_batch_n2s(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        # train_loss = process_batch_n2s_with_clean_inference(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        train_loss, _ = process_batch_n2s_patch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            speckle_module=speckle_module,
            visualize=visualise,
            alpha=alpha,
        )

        model.eval()
        with torch.no_grad():
            # val_loss = process_batch_n2s(val_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
            # val_loss = process_batch_n2s_with_clean_inference(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
            val_loss, _ = process_batch_n2s_patch(
                model,
                val_loader,
                criterion,
                optimizer,
                device=device,
                speckle_module=speckle_module,
                visualize=visualise,
                alpha=alpha,
            )

        print(
            f"Epoch [{starting_epoch + epoch + 1}/{epochs}], Average Loss: {train_loss:.6f}"
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


from contextlib import nullcontext


def train_n2s_patch(
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
    sample=None,
    train_config=None,
    best_metrics_score=float("-inf"),
    patch_size=64,
    stride=32,
    n_partitions=2,
    adaptive_loss=False,
):
    last_checkpoint_path = checkpoint_path + "_patched_last_checkpoint.pth"
    best_checkpoint_path = checkpoint_path + "_patched_best_checkpoint.pth"
    best_metrics_checkpoint_path = (
        checkpoint_path + "_patched_best_metrics_checkpoint.pth"
    )

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()

    for epoch in tqdm_notebook(range(starting_epoch, starting_epoch + epochs)):
        print(f"Epoch {epoch + 1}/{epochs}")
        model.train()
        # train_loss = process_batch_n2s(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        # train_loss = process_batch_n2s_with_clean_inference(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
        train_loss, _ = process_batch_n2s_patch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            speckle_module=speckle_module,
            visualize=False,
            alpha=alpha,
            scheduler=None,
            sample=sample,
            patch_size=patch_size,
            stride=stride,
            n_partitions=n_partitions,
            adaptive_loss=adaptive_loss,
        )

        model.eval()
        with torch.no_grad():
            # val_loss = process_batch_n2s(val_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
            # val_loss = process_batch_n2s_with_clean_inference(train_loader, model, criterion, optimizer, epoch, epochs, device, visualise, speckle_module, alpha)
            val_loss, val_metrics = process_batch_n2s_patch(
                model,
                val_loader,
                criterion,
                optimizer=None,
                device=device,
                speckle_module=speckle_module,
                visualize=visualise,
                alpha=alpha,
                scheduler=scheduler,
                sample=sample,
                patch_size=patch_size,
                stride=stride,
                n_partitions=n_partitions,
                adaptive_loss=adaptive_loss,
            )

            val_metrics_score = (
                val_metrics.get("snr", 0) * 0.3
                + val_metrics.get("cnr", 0) * 0.3
                + val_metrics.get("enl", 0) * 0.2
                + val_metrics.get("epi", 0) * 0.2
            )

        print(
            f"Epoch [{starting_epoch + epoch + 1}/{epochs}], Average Loss: {train_loss:.6f}"
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

    elapsed_time = time.time() - start_time
    print(f"Training completed in {elapsed_time / 60:.2f} minutes")

    return model


from spdn.utils.data_utils.patch_processing import (
    extract_patches,
    reconstruct_from_patches,
)


def _process_batch_n2s_patch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
    n_partitions=4,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    metrics = None

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            raw1, _ = batch
            raw1 = raw1.to(device)

            # Extract patches
            raw1_patches, patch_locations1 = extract_patches(raw1, patch_size, stride)

            # Create partition masks for patch size
            partition_masks = create_random_partition_masks(
                (patch_size, patch_size), n_partitions=n_partitions, device=device
            )

            # Process patches in sub-batches
            sub_batch_size = 8
            batch_loss = 0.0
            all_output1_patches = []
            accumulated_loss = None

            for i in range(0, len(raw1_patches), sub_batch_size):
                raw1_sub_batch = raw1_patches[i : i + sub_batch_size]

                # N2S processing for each partition
                total_n2s_loss = 0
                final_output = torch.zeros_like(raw1_sub_batch)

                for p in range(len(partition_masks)):
                    curr_mask = (
                        partition_masks[p]
                        .unsqueeze(0)
                        .unsqueeze(0)
                        .expand_as(raw1_sub_batch)
                    )
                    comp_mask = 1 - curr_mask
                    masked_input = raw1_sub_batch * comp_mask

                    curr_outputs = model(masked_input)
                    final_output += curr_outputs * curr_mask

                    pred = curr_outputs * curr_mask
                    target = raw1_sub_batch * curr_mask
                    loss = criterion(pred, target)
                    total_n2s_loss += loss

                n2s_loss = total_n2s_loss / len(partition_masks)
                all_output1_patches.append(final_output.detach())

                # Add speckle module loss if provided
                if speckle_module is not None:
                    flow_inputs = speckle_module(raw1_sub_batch)[
                        "flow_component"
                    ].detach()
                    flow_inputs = normalize_image_torch(flow_inputs)
                    flow_outputs = speckle_module(final_output)[
                        "flow_component"
                    ].detach()
                    flow_outputs = normalize_image_torch(flow_outputs)
                    flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))

                    sub_loss = n2s_loss + flow_loss * alpha
                else:
                    sub_loss = n2s_loss

                batch_loss += sub_loss.item() * len(raw1_sub_batch)

                # Accumulate gradients only during training
                if optimizer:
                    if accumulated_loss is None:
                        accumulated_loss = sub_loss
                    else:
                        accumulated_loss = accumulated_loss + sub_loss

            # Perform backprop once per batch, only during training
            if optimizer is not None and accumulated_loss is not None:
                optimizer.zero_grad()
                accumulated_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += batch_loss / len(raw1_patches)

            # Visualization
            if visualize and batch_idx == 0:
                output1_patches = torch.cat(all_output1_patches, dim=0)
                reconstructed_outputs1 = reconstruct_from_patches(
                    output1_patches, patch_locations1, raw1.shape, patch_size
                )

                sample_output = None
                if sample is not None:
                    model_was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        sample_output = model(sample)
                    if model_was_training:
                        model.train()

                # Filter out None values for plotting
                if speckle_module is not None:
                    titles = ["Input Image", "Flow Input", "Flow Output", "N2S Output"]
                    flow_inputs_full = speckle_module(raw1)["flow_component"].detach()
                    flow_outputs_full = speckle_module(reconstructed_outputs1)[
                        "flow_component"
                    ].detach()

                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs_full[0][0].cpu().numpy(),
                        flow_outputs_full[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {
                        "Flow Loss": flow_loss.item(),
                        "Total Loss": batch_loss / len(raw1_patches),
                    }
                else:
                    titles = ["Input Image", "N2S Output"]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {"Total Loss": batch_loss / len(raw1_patches)}

                plot_images(images, titles, losses)

                from spdn.utils import evaluate_oct_denoising

                metrics = evaluate_oct_denoising(
                    raw1[0][0].cpu().numpy(), reconstructed_outputs1[0][0].cpu().numpy()
                )

    return total_loss / len(loader), metrics


def _process_batch_n2s_patch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
    n_partitions=2,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    metrics = None

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            print(f"Processing batch {batch_idx + 1}/{len(loader)}")
            raw1, _ = batch
            raw1 = raw1.to(device)

            # Extract patches
            raw1_patches, patch_locations1 = extract_patches(raw1, patch_size, stride)

            # Create partition masks once
            partition_masks = create_random_partition_masks(
                (patch_size, patch_size), n_partitions=n_partitions, device=device
            )

            # Pre-compute speckle flow inputs if needed
            flow_inputs_cache = None
            if speckle_module is not None:
                with torch.no_grad():
                    flow_inputs_cache = speckle_module(raw1_patches)[
                        "flow_component"
                    ].detach()
                    flow_inputs_cache = normalize_image_torch(flow_inputs_cache)

            # Process all patches at once with vectorized operations
            batch_loss = 0.0

            # Vectorized N2S processing
            n_patches = raw1_patches.shape[0]

            # Expand masks to match patch batch size
            expanded_masks = torch.stack(
                [
                    mask.unsqueeze(0).expand(n_patches, -1, -1)
                    for mask in partition_masks
                ]
            )  # Shape: [n_partitions, n_patches, H, W]

            # Create complementary masks
            comp_masks = 1 - expanded_masks

            # Process all partitions simultaneously
            all_masked_inputs = []
            all_targets = []

            for p in range(n_partitions):
                curr_mask = expanded_masks[p].unsqueeze(1)  # Add channel dim
                comp_mask = comp_masks[p].unsqueeze(1)

                masked_input = raw1_patches * comp_mask
                target = raw1_patches * curr_mask

                all_masked_inputs.append(masked_input)
                all_targets.append(target)

            # Batch process all masked inputs
            all_masked_batch = torch.cat(all_masked_inputs, dim=0)
            all_outputs_batch = model(all_masked_batch)

            # Split outputs back by partition
            outputs_by_partition = torch.split(all_outputs_batch, n_patches, dim=0)

            # Compute N2S loss and reconstruct final output
            final_output = torch.zeros_like(raw1_patches)
            total_n2s_loss = 0

            for p, (output, target, mask) in enumerate(
                zip(outputs_by_partition, all_targets, expanded_masks)
            ):
                mask = mask.unsqueeze(1)
                pred = output * mask
                final_output += pred

                loss = criterion(pred, target)
                total_n2s_loss += loss

            n2s_loss = total_n2s_loss / n_partitions

            # Speckle module loss (if enabled)
            if speckle_module is not None:
                with torch.no_grad():
                    flow_outputs = speckle_module(final_output)[
                        "flow_component"
                    ].detach()
                    flow_outputs = normalize_image_torch(flow_outputs)
                    flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs_cache))

                total_batch_loss = n2s_loss + flow_loss * alpha
            else:
                total_batch_loss = n2s_loss

            batch_loss = total_batch_loss.item()

            # Backpropagation (training only)
            if optimizer is not None:
                optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += batch_loss

            # Visualization (first batch only)
            if visualize:
                reconstructed_outputs1 = reconstruct_from_patches(
                    final_output, patch_locations1, raw1.shape, patch_size
                )

                reconstructed_outputs1 = reconstructed_outputs1.detach().cpu()

                sample_output = None
                if sample is not None:
                    model_was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        sample_output = model(sample)
                    if model_was_training:
                        model.train()

                # Visualization setup
                if speckle_module is not None:
                    titles = ["Input Image", "Flow Input", "Flow Output", "N2S Output"]
                    flow_inputs_full = speckle_module(raw1)["flow_component"].detach()
                    flow_outputs_full = speckle_module(reconstructed_outputs1)[
                        "flow_component"
                    ].detach()

                    # detach

                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs_full[0][0].cpu().numpy(),
                        flow_outputs_full[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {"Flow Loss": flow_loss.item(), "Total Loss": batch_loss}
                else:
                    titles = ["Input Image", "N2S Output"]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {"Total Loss": batch_loss}

                plot_images(images, titles, losses)

                from spdn.utils import evaluate_oct_denoising

                metrics = evaluate_oct_denoising(
                    raw1[0][0].cpu().numpy(), reconstructed_outputs1[0][0].cpu().numpy()
                )

    return total_loss / len(loader), metrics


def _process_batch_n2s_patch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
    n_partitions=2,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    metrics = None

    # Pre-compute checkerboard masks once
    y_coords, x_coords = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing="ij",
    )
    mask1 = (
        ((y_coords + x_coords) % 2).float().unsqueeze(0).unsqueeze(0)
    )  # [1, 1, H, W]
    mask2 = 1 - mask1

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            print(f"Processing batch {batch_idx + 1}/{len(loader)}")
            raw1, _ = batch
            raw1 = raw1.to(device)

            # Extract patches
            raw1_patches, patch_locations = extract_patches(raw1, patch_size, stride)
            n_patches = raw1_patches.shape[0]

            # Pre-compute speckle flow inputs if needed
            flow_inputs_cache = None
            if speckle_module is not None:
                with torch.no_grad():
                    flow_inputs_cache = speckle_module(raw1_patches)[
                        "flow_component"
                    ].detach()
                    flow_inputs_cache = normalize_image_torch(flow_inputs_cache)

            # Expand masks to batch size
            masks = torch.cat(
                [
                    mask1.expand(n_patches, -1, -1, -1),
                    mask2.expand(n_patches, -1, -1, -1),
                ]
            )  # [2*n_patches, 1, H, W]

            # Create masked inputs and targets
            patches_doubled = torch.cat(
                [raw1_patches, raw1_patches]
            )  # [2*n_patches, 1, H, W]
            masked_inputs = patches_doubled * (1 - masks)
            targets = patches_doubled * masks

            # Single forward pass for all masked inputs
            outputs = model(masked_inputs)

            # Apply masks to outputs and compute loss
            predicted_masked = outputs * masks
            n2s_loss = criterion(predicted_masked, targets)

            # Reconstruct final output by combining both partitions
            pred1, pred2 = torch.split(predicted_masked, n_patches, dim=0)
            final_output = pred1 + pred2

            total_batch_loss = n2s_loss

            # Speckle module loss (if enabled)
            if speckle_module is not None:
                with torch.no_grad():
                    flow_outputs = speckle_module(final_output)[
                        "flow_component"
                    ].detach()
                    flow_outputs = normalize_image_torch(flow_outputs)
                    flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs_cache))

                total_batch_loss = n2s_loss + flow_loss * alpha

            # Backpropagation (training only)
            if optimizer is not None:
                optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += total_batch_loss.item()

            # Visualization (first batch only)
            if visualize:
                reconstructed_outputs1 = reconstruct_from_patches(
                    final_output, patch_locations, raw1.shape, patch_size
                )
                reconstructed_outputs1 = reconstructed_outputs1.detach().cpu()

                sample_output = None
                if sample is not None:
                    model_was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        sample_output = model(sample)
                    if model_was_training:
                        model.train()

                # Visualization setup
                if speckle_module is not None:
                    titles = ["Input Image", "Flow Input", "Flow Output", "N2S Output"]
                    flow_inputs_full = speckle_module(raw1)["flow_component"].detach()
                    # flow_outputs_full = speckle_module(reconstructed_outputs1)['flow_component'].detach()
                    flow_outputs_full = speckle_module(
                        reconstructed_outputs1.to(device)
                    )["flow_component"].detach()

                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs_full[0][0].cpu().numpy(),
                        flow_outputs_full[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {
                        "Flow Loss": flow_loss.item(),
                        "Total Loss": total_batch_loss.item(),
                    }
                else:
                    titles = ["Input Image", "N2S Output"]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {"Total Loss": total_batch_loss.item()}

                plot_images(images, titles, losses)

                from spdn.utils import evaluate_oct_denoising

                metrics = evaluate_oct_denoising(
                    raw1[0][0].cpu().numpy(), reconstructed_outputs1[0][0].cpu().numpy()
                )

    return total_loss / len(loader), metrics


def _process_batch_n2s_patch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
    n_partitions=2,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    metrics = None

    y_coords, x_coords = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing="ij",
    )
    mask1 = ((y_coords + x_coords) % 2).float().unsqueeze(0).unsqueeze(0)
    mask2 = 1 - mask1

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            print(f"Processing batch {batch_idx + 1}/{len(loader)}")
            raw1, _ = batch
            raw1 = raw1.to(device)

            # Extract patches
            raw1_patches, patch_locations = extract_patches(raw1, patch_size, stride)
            n_patches = raw1_patches.shape[0]

            # Expand masks to batch size
            masks = torch.cat(
                [
                    mask1.expand(n_patches, -1, -1, -1),
                    mask2.expand(n_patches, -1, -1, -1),
                ]
            )  # [2*n_patches, 1, H, W]

            # Create masked inputs and targets
            patches_doubled = torch.cat(
                [raw1_patches, raw1_patches]
            )  # [2*n_patches, 1, H, W]
            masked_inputs = patches_doubled * (1 - masks)
            targets = patches_doubled * masks

            # Single forward pass for all masked inputs
            outputs = model(masked_inputs)

            # Apply masks to outputs and compute loss
            predicted_masked = outputs * masks
            n2s_loss = criterion(predicted_masked, targets)

            # Reconstruct final output by combining both partitions
            pred1, pred2 = torch.split(predicted_masked, n_patches, dim=0)
            final_output = pred1 + pred2

            total_batch_loss = n2s_loss

            # Speckle module loss (if enabled) - EXACTLY like N2V
            if speckle_module is not None:
                flow_inputs = speckle_module(raw1_patches)["flow_component"].detach()
                flow_inputs = normalize_image_torch(flow_inputs)

                flow_outputs = speckle_module(final_output)["flow_component"].detach()
                flow_outputs = normalize_image_torch(flow_outputs)

                flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                total_batch_loss = n2s_loss + flow_loss * alpha

            # Backpropagation (training only)
            if optimizer is not None:
                optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += total_batch_loss.item()

            # Visualization (first batch only)
            if visualize:
                reconstructed_outputs1 = reconstruct_from_patches(
                    final_output, patch_locations, raw1.shape, patch_size
                )
                reconstructed_outputs1 = reconstructed_outputs1.detach().cpu()

                sample_output = None
                if sample is not None:
                    model_was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        sample_output = model(sample)
                    if model_was_training:
                        model.train()

                # Visualization setup
                if speckle_module is not None:
                    titles = ["Input Image", "Flow Input", "Flow Output", "N2S Output"]
                    flow_inputs_full = speckle_module(raw1)["flow_component"].detach()
                    flow_outputs_full = speckle_module(
                        reconstructed_outputs1.to(device)
                    )["flow_component"].detach()

                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs_full[0][0].cpu().numpy(),
                        flow_outputs_full[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {
                        "Flow Loss": flow_loss.item(),
                        "Total Loss": total_batch_loss.item(),
                    }
                else:
                    titles = ["Input Image", "N2S Output"]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {"Total Loss": total_batch_loss.item()}

                plot_images(images, titles, losses)

                from spdn.utils import evaluate_oct_denoising

                metrics = evaluate_oct_denoising(
                    raw1[0][0].cpu().numpy(), reconstructed_outputs1[0][0].cpu().numpy()
                )

    return total_loss / len(loader), metrics


def process_batch_n2s_patch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
    n_partitions=2,
    adaptive_loss=True,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    metrics = None

    # Pre-compute checkerboard masks once
    y_coords, x_coords = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing="ij",
    )
    mask1 = (
        ((y_coords + x_coords) % 2).float().unsqueeze(0).unsqueeze(0)
    )  # [1, 1, H, W]
    mask2 = 1 - mask1

    context_manager = torch.no_grad() if not optimizer else nullcontext()

    with context_manager:
        for batch_idx, batch in enumerate(tqdm(loader)):
            print(f"Processing batch {batch_idx + 1}/{len(loader)}")
            raw1, _ = batch
            raw1 = raw1.to(device)

            # Extract patches
            raw1_patches, patch_locations = extract_patches(raw1, patch_size, stride)
            n_patches = raw1_patches.shape[0]

            # Process patches in sub-batches
            sub_batch_size = 32
            batch_loss = 0.0
            all_output_patches = [] if visualize else None

            # Clear gradients once per batch
            if optimizer is not None:
                optimizer.zero_grad()

            for i in range(0, n_patches, sub_batch_size):
                current_batch_size = min(sub_batch_size, n_patches - i)
                patch_sub_batch = raw1_patches[i : i + current_batch_size]

                # Get masks for current sub-batch
                mask1_batch = mask1.expand(current_batch_size, -1, -1, -1)
                mask2_batch = mask2.expand(current_batch_size, -1, -1, -1)

                # Process partition 1
                masked_input1 = patch_sub_batch * (1 - mask1_batch)
                output1 = model(masked_input1)
                pred1 = output1 * mask1_batch

                # Process partition 2
                masked_input2 = patch_sub_batch * (1 - mask2_batch)
                output2 = model(masked_input2)
                pred2 = output2 * mask2_batch

                # Combine predictions
                final_output = pred1 + pred2
                if visualize:
                    all_output_patches.append(final_output.detach())

                # Compute N2S loss
                target1 = patch_sub_batch * mask1_batch
                target2 = patch_sub_batch * mask2_batch
                n2s_loss = criterion(pred1, target1) + criterion(pred2, target2)

                sub_loss = n2s_loss

                # Speckle module loss (if enabled)
                if speckle_module is not None:
                    with torch.no_grad():
                        flow_inputs = speckle_module(patch_sub_batch)["flow_component"]
                        flow_inputs = normalize_image_torch(flow_inputs)

                    flow_outputs = speckle_module(final_output)["flow_component"]
                    flow_outputs = normalize_image_torch(flow_outputs)

                    flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))
                    # sub_loss = n2s_loss + flow_loss * alpha

                    if adaptive_loss:
                        alpha_adaptive = n2s_loss.detach() / (flow_loss.detach() + 1e-8)
                        sub_loss = n2s_loss + flow_loss * alpha_adaptive
                    else:
                        sub_loss = n2s_loss + flow_loss * alpha

                batch_loss += sub_loss.item()

                # Backpropagation (accumulate gradients)
                if optimizer is not None:
                    sub_loss.backward()

            # Single optimizer step per batch
            if optimizer is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += batch_loss

            # Visualization (first batch only)
            if visualize and batch_idx == 0:
                # Reconstructed outputs only computed when needed
                output_patches = torch.cat(all_output_patches, dim=0)
                reconstructed_outputs1 = reconstruct_from_patches(
                    output_patches, patch_locations, raw1.shape, patch_size
                )

                sample_output = None
                if sample is not None:
                    model_was_training = model.training
                    model.eval()
                    with torch.no_grad():
                        sample_output = model(sample)
                    if model_was_training:
                        model.train()

                # Visualization setup
                if speckle_module is not None:
                    with torch.no_grad():
                        flow_inputs_full = speckle_module(raw1)["flow_component"]
                        flow_outputs_full = speckle_module(reconstructed_outputs1)[
                            "flow_component"
                        ]

                    titles = ["Input Image", "Flow Input", "Flow Output", "N2S Output"]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        flow_inputs_full[0][0].cpu().numpy(),
                        flow_outputs_full[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {
                        "Flow Loss": flow_loss.item() if speckle_module else 0,
                        "Total Loss": batch_loss / n_patches,
                    }
                else:
                    titles = ["Input Image", "N2S Output"]
                    images = [
                        raw1[0][0].cpu().numpy(),
                        reconstructed_outputs1[0][0].cpu().numpy(),
                    ]

                    if sample is not None and sample_output is not None:
                        titles.extend(["Sample Input", "Sample Output"])
                        images.extend(
                            [
                                sample.cpu().numpy()[0][0],
                                sample_output[0][0].cpu().numpy(),
                            ]
                        )

                    losses = {"Total Loss": batch_loss / n_patches}

                plot_images(images, titles, losses)

                from spdn.utils import evaluate_oct_denoising

                metrics = evaluate_oct_denoising(
                    raw1[0][0].cpu().numpy(), reconstructed_outputs1[0][0].cpu().numpy()
                )

    return total_loss / len(loader), metrics


def process_batch_n2s_patch(
    model,
    loader,
    criterion,
    optimizer=None,
    device="cuda",
    speckle_module=None,
    visualize=False,
    alpha=1.0,
    scheduler=None,
    sample=None,
    patch_size=64,
    stride=32,
    n_partitions=2,
    adaptive_loss=True,
):
    if optimizer:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    metrics = None

    # Pre-compute checkerboard masks once
    y_coords, x_coords = torch.meshgrid(
        torch.arange(patch_size, device=device),
        torch.arange(patch_size, device=device),
        indexing="ij",
    )
    mask1 = ((y_coords + x_coords) % 2).float().unsqueeze(0).unsqueeze(0)
    mask2 = 1 - mask1

    for batch_idx, batch in enumerate(tqdm(loader)):
        raw1, _ = batch
        raw1 = raw1.to(device)

        # Pre-compute flow input once per batch (computational optimization)
        flow_inputs = None
        if speckle_module is not None:
            with torch.no_grad():
                flow_inputs = speckle_module(raw1)["flow_component"]
                flow_inputs = normalize_image_torch(flow_inputs)

        # Extract patches
        raw1_patches, patch_locations = extract_patches(raw1, patch_size, stride)
        n_patches = raw1_patches.shape[0]

        # Process patches
        sub_batch_size = 32
        batch_n2s_loss = 0.0
        all_output_patches = []
        final_pred1 = None
        final_pred2 = None
        final_target1 = None
        final_target2 = None

        if optimizer is not None:
            optimizer.zero_grad()

        for i in range(0, n_patches, sub_batch_size):
            current_batch_size = min(sub_batch_size, n_patches - i)
            patch_sub_batch = raw1_patches[i : i + current_batch_size]

            # Get masks for current sub-batch
            mask1_batch = mask1.expand(current_batch_size, -1, -1, -1)
            mask2_batch = mask2.expand(current_batch_size, -1, -1, -1)

            # Process partitions
            masked_input1 = patch_sub_batch * (1 - mask1_batch)
            output1 = model(masked_input1)
            pred1 = output1 * mask1_batch

            masked_input2 = patch_sub_batch * (1 - mask2_batch)
            output2 = model(masked_input2)
            pred2 = output2 * mask2_batch

            # Combine predictions
            final_output = pred1 + pred2
            all_output_patches.append(final_output.detach().clone())

            # Keep final outputs for gradient computation if training
            if optimizer is not None:
                final_pred1 = pred1
                final_pred2 = pred2
                final_target1 = patch_sub_batch * mask1_batch
                final_target2 = patch_sub_batch * mask2_batch

            # Accumulate N2S loss (scalar computation for efficiency)
            target1 = patch_sub_batch * mask1_batch
            target2 = patch_sub_batch * mask2_batch
            n2s_loss = criterion(pred1, target1) + criterion(pred2, target2)
            batch_n2s_loss += n2s_loss.item() * current_batch_size

        # Compute average N2S loss
        avg_n2s_loss = batch_n2s_loss / n_patches

        # Flow computation (only if speckle module exists)
        flow_loss_value = 0.0
        if speckle_module is not None:
            # Reconstruct full image from patches
            output_patches_cat = torch.cat(all_output_patches, dim=0)
            reconstructed_outputs = reconstruct_from_patches(
                output_patches_cat, patch_locations, raw1.shape, patch_size
            )

            # Single flow computation on reconstructed image
            with torch.no_grad():
                flow_outputs = speckle_module(reconstructed_outputs)["flow_component"]
                flow_outputs = normalize_image_torch(flow_outputs)
                flow_loss_value = torch.mean(
                    torch.abs(flow_inputs - flow_outputs)
                ).item()

        # Compute total loss
        if adaptive_loss and speckle_module is not None:
            # print("Using adaptive loss scaling")
            alpha_adaptive = avg_n2s_loss / (flow_loss_value + 1e-8)
            total_batch_loss = avg_n2s_loss + flow_loss_value * alpha_adaptive
        else:
            total_batch_loss = avg_n2s_loss + flow_loss_value * alpha

        # Gradient computation (only during training)
        if optimizer is not None and final_pred1 is not None:
            # Compute loss with gradients for backprop on final sub-batch
            if speckle_module is not None:
                # Recompute flow loss with gradients for final batch
                final_combined = final_pred1 + final_pred2

                with torch.no_grad():
                    flow_inputs_patch = speckle_module(final_combined.detach())[
                        "flow_component"
                    ]
                    flow_inputs_patch = normalize_image_torch(flow_inputs_patch)

                flow_outputs_patch = speckle_module(final_combined)["flow_component"]
                flow_outputs_patch = normalize_image_torch(flow_outputs_patch)
                flow_loss_grad = torch.mean(
                    torch.abs(flow_outputs_patch - flow_inputs_patch)
                )

                if adaptive_loss:
                    # print("Using adaptive loss scaling")
                    n2s_loss_grad = criterion(final_pred1, final_target1) + criterion(
                        final_pred2, final_target2
                    )
                    alpha_adaptive = n2s_loss_grad.detach() / (
                        flow_loss_grad.detach() + 1e-8
                    )
                    train_loss = n2s_loss_grad + flow_loss_grad * alpha_adaptive
                else:
                    train_loss = (
                        criterion(final_pred1, final_target1)
                        + criterion(final_pred2, final_target2)
                        + flow_loss_grad * alpha
                    )
            else:
                train_loss = criterion(final_pred1, final_target1) + criterion(
                    final_pred2, final_target2
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
                    output_patches_cat, patch_locations, raw1.shape, patch_size
                )

            sample_output = model(sample) if sample is not None else None

            if speckle_module is not None:
                titles = [
                    "Input Image",
                    "Flow Input",
                    "Flow Output",
                    "N2S Output",
                    "Sample Input",
                    "Sample Output",
                ]
                images = [
                    raw1[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().numpy(),
                    flow_outputs[0][0].cpu().numpy(),
                    reconstructed_outputs[0][0].cpu().numpy(),
                    sample.cpu().numpy()[0][0] if sample is not None else None,
                    sample_output.cpu().numpy()[0][0]
                    if sample_output is not None
                    else None,
                ]
                losses = {"Flow Loss": flow_loss_value, "Total Loss": total_batch_loss}
            else:
                titles = ["Input Image", "N2S Output", "Sample Input", "Sample Output"]
                images = [
                    raw1[0][0].cpu().numpy(),
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

    # Always return a tuple to match the unpacking expectation
    return (
        (total_loss / len(loader), metrics)
        if metrics is not None
        else (total_loss / len(loader), None)
    )
