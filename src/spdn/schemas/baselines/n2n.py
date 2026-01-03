import time
import torch
from spdn.utils.eval_utils.visualise import plot_images


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
):
    mode = "train" if model.training else "val"

    epoch_loss = 0

    for batch_idx, (input_imgs, target_imgs) in enumerate(data_loader):
        input_imgs = input_imgs.to(device)
        target_imgs = target_imgs.to(device)

        if speckle_module is not None:
            flow_inputs = speckle_module(input_imgs)
            flow_inputs = flow_inputs["flow_component"].detach()
            flow_inputs = normalize_image_torch(flow_inputs)
            # flow_inputs = threshold_flow_component(flow_inputs, threshold=0.05)
            outputs = model(input_imgs)
            flow_outputs = speckle_module(outputs)
            flow_outputs = flow_outputs["flow_component"].detach()
            flow_outputs = normalize_image_torch(flow_outputs)
            # flow_outputs = threshold_flow_component(flow_outputs, threshold=0.05)
            flow_loss = torch.mean(torch.abs(flow_outputs - flow_inputs))

            loss = criterion(outputs, target_imgs) + flow_loss * alpha
        else:
            try:
                outputs = model(input_imgs)
                loss = criterion(outputs, target_imgs)
            except Exception as e:
                print(f"Error in model output: {e}")

            # physics_loss = lognormal_consistency_loss(outputs, target_imgs)
            # loss += physics_loss * 0.01

        if mode == "train":
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            scheduler.step(loss)

        epoch_loss += loss.item()

        if visualise and batch_idx % 10 == 0:
            assert input_imgs[0][0].shape == (256, 256)
            assert target_imgs[0][0].shape == (256, 256)
            assert outputs[0][0].shape == (256, 256)

            if speckle_module is not None:
                titles = [
                    "Input Image",
                    "Flow Input",
                    "Flow Output",
                    "Target Image",
                    "Output Image",
                ]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    flow_inputs[0][0].cpu().detach().numpy(),
                    flow_outputs[0][0].cpu().detach().numpy(),
                    target_imgs[0][0].cpu().numpy(),
                    outputs[0][0].cpu().detach().numpy(),
                ]
                losses = {"Flow Loss": flow_loss.item(), "Total Loss": loss.item()}
            else:
                titles = ["Input Image", "Target Image", "Output Image"]
                images = [
                    input_imgs[0][0].cpu().numpy(),
                    target_imgs[0][0].cpu().numpy(),
                    outputs[0][0].cpu().detach().numpy(),
                ]
                losses = {"Total Loss": loss.item()}

            plot_images(images, titles, losses)

    return epoch_loss / len(data_loader)


def train_n2n(
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
):
    last_checkpoint_path = checkpoint_path + "_last_checkpoint.pth"
    best_checkpoint_path = checkpoint_path + "_best_checkpoint.pth"

    print(f"Saving checkpoints to {best_checkpoint_path}")

    start_time = time.time()
    for epoch in range(starting_epoch, starting_epoch + epochs):
        model.train()
        visualise = False
        train_loss = process_batch(
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
        )

        model.eval()
        visualise = True
        with torch.no_grad():
            val_loss = process_batch(
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
