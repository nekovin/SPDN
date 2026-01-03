import torch


def blind_spot_masking(tensor, mask, kernel_size=5):
    b, c, h, w = tensor.shape
    masked_tensor = tensor.clone()

    # For each batch and channel
    for bi in range(b):
        for ci in range(c):
            # Get masked positions for this channel
            masked_positions = torch.nonzero(mask[bi, ci], as_tuple=True)

            if len(masked_positions[0]) == 0:
                continue

            # For each masked position
            for i in range(len(masked_positions[0])):
                y, x = masked_positions[0][i], masked_positions[1][i]

                # Extract neighborhood
                half_k = kernel_size // 2
                y_min, y_max = max(0, y - half_k), min(h, y + half_k + 1)
                x_min, x_max = max(0, x - half_k), min(w, x + half_k + 1)

                # Get neighborhood excluding center pixel
                neighborhood = tensor[bi, ci, y_min:y_max, x_min:x_max].flatten()

                # Calculate the index of the center pixel
                center_y, center_x = y - y_min, x - x_min
                center_idx = center_y * (x_max - x_min) + center_x

                # Create a mask to exclude the center pixel
                valid_indices = torch.ones(
                    neighborhood.shape[0], dtype=torch.bool, device=tensor.device
                )
                if 0 <= center_idx < valid_indices.shape[0]:
                    valid_indices[center_idx] = False

                # Select a random non-center pixel
                valid_neighborhood = neighborhood[valid_indices]
                if valid_neighborhood.shape[0] > 0:
                    # Random selection
                    rand_idx = torch.randint(
                        0, valid_neighborhood.shape[0], (1,), device=tensor.device
                    )
                    masked_tensor[bi, ci, y, x] = valid_neighborhood[rand_idx]

    return masked_tensor


def fast_blind_spot(tensor, mask, kernel_size=5):
    # Create output tensor
    masked_tensor = tensor.clone()

    # For each batch
    for b in range(tensor.size(0)):
        # Get all masked positions at once (across all channels)
        masked_positions = torch.nonzero(mask[b], as_tuple=False)

        if len(masked_positions) == 0:
            continue

        # For each masked position
        for pos in masked_positions:
            c, y, x = pos[0], pos[1], pos[2]

            # Get random position offset (-2 to 2 for kernel_size=5)
            offset = kernel_size // 2
            dy = torch.randint(-offset, offset + 1, (1,)).item()
            dx = torch.randint(-offset, offset + 1, (1,)).item()

            # Ensure we don't pick (0,0) offset
            if dy == 0 and dx == 0:
                dy = 1  # Simple fix: just move one pixel up

            # Sample from neighborhood with bounds checking
            ny, nx = y + dy, x + dx
            ny = max(0, min(ny, tensor.size(2) - 1))
            nx = max(0, min(nx, tensor.size(3) - 1))

            # Replace masked pixel
            masked_tensor[b, c, y, x] = tensor[b, c, ny, nx]

    return masked_tensor


def blind_spot_masking_fast(tensor, mask, kernel_size=5):
    device = tensor.device
    b, c, h, w = tensor.shape
    masked_tensor = tensor.clone()
    half_k = kernel_size // 2

    # Process all batches and channels in parallel
    # Get all masked positions
    for bi in range(b):
        for ci in range(c):
            y_coords, x_coords = torch.where(mask[bi, ci])

            if len(y_coords) == 0:
                continue

            # For efficiency, process in batches of masked pixels
            batch_size = 1000
            for i in range(0, len(y_coords), batch_size):
                y_batch = y_coords[i : i + batch_size]
                x_batch = x_coords[i : i + batch_size]

                for idx in range(len(y_batch)):
                    y, x = y_batch[idx].item(), x_batch[idx].item()

                    # Define patch boundaries
                    y_min, y_max = max(0, y - half_k), min(h, y + half_k + 1)
                    x_min, x_max = max(0, x - half_k), min(w, x + half_k + 1)

                    # Create patch mask excluding center
                    patch = tensor[bi, ci, y_min:y_max, x_min:x_max]
                    patch_mask = torch.ones_like(patch, dtype=torch.bool)
                    center_y, center_x = y - y_min, x - x_min
                    if (
                        0 <= center_y < patch_mask.shape[0]
                        and 0 <= center_x < patch_mask.shape[1]
                    ):
                        patch_mask[center_y, center_x] = False

                    # Get valid values and pick one randomly
                    valid_values = patch[patch_mask]
                    if len(valid_values) > 0:
                        idx = torch.randint(0, len(valid_values), (1,), device=device)
                        masked_tensor[bi, ci, y, x] = valid_values[idx]

    return masked_tensor


def subset_blind_spot_masking(tensor, mask_ratio=0.1, kernel_size=5):
    device = tensor.device
    b, c, h, w = tensor.shape
    masked_tensor = tensor.clone()
    half_k = kernel_size // 2

    mask = torch.rand(b, c, h, w, device=device) < mask_ratio

    padded = torch.nn.functional.pad(
        tensor, (half_k, half_k, half_k, half_k), mode="reflect"
    )
    neighborhoods = padded.unfold(2, kernel_size, 1).unfold(3, kernel_size, 1)

    center_mask = torch.ones(
        (kernel_size, kernel_size), dtype=torch.bool, device=device
    )
    center_mask[half_k, half_k] = False

    for bi in range(b):
        for ci in range(c):
            y_coords, x_coords = torch.where(mask[bi, ci])

            if len(y_coords) == 0:
                continue

            batch_size = 10000
            for i in range(0, len(y_coords), batch_size):
                y_batch = y_coords[i : i + batch_size]
                x_batch = x_coords[i : i + batch_size]
                batch_len = len(y_batch)

                pixel_neighborhoods = neighborhoods[bi, ci, y_batch, x_batch]

                masked_neighborhoods = pixel_neighborhoods.reshape(batch_len, -1)[
                    :, center_mask.reshape(-1)
                ]

                rand_indices = torch.randint(
                    0, masked_neighborhoods.shape[1], (batch_len,), device=device
                )
                selected_values = masked_neighborhoods[
                    torch.arange(batch_len, device=device), rand_indices
                ]

                masked_tensor[bi, ci, y_batch, x_batch] = selected_values

    return masked_tensor, mask
