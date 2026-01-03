import torch


def extract_patches(image, patch_size=64, stride=32):
    """Extract patches from an image with given patch size and stride."""
    stride = patch_size // 4
    # Handle different image formats
    if len(image.shape) == 4:  # (B, C, H, W)
        b, c, h, w = image.shape
    elif len(image.shape) == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension
        b, c, h, w = image.shape
    elif len(image.shape) == 2:  # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        b, c, h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    all_patches = []
    all_locations = []

    for i in range(b):
        img_patches = []
        img_locations = []

        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[i, :, y : y + patch_size, x : x + patch_size]
                img_patches.append(patch)
                img_locations.append((y, x))

        all_patches.extend(img_patches)
        all_locations.extend([(i, y, x) for y, x in img_locations])

    return torch.stack(all_patches), all_locations


def extract_patches(image, patch_size, stride):
    if len(image.shape) == 4:  # (B, C, H, W)
        b, c, h, w = image.shape
    elif len(image.shape) == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension
        b, c, h, w = image.shape
    elif len(image.shape) == 2:  # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        b, c, h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    all_patches = []
    all_locations = []

    for i in range(b):
        img_patches = []
        img_locations = []

        # Regular grid sampling
        for y in range(0, h - patch_size + 1, stride):
            for x in range(0, w - patch_size + 1, stride):
                patch = image[i, :, y : y + patch_size, x : x + patch_size]
                img_patches.append(patch)
                img_locations.append((y, x))

        # Additional sampling for bottom edge regions
        # Using half the stride to increase sampling density at the bottom
        bottom_start = max(0, h - patch_size - stride)
        for y in range(bottom_start, h - patch_size + 1, stride // 2):
            for x in range(0, w - patch_size + 1, stride):
                # Skip if this patch was already included in regular sampling
                if y % stride == 0:
                    continue
                patch = image[i, :, y : y + patch_size, x : x + patch_size]
                img_patches.append(patch)
                img_locations.append((y, x))

        all_patches.extend(img_patches)
        all_locations.extend([(i, y, x) for y, x in img_locations])

    return torch.stack(all_patches), all_locations


def _reconstruct_from_patches(patches, locations, image_shape, patch_size):
    """Reconstruct an image from patches based on their locations."""
    # Handle different shape formats
    if len(image_shape) == 4:  # (B, C, H, W)
        b, c, h, w = image_shape
    elif len(image_shape) == 3:  # (C, H, W)
        c, h, w = image_shape
        b = 1
    else:
        raise ValueError(f"Unexpected image shape: {image_shape}")

    # Check location format
    sample_location = locations[0]
    if len(sample_location) == 3:  # (batch_idx, y, x)
        # Need to group by batch
        batch_reconstructed = []

        # Group patches by batch index
        batch_patches = [[] for _ in range(b)]
        batch_locations = [[] for _ in range(b)]

        for i, (patch, location) in enumerate(zip(patches, locations)):
            batch_idx, y, x = location
            batch_patches[batch_idx].append(patch)
            batch_locations[batch_idx].append((y, x))

        # Reconstruct each batch item
        for i in range(b):
            if len(batch_patches[i]) > 0:
                # Call recursively with simpler locations
                reconstructed = reconstruct_from_patches(
                    torch.stack(batch_patches[i]),
                    batch_locations[i],
                    (c, h, w),  # Single image shape
                    patch_size,
                )
                batch_reconstructed.append(reconstructed)
            else:
                # Empty tensor if no patches for this batch
                batch_reconstructed.append(
                    torch.zeros((c, h, w), device=patches[0].device)
                )

        return torch.stack(batch_reconstructed)

    elif len(sample_location) == 2:  # (y, x)
        # Simple reconstruction for a single image
        reconstructed = torch.zeros((c, h, w), device=patches[0].device)
        torch.zeros((h, w), device=patches[0].device)

        # Add extra padding around image edges
        pad_size = patch_size // 2
        padded_reconstructed = torch.zeros(
            (c, h + 2 * pad_size, w + 2 * pad_size), device=patches[0].device
        )
        padded_weights = torch.zeros(
            (h + 2 * pad_size, w + 2 * pad_size), device=patches[0].device
        )

        for patch, (y, x) in zip(patches, locations):
            # Add patches to padded reconstruction at offset positions
            padded_y = y + pad_size
            padded_x = x + pad_size
            padded_reconstructed[
                :, padded_y : padded_y + patch_size, padded_x : padded_x + patch_size
            ] += patch
            padded_weights[
                padded_y : padded_y + patch_size, padded_x : padded_x + patch_size
            ] += 1

        # Handle edge weighting - add extra virtual patches for corners and edges
        # This ensures corners get similar weight averaging as center regions
        for y in range(0, pad_size, patch_size // 2):
            for x in range(0, w + pad_size, patch_size // 2):
                # Add weight for top edge
                if y < h and padded_weights[y, pad_size + x] > 0:
                    padded_weights[0:pad_size, pad_size + x] += 1

                # Add weight for bottom edge
                if y < h and padded_weights[h + pad_size - 1 - y, pad_size + x] > 0:
                    padded_weights[h + pad_size : h + 2 * pad_size, pad_size + x] += 1

        for x in range(0, pad_size, patch_size // 2):
            for y in range(0, h + pad_size, patch_size // 2):
                # Add weight for left edge
                if x < w and padded_weights[pad_size + y, x] > 0:
                    padded_weights[pad_size + y, 0:pad_size] += 1

                # Add weight for right edge
                if x < w and padded_weights[pad_size + y, w + pad_size - 1 - x] > 0:
                    padded_weights[pad_size + y, w + pad_size : w + 2 * pad_size] += 1

        # Normalize and extract the central region
        padded_weights = padded_weights.unsqueeze(0).repeat(c, 1, 1)
        padded_weights[padded_weights == 0] = 1  # Avoid division by zero
        padded_reconstructed = padded_reconstructed / padded_weights

        # Extract the central region (discard padding)
        reconstructed = padded_reconstructed[
            :, pad_size : pad_size + h, pad_size : pad_size + w
        ]

        return reconstructed

    else:
        raise ValueError(f"Unexpected location format: {sample_location}")


############


def extract_patches(image, patch_size, stride):
    """
    Extract patches from images with consistent sampling and optimal performance.

    Args:
        image: Input tensor of shape (B, C, H, W), (C, H, W), or (H, W)
        patch_size: Size of square patches to extract
        stride: Stride between patch centers

    Returns:
        tuple: (patches tensor of shape [N, C, patch_size, patch_size],
                list of patch locations [(batch_idx, y, x), ...])
    """
    # Standardize input format
    if len(image.shape) == 4:  # (B, C, H, W)
        b, c, h, w = image.shape
    elif len(image.shape) == 3:  # (C, H, W)
        image = image.unsqueeze(0)  # Add batch dimension
        b, c, h, w = image.shape
    elif len(image.shape) == 2:  # (H, W)
        image = image.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        b, c, h, w = image.shape
    else:
        raise ValueError(f"Unexpected image shape: {image.shape}")

    # Calculate valid patch extraction points
    y_positions = list(range(0, h - patch_size + 1, stride))
    x_positions = list(range(0, w - patch_size + 1, stride))

    # Handle edge cases by adding the last valid position if it's not already included
    if h - patch_size > 0 and (h - patch_size) % stride != 0:
        y_positions.append(h - patch_size)
    if w - patch_size > 0 and (w - patch_size) % stride != 0:
        x_positions.append(w - patch_size)

    # Pre-allocate for better memory efficiency
    b * len(y_positions) * len(x_positions)
    all_patches = []
    all_locations = []

    # Extract patches using efficient indexing
    for batch_idx in range(b):
        for y in y_positions:
            for x in x_positions:
                # Extract patch directly with proper indexing
                patch = image[
                    batch_idx : batch_idx + 1, :, y : y + patch_size, x : x + patch_size
                ].squeeze(0)
                all_patches.append(patch)
                all_locations.append((batch_idx, y, x))

    # Stack tensors efficiently
    patches_tensor = torch.stack(all_patches)

    return patches_tensor, all_locations


def reconstruct_from_patches(patches, locations, image_shape, patch_size):
    """Basic reconstruction from patches without special features."""
    # Handle different shape formats
    if len(image_shape) == 4:  # (B, C, H, W)
        b, c, h, w = image_shape
    elif len(image_shape) == 3:  # (C, H, W)
        c, h, w = image_shape
        b = 1
    else:
        raise ValueError(f"Unexpected image shape: {image_shape}")

    # Check location format
    sample_location = locations[0]
    if len(sample_location) == 3:  # (batch_idx, y, x)
        # Need to group by batch
        batch_reconstructed = []

        # Group patches by batch index
        batch_patches = [[] for _ in range(b)]
        batch_locations = [[] for _ in range(b)]

        for i, (patch, location) in enumerate(zip(patches, locations)):
            batch_idx, y, x = location
            batch_patches[batch_idx].append(patch)
            batch_locations[batch_idx].append((y, x))

        # Reconstruct each batch item
        for i in range(b):
            if len(batch_patches[i]) > 0:
                # Call recursively with simpler locations
                reconstructed = reconstruct_from_patches(
                    torch.stack(batch_patches[i]),
                    batch_locations[i],
                    (c, h, w),  # Single image shape
                    patch_size,
                )
                batch_reconstructed.append(reconstructed)
            else:
                # Empty tensor if no patches for this batch
                batch_reconstructed.append(
                    torch.zeros((c, h, w), device=patches[0].device)
                )

        return torch.stack(batch_reconstructed)

    elif len(sample_location) == 2:  # (y, x)
        # Simple reconstruction for a single image
        reconstructed = torch.zeros((c, h, w), device=patches[0].device)
        weights = torch.zeros((h, w), device=patches[0].device)

        # Direct accumulation of patches
        for patch, (y, x) in zip(patches, locations):
            # Handle patches that might go beyond image boundaries
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            patch_h = y_end - y
            patch_w = x_end - x

            # Add patch contribution
            reconstructed[:, y:y_end, x:x_end] += patch[:, :patch_h, :patch_w]
            weights[y:y_end, x:x_end] += 1

        # Normalize by weights
        weights = weights.unsqueeze(0).repeat(c, 1, 1)
        weights[weights == 0] = 1  # Avoid division by zero
        reconstructed = reconstructed / weights

        return reconstructed

    else:
        raise ValueError(f"Unexpected location format: {sample_location}")


def threshold_patches(patches, threshold=0.5):
    return torch.where(patches < threshold, torch.zeros_like(patches), patches)
