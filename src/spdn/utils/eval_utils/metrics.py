import numpy as np
from skimage.metrics import structural_similarity
from scipy import ndimage
import cv2
import torch
import os
import time
import matplotlib.pyplot as plt
from spdn.utils.data_utils.paired_preprocessing import paired_preprocessing


def calculate_psnr(img1, img2, max_value=1.0):
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)

    if img1.shape != img2.shape:
        raise ValueError(
            f"Images must have the same dimensions: {img1.shape} vs {img2.shape}"
        )

    mse = np.mean((img1 - img2) ** 2)
    # if mse == 0:
    # return float('inf')

    return 20 * np.log10(max_value / np.sqrt(mse))


def calculate_ssim(img1, img2, max_value=1.0):
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    Value ranges from -1 to 1, higher is better.

    Args:
        img1, img2: Images to compare
        max_value: Maximum possible pixel value (1.0 for normalized images)

    Returns:
        SSIM value
    """
    # Handle multi-channel images
    if img1.ndim == 3 and img1.shape[2] == 1:
        img1 = img1[:, :, 0]
    if img2.ndim == 3 and img2.shape[2] == 1:
        img2 = img2[:, :, 0]

    return structural_similarity(img1, img2, data_range=max_value)


def calculate_snr(img, background_mask=None):
    """
    Calculate Signal-to-Noise Ratio (SNR) for OCT images.
    Higher is better.

    Args:
        img: Input image
        background_mask: Optional mask of background regions. If None, estimates background.

    Returns:
        SNR value in dB
    """
    # Ensure image is in the right format
    img = np.asarray(img, dtype=np.float32)

    if background_mask is None:
        # Assume bottom 10% intensity pixels are background
        threshold = np.percentile(img, 50)
        background_mask = img <= threshold

    # Calculate signal and noise regions
    signal = img[~background_mask]
    noise = img[background_mask]

    # Calculate SNR
    signal_mean = np.mean(signal)
    noise_std = np.std(noise)

    if noise_std == 0:
        print("Noise standard deviation is zero, SNR cannot be calculated.")
        return np.nan

    epsilon = 1e-10  # Small value to avoid log(0)
    # return 20 * np.log10(signal_mean / noise_std + epsilon)
    ratio = abs(signal_mean) / noise_std
    return 20 * np.log10(ratio + epsilon)


def calculate_enl(img, region_mask=None):
    """
    Calculate Equivalent Number of Looks (ENL) for speckle assessment.
    Higher is better, indicates more smoothing of speckle noise.

    Args:
        img: Input image
        region_mask: Optional mask of region to calculate ENL. If None, uses whole image.

    Returns:
        ENL value
    """
    # Ensure image is in the right format
    img = np.asarray(img, dtype=np.float32)

    # If no mask provided, use the whole image
    if region_mask is None:
        region = img
    else:
        region = img[region_mask]

    if len(region) == 0:
        return 0.0

    # Calculate ENL
    mean_val = np.mean(region)
    var_val = np.var(region)

    if var_val == 0:
        return float("inf")

    return (mean_val**2) / var_val


def calculate_enl(img, roi_masks=None):
    """
    Calculate ENL (Equivalent Number of Looks) for one or more ROIs.

    Args:
        img (np.ndarray): Input image.
        roi_masks (list of np.ndarray): List of binary masks for ENL regions.

    Returns:
        float: Average ENL across all ROIs.
    """
    img = np.asarray(img, dtype=np.float32)

    if roi_masks is None or len(roi_masks) == 0:
        region = img
        mean_val = np.mean(region)
        var_val = np.var(region)
        return float("inf") if var_val == 0 else (mean_val**2) / var_val

    enls = []
    for mask in roi_masks:
        region = img[mask > 0]
        if len(region) == 0:
            continue
        mean_val = np.mean(region)
        var_val = np.var(region)
        enls.append(float("inf") if var_val == 0 else (mean_val**2) / var_val)

    return np.mean(enls) if enls else 0.0


def calculate_epi(img1, img2):
    """
    Calculate Edge Preservation Index (EPI).
    Value ranges from 0 to 1, higher is better.

    Args:
        img1: Original image
        img2: Denoised image

    Returns:
        EPI value
    """
    # Ensure images are in the same shape and type
    img1 = np.asarray(img1, dtype=np.float32)
    img2 = np.asarray(img2, dtype=np.float32)

    # Apply Sobel operator to detect edges
    edges1_h = ndimage.sobel(img1, axis=0)
    edges1_v = ndimage.sobel(img1, axis=1)
    edges1 = np.sqrt(edges1_h**2 + edges1_v**2)

    edges2_h = ndimage.sobel(img2, axis=0)
    edges2_v = ndimage.sobel(img2, axis=1)
    edges2 = np.sqrt(edges2_h**2 + edges2_v**2)

    # Calculate correlation between edge maps
    edges1_mean = np.mean(edges1)
    edges2_mean = np.mean(edges2)

    numerator = np.sum((edges1 - edges1_mean) * (edges2 - edges2_mean))
    denominator = np.sqrt(
        np.sum((edges1 - edges1_mean) ** 2) * np.sum((edges2 - edges2_mean) ** 2)
    )

    if denominator == 0:
        return 0.0

    return numerator / denominator


def auto_select_roi(img, n_regions=3, min_size=100):
    """
    Automatically select regions of interest for CNR calculation

    Args:
        img: Input image
        n_regions: Number of regions to select
        min_size: Minimum region size

    Returns:
        List of masks for ROIs
    """
    if img.max() <= 1.0:
        img_8bit = (img * 255).astype(np.uint8)
    else:
        img_8bit = img.astype(np.uint8)

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        img_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # Find connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    # Select regions based on size
    region_sizes = stats[1:, cv2.CC_STAT_AREA]  # Skip background (label 0)
    valid_regions = [i + 1 for i, size in enumerate(region_sizes) if size >= min_size]

    # Sort regions by size and select the n_regions largest (excluding background)
    valid_regions = sorted(
        valid_regions, key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True
    )[:n_regions]

    # Create masks for selected regions
    masks = []
    for region_idx in valid_regions:
        mask = labels == region_idx
        masks.append(mask)

    return masks


def calculate_cnr_whole(image):
    """
    Calculate CNR using a percentile-based approach that doesn't require manual ROI selection.
    Works well for evaluating the whole image quality.

    Args:
        image: Input image (2D numpy array)

    Returns:
        cnr: Contrast-to-noise ratio
    """
    # Ensure image is normalized
    if image.max() > 1.0:
        image = image / image.max()

    # Use percentiles to identify signal and background regions
    # Signal: top 10% brightest pixels
    # Background: bottom 10% darkest pixels
    signal_threshold = np.percentile(image, 90)
    background_threshold = np.percentile(image, 10)

    signal_mask = image >= signal_threshold
    background_mask = image <= background_threshold

    # Extract regions
    signal_region = image[signal_mask]
    background_region = image[background_mask]

    # Calculate statistics
    signal_mean = np.mean(signal_region)
    background_mean = np.mean(background_region)
    signal_std = np.std(signal_region)
    background_std = np.std(background_region)

    # Calculate CNR
    # Avoid division by zero
    denominator = np.sqrt(signal_std**2 + background_std**2)
    if denominator == 0:
        return 0

    cnr = abs(signal_mean - background_mean) / denominator

    # Convert to dB for consistency with other metrics
    cnr_db = 20 * np.log10(cnr) if cnr > 0 else -np.inf

    return cnr_db


def auto_select_roi_using_layers(img, layer_boundaries):
    """
    Create ROIs based on anatomical layer segmentation
    layer_boundaries: shape (3, width) - the automatic layers
    """
    height, width = img.shape

    # Create masks between layer boundaries
    foreground_mask = np.zeros_like(img, dtype=bool)
    background_mask = np.zeros_like(img, dtype=bool)

    # Example: Use region between layer 1 and 2 as foreground (high signal retinal tissue)
    for x in range(width):
        if layer_boundaries[0, x] < layer_boundaries[1, x]:  # Ensure valid boundaries
            y_start = int(layer_boundaries[0, x])
            y_end = int(layer_boundaries[1, x])
            foreground_mask[y_start:y_end, x] = True

    # Use deeper region as background (lower signal)
    for x in range(width):
        if layer_boundaries[2, x] + 20 < height:  # 20 pixels below layer 3
            y_start = int(layer_boundaries[2, x]) + 10
            y_end = min(int(layer_boundaries[2, x]) + 30, height)
            background_mask[y_start:y_end, x] = True

    return [foreground_mask, background_mask]


def calculate_cnr(img, roi_masks=None):
    """
    Calculate average pairwise CNR for an image using multiple ROI masks.

    Args:
        img (np.ndarray): Input image.
        roi_masks (list of np.ndarray): List of binary ROI masks.

    Returns:
        float: Mean pairwise CNR across all ROI combinations.
    """
    img = np.asarray(img, dtype=np.float32)

    if roi_masks is None or len(roi_masks) < 2:
        raise ValueError("Need at least two ROI masks for CNR calculation.")

    # Convert roi_masks to numpy if they're torch tensors
    numpy_masks = []
    for mask in roi_masks:
        if hasattr(mask, "cpu"):
            numpy_masks.append(mask.cpu().numpy())
        elif isinstance(mask, torch.Tensor):
            numpy_masks.append(mask.numpy())
        else:
            numpy_masks.append(np.asarray(mask))

    print(f"Calculating CNR using {len(numpy_masks)} ROI masks")

    cnr_values = []
    for i in range(len(numpy_masks)):
        for j in range(i + 1, len(numpy_masks)):
            fg = img[numpy_masks[i] > 0]
            bg = img[numpy_masks[j] > 0]

            if len(fg) == 0 or len(bg) == 0:
                print(
                    f"Warning: Empty ROI found (ROI {i} has {len(fg)} pixels, ROI {j} has {len(bg)} pixels)"
                )
                continue

            fg_mean, fg_std = np.mean(fg), np.std(fg)
            bg_mean, bg_std = np.mean(bg), np.std(bg)

            print(f"ROI {i}: mean={fg_mean:.4f}, std={fg_std:.4f}, pixels={len(fg)}")
            print(f"ROI {j}: mean={bg_mean:.4f}, std={bg_std:.4f}, pixels={len(bg)}")

            denom = np.sqrt(fg_std**2 + bg_std**2)
            if denom == 0:
                print(f"Warning: Zero denominator for ROI pair {i},{j}")
                continue

            cnr = 10 * np.log10(np.abs(fg_mean - bg_mean) / denom)
            print(f"CNR between ROI {i} and {j}: {cnr:.4f} dB")
            cnr_values.append(cnr)

    final_cnr = float(np.mean(cnr_values)) if cnr_values else 0.0
    print(f"Average CNR: {final_cnr:.4f} dB")
    return final_cnr


def evaluate_oct_denoising(original, denoised, reference=None, layers=None):
    metrics = {}

    print("Original image intensity distribution:")
    print(f"  0-0.1 range: {np.sum((original >= 0) & (original <= 0.1))} pixels")
    print(f"  0.1-0.3 range: {np.sum((original > 0.1) & (original <= 0.3))} pixels")
    print(f"  0.3-0.7 range: {np.sum((original > 0.3) & (original <= 0.7))} pixels")

    print("Denoised image intensity distribution:")
    print(f"  0-0.1 range: {np.sum((denoised >= 0) & (denoised <= 0.1))} pixels")
    print(f"  0.1-0.3 range: {np.sum((denoised > 0.1) & (denoised <= 0.3))} pixels")
    print(f"  0.3-0.7 range: {np.sum((denoised > 0.3) & (denoised <= 0.7))} pixels")

    if reference is None:
        metrics["psnr"] = np.nan
        metrics["ssim"] = np.nan
    else:
        metrics["psnr"] = calculate_psnr(denoised, reference)
        metrics["ssim"] = calculate_ssim(denoised, reference)

    metrics["snr"] = calculate_snr(denoised) - calculate_snr(original)

    if layers is None:
        try:
            roi_masks = get_roi(denoised)
            # roi_masks = auto_select_roi(denoised)
            print("Using static ROIs for CNR calculation.")
        except Exception as e:
            raise e
    else:
        raise ValueError(
            "Layer-based ROI selection is not implemented yet. Please provide ROI masks manually."
        )

    # if len(roi_masks) >= 2:
    # print(f"Using {len(roi_masks)} ROIs for CNR calculation.")
    # metrics['cnr'] = calculate_cnr(denoised, roi_masks) - calculate_cnr(original, roi_masks)
    if len(roi_masks) >= 2:
        print(f"Using {len(roi_masks)} ROIs for CNR calculation.")

        print("\n--- ORIGINAL IMAGE CNR ---")
        cnr_original = calculate_cnr(original, roi_masks)

        print("\n--- DENOISED IMAGE CNR ---")
        cnr_denoised = calculate_cnr(denoised, roi_masks)

        print("\n--- CNR SUMMARY ---")
        print(f"Original CNR: {cnr_original:.4f} dB")
        print(f"Denoised CNR: {cnr_denoised:.4f} dB")
        print(f"CNR Change: {cnr_denoised - cnr_original:.4f} dB")

        if cnr_denoised > cnr_original:
            print("✓ GOOD: Denoising improved contrast")
        else:
            print("✗ WARNING: Denoising reduced contrast - may be over-smoothing")

        metrics["cnr"] = cnr_denoised - cnr_original
    else:
        raise ValueError(
            "Not enough ROIs selected for CNR calculation. At least 2 are required."
        )

    if len(roi_masks) > 0:
        metrics["enl"] = calculate_enl(denoised, roi_masks) - calculate_enl(
            original, roi_masks
        )

    metrics["epi"] = calculate_epi(original, denoised)

    return metrics


def evaluate_oct_denoising(original, denoised, reference=None, layers=None):
    metrics = {}

    if reference is None:
        metrics["psnr"] = np.nan
        metrics["ssim"] = np.nan
    else:
        # Apply same correction to reference if provided
        # reference_norm = (reference - global_min) / (global_max - global_min)
        metrics["psnr"] = calculate_psnr(denoised, reference)
        metrics["ssim"] = calculate_ssim(denoised, reference)

    metrics["snr"] = calculate_snr(denoised)  # - calculate_snr(original)
    metrics["snr_change"] = calculate_snr(denoised) - calculate_snr(original)

    if layers is None:
        try:
            roi_masks = get_roi(denoised)
            print("Using static ROIs for CNR calculation.")
        except Exception as e:
            raise e
    else:
        # raise ValueError("Layer-based ROI selection is not implemented yet. Please provide ROI masks manually.")
        roi_masks = auto_select_roi_using_layers(denoised, layers)

    if len(roi_masks) >= 2:
        print(f"Using {len(roi_masks)} ROIs for CNR calculation.")

        print("\n--- ORIGINAL IMAGE CNR ---")
        cnr_original = calculate_cnr(original, roi_masks)

        print("\n--- DENOISED IMAGE CNR (with black correction) ---")
        cnr_denoised = calculate_cnr(denoised, roi_masks)

        print("\n--- CNR SUMMARY ---")
        print(f"Original CNR: {cnr_original:.4f} dB")
        print(f"Denoised CNR: {cnr_denoised:.4f} dB")
        print(f"CNR Change: {cnr_denoised - cnr_original:.4f} dB")

        if cnr_denoised > cnr_original:
            print("✓ GOOD: Denoising improved contrast")
        else:
            print("✗ WARNING: Denoising reduced contrast - may be over-smoothing")

        metrics["cnr"] = cnr_denoised
        metrics["cnr_change"] = cnr_denoised - cnr_original
    else:
        raise ValueError(
            "Not enough ROIs selected for CNR calculation. At least 2 are required."
        )

    if len(roi_masks) > 0:
        metrics["enl"] = calculate_enl(
            denoised, roi_masks
        )  # - calculate_enl(original, roi_masks)
        metrics["enl_change"] = calculate_enl(denoised, roi_masks) - calculate_enl(
            original, roi_masks
        )

    metrics["epi"] = calculate_epi(original, denoised)

    return metrics


def denoise_image(model, image, device=None):
    """Apply model to denoise a single image"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    if len(image.shape) == 2:
        image = image[:, :, np.newaxis]

    input_tensor = (
        torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    )

    with torch.no_grad():
        output = model(input_tensor)

    output_image = output.squeeze().cpu().numpy()

    if len(output_image.shape) == 2:
        return output_image
    else:
        return output_image.transpose(1, 2, 0)


def validate_model(
    model, n_patients=5, n_images_per_patient=10, device=None, save_results=True
):
    """Validate model on a separate set of images"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load validation data
    val_dataset = paired_preprocessing(n_patients, n_images_per_patient, sample=True)

    # Track metrics
    metrics_summary = {
        "snr_original": [],
        "snr_denoised": [],
        "cnr_original": [],
        "cnr_denoised": [],
        "enl_original": [],
        "enl_denoised": [],
        "epi": [],
        "ssim_self": [],
    }

    # Create save directory if needed
    save_path = os.path.join(os.getcwd(), "validation_results")
    if save_results and not os.path.exists(save_path):
        os.makedirs(save_path)

    # Keep track of one example for visualization
    example_result = None

    for patient_id, data in val_dataset.items():
        for i in range(len(data) - 1):
            input_img = data[i][0]
            target_img = data[i + 1][0]  # Next scan as reference

            # Denoise the input image
            denoised_img = denoise_image(model, input_img, device)

            # Calculate metrics
            metrics = evaluate_oct_denoising(input_img, denoised_img, target_img)

            # Add to summary
            for key in metrics_summary:
                if key in metrics:
                    metrics_summary[key].append(metrics[key])

            # Store the first result as our example
            if example_result is None:
                example_result = {
                    "patient_id": patient_id,
                    "image_id": i,
                    "input": input_img,
                    "target": target_img,
                    "denoised": denoised_img,
                    "metrics": {
                        k: metrics[k]
                        for k in metrics
                        if not np.isnan(metrics[k]) and not np.isinf(metrics[k])
                    },
                }

    # Calculate averages only for metrics that have values
    avg_metrics = {}
    for k, v in metrics_summary.items():
        if len(v) > 0:
            avg_metrics[k] = np.mean(v)

    # Print results
    print("\n===== Validation Results =====")

    # Print metrics if available
    if "snr_original" in avg_metrics and "snr_denoised" in avg_metrics:
        print(
            f"SNR: {avg_metrics['snr_original']:.2f} dB → {avg_metrics['snr_denoised']:.2f} dB"
        )

    if "cnr_original" in avg_metrics and "cnr_denoised" in avg_metrics:
        print(
            f"CNR: {avg_metrics['cnr_original']:.2f} dB → {avg_metrics['cnr_denoised']:.2f} dB"
        )

    if "enl_original" in avg_metrics and "enl_denoised" in avg_metrics:
        print(
            f"ENL: {avg_metrics['enl_original']:.2f} → {avg_metrics['enl_denoised']:.2f}"
        )

    if "epi" in avg_metrics:
        print(f"Edge Preservation Index: {avg_metrics['epi']:.4f}")

    if "ssim_self" in avg_metrics:
        print(f"SSIM (self-reference): {avg_metrics['ssim_self']:.4f}")

    print("=============================\n")

    # Save results
    if save_results and example_result is not None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        results_dir = os.path.join(save_path, f"results_{timestamp}")
        os.makedirs(results_dir, exist_ok=True)

        # Save summary metrics
        with open(os.path.join(results_dir, "metrics_summary.txt"), "w") as f:
            for k, v in avg_metrics.items():
                f.write(f"{k}: {v}\n")

        # Save the single example image
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(example_result["input"], cmap="gray")
        plt.title("Input (Noisy)")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(example_result["denoised"], cmap="gray")
        plt.title("Denoised")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(example_result["target"], cmap="gray")
        plt.title("Target")
        plt.axis("off")

        plt.savefig(os.path.join(results_dir, "example_comparison.png"))
        plt.close()

        print(f"Validation results saved to {results_dir}")

    return avg_metrics


from IPython.display import display


def display_metrics(metrics):
    import pandas as pd

    df = pd.DataFrame(metrics)

    def highlight_max(s):
        is_max = s == s.max()
        return ["font-weight: bold" if v else "" for v in is_max]

    styled_df = df.style.apply(highlight_max, axis=1)
    display(styled_df)
    return df


def display_grouped_metrics(metrics):
    import pandas as pd
    from IPython.display import display

    # Define base schemas to group by
    base_schemas = ["n2n", "n2v", "n2s", "pfn"]

    schema_tables = {}

    # Process each metric key
    for metric_key in metrics.keys():
        # Find which base schema this metric belongs to
        matched_schema = None
        for schema in base_schemas:
            if schema in metric_key:
                matched_schema = schema
                break

        if matched_schema:
            if matched_schema not in schema_tables:
                schema_tables[matched_schema] = {}

            schema_tables[matched_schema][metric_key] = metrics[metric_key]

    # Display tables for each schema group
    all_dfs = {}
    for schema, data in schema_tables.items():
        if data:  # Only process if we have data
            df = pd.DataFrame(data)

            # Apply styling to highlight maximum values in each row
            def highlight_max(s):
                is_max = s == s.max()
                return ["font-weight: bold" if v else "" for v in is_max]

            styled_df = df.style.apply(highlight_max, axis=1)

            print(f"\n--- {schema.upper()} Models ---")
            display(styled_df)
            all_dfs[schema] = df

    return all_dfs


###


def get_roi(denoised):
    roi_list = []
    try:
        # Convert to numpy if it's a torch tensor
        if hasattr(denoised, "cpu"):
            denoised_np = denoised.cpu().numpy()
        elif isinstance(denoised, torch.Tensor):
            denoised_np = denoised.numpy()
        else:
            denoised_np = np.asarray(denoised)

        h, w = denoised_np.shape[-2:]
        roi_size = h // 15
        center_y = h // 2
        w // 4

        coords = [
            (center_y - roi_size, w // 2 - roi_size),  # Center tissue
            # (center_y - roi_size, 3 * quarter_w - roi_size),  # Right tissue
            (10, w // 2 - roi_size),  # Top background (dark region)
            # (h - roi_size - 10, w // 2 - roi_size)           # Bottom background
        ]

        for idx, (y, x) in enumerate(coords):
            # Create numpy array ROI mask instead of torch tensor
            roi = np.zeros_like(denoised_np, dtype=np.float32)

            # Ensure coordinates are within bounds
            y_start = max(0, y)

            # y_end = min(h, y + roi_size)
            y_end = min(h, y + 3 * roi_size)
            x_start = max(0, x)
            x_end = min(w, x + 2 * roi_size)  # Keep your original 2*roi_size width

            roi[y_start:y_end, x_start:x_end] = 1

            # Count pixels in this ROI
            roi_pixels = np.sum(roi > 0)
            print(
                f"ROI {idx}: shape={roi.shape}, coordinates=({y_start}:{y_end}, {x_start}:{x_end}), pixels={roi_pixels}"
            )

            roi_list.append(roi)

        # Visualization
        overlay = denoised_np.copy()
        mask = np.sum(roi_list, axis=0)

        plt.figure(figsize=(6, 6))
        plt.imshow(overlay, cmap="gray")
        plt.imshow(mask, cmap="Reds", alpha=0.3)  # Increased alpha to see ROIs better
        plt.title("Static ROI Visualisation")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error creating static ROIs: {e}")
        raise e

    return roi_list


def get_roi(denoised):
    roi_list = []
    try:
        # Convert to numpy if it's a torch tensor
        if hasattr(denoised, "cpu"):
            denoised_np = denoised.cpu().numpy()
        elif isinstance(denoised, torch.Tensor):
            denoised_np = denoised.numpy()
        else:
            denoised_np = np.asarray(denoised)

        h, w = denoised_np.shape[-2:]
        roi_size = h // 15
        center_y = h // 2

        coords = [
            (center_y - roi_size, w // 2 - roi_size),  # Center tissue
            (10, w // 2 - roi_size),  # Top background (dark region)
        ]

        # For visualization, create boundary boxes instead
        from matplotlib.patches import Rectangle

        plt.figure(figsize=(6, 6))
        plt.imshow(denoised_np, cmap="gray")

        for idx, (y, x) in enumerate(coords):
            roi = np.zeros_like(denoised_np, dtype=np.float32)

            y_start = max(0, y)
            y_end = min(h, y + 2 * roi_size)
            x_start = max(0, x)
            x_end = min(w, x + 2 * roi_size)

            roi[y_start:y_end, x_start:x_end] = 1

            roi_pixels = np.sum(roi > 0)
            print(
                f"ROI {idx}: coordinates=({y_start}:{y_end}, {x_start}:{x_end}), pixels={roi_pixels}"
            )

            roi_list.append(roi)

            # Add rectangle boundary to plot
            rect = Rectangle(
                (x_start, y_start),
                x_end - x_start,
                y_end - y_start,
                linewidth=2,
                edgecolor="red",
                facecolor="none",
                alpha=0.8,
            )
            plt.gca().add_patch(rect)

            # Add ROI label
            plt.text(
                x_start + 5,
                y_start + 15,
                f"ROI {idx}",
                color="red",
                fontsize=10,
                fontweight="bold",
            )

        plt.title("Static ROI Visualisation")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error creating static ROIs: {e}")
        raise e

    return roi_list


def calculate_cnr(img, roi_masks=None):
    """
    Calculate average pairwise CNR for an image using multiple ROI masks.

    Args:
        img (np.ndarray): Input image.
        roi_masks (list of np.ndarray): List of binary ROI masks.

    Returns:
        float: Mean pairwise CNR across all ROI combinations.
    """
    img = np.asarray(img, dtype=np.float32)

    if roi_masks is None or len(roi_masks) < 2:
        raise ValueError("Need at least two ROI masks for CNR calculation.")

    # Convert roi_masks to numpy if they're torch tensors
    numpy_masks = []
    for mask in roi_masks:
        if hasattr(mask, "cpu"):
            numpy_masks.append(mask.cpu().numpy())
        elif isinstance(mask, torch.Tensor):
            numpy_masks.append(mask.numpy())
        else:
            numpy_masks.append(np.asarray(mask))

    print(f"Calculating CNR using {len(numpy_masks)} ROI masks")

    # ADD ROI ANALYSIS FIRST
    print("=== ROI VALUE ANALYSIS ===")
    for k, mask in enumerate(numpy_masks):
        region_pixels = img[mask > 0]
        region_mean = np.mean(region_pixels)
        region_std = np.std(region_pixels)
        # region_type = "TISSUE" if k < 2 else "BACKGROUND"
        if len(numpy_masks) == 2:
            region_type = (
                "TISSUE" if k == 0 else "BACKGROUND"
            )  # First is tissue, second is background
        else:
            region_type = "TISSUE" if k < 2 else "BACKGROUND"
        print(
            f"ROI {k} ({region_type}): mean = {region_mean:.4f}, std = {region_std:.4f}"
        )

        if k >= 2:  # Background ROIs
            very_dark_pixels = np.sum(region_pixels < 0.1)
            dark_pixels = np.sum(region_pixels < 0.2)
            print(
                f"  Very dark pixels (< 0.1): {very_dark_pixels}/{len(region_pixels)} ({very_dark_pixels / len(region_pixels) * 100:.1f}%)"
            )
            print(
                f"  Dark pixels (< 0.2): {dark_pixels}/{len(region_pixels)} ({dark_pixels / len(region_pixels) * 100:.1f}%)"
            )
    print("=== END ANALYSIS ===")

    cnr_values = []
    for i in range(len(numpy_masks)):
        for j in range(i + 1, len(numpy_masks)):
            fg = img[numpy_masks[i] > 0]
            bg = img[numpy_masks[j] > 0]

            if len(fg) == 0 or len(bg) == 0:
                print(
                    f"Warning: Empty ROI found (ROI {i} has {len(fg)} pixels, ROI {j} has {len(bg)} pixels)"
                )
                continue

            fg_mean, fg_std = np.mean(fg), np.std(fg)
            bg_mean, bg_std = np.mean(bg), np.std(bg)

            print(f"ROI {i}: mean={fg_mean:.4f}, std={fg_std:.4f}, pixels={len(fg)}")
            print(f"ROI {j}: mean={bg_mean:.4f}, std={bg_std:.4f}, pixels={len(bg)}")

            denom = np.sqrt(fg_std**2 + bg_std**2)
            if denom == 0:
                print(f"Warning: Zero denominator for ROI pair {i},{j}")
                continue

            cnr = 10 * np.log10(np.abs(fg_mean - bg_mean) / denom)
            print(f"CNR between ROI {i} and {j}: {cnr:.4f} dB")
            cnr_values.append(cnr)

    final_cnr = float(np.mean(cnr_values)) if cnr_values else 0.0
    print(f"Average CNR: {final_cnr:.4f} dB")
    return final_cnr


#


def calculate_cnr_with_full_debug(img, roi_masks=None, image_name="Unknown"):
    """
    Calculate CNR with comprehensive debugging to understand the contradiction
    """
    img = np.asarray(img, dtype=np.float32)

    if roi_masks is None or len(roi_masks) < 2:
        raise ValueError("Need at least two ROI masks for CNR calculation.")

    print(f"\n{'=' * 60}")
    print(f"COMPREHENSIVE CNR DEBUG FOR {image_name}")
    print(f"{'=' * 60}")

    # Overall image statistics
    print("WHOLE IMAGE STATS:")
    print(f"  Min: {img.min():.4f}")
    print(f"  Max: {img.max():.4f}")
    print(f"  Mean: {img.mean():.4f}")
    print(f"  Std: {img.std():.4f}")

    # Intensity distribution
    print("\nINTENSITY DISTRIBUTION:")
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        count = np.sum(img <= threshold)
        percentage = count / img.size * 100
        print(f"  <= {threshold}: {count} pixels ({percentage:.1f}%)")

    # Convert roi_masks to numpy
    numpy_masks = []
    for mask in roi_masks:
        if hasattr(mask, "cpu"):
            numpy_masks.append(mask.cpu().numpy())
        elif isinstance(mask, torch.Tensor):
            numpy_masks.append(mask.numpy())
        else:
            numpy_masks.append(np.asarray(mask))

    print("\nROI DETAILED ANALYSIS:")
    roi_stats = []

    for k, mask in enumerate(numpy_masks):
        region_pixels = img[mask > 0]
        region_mean = np.mean(region_pixels)
        region_std = np.std(region_pixels)
        region_min = np.min(region_pixels)
        region_max = np.max(region_pixels)
        region_type = "TISSUE" if k < 2 else "BACKGROUND"

        roi_stats.append(
            {
                "mean": region_mean,
                "std": region_std,
                "min": region_min,
                "max": region_max,
                "pixels": len(region_pixels),
            }
        )

        print(f"  ROI {k} ({region_type}):")
        print(f"    Mean: {region_mean:.4f}")
        print(f"    Std:  {region_std:.4f}")
        print(f"    Min:  {region_min:.4f}")
        print(f"    Max:  {region_max:.4f}")
        print(f"    Pixels: {len(region_pixels)}")

        # Show distribution within this ROI
        if k >= 2:  # Background ROIs
            very_dark = np.sum(region_pixels < 0.1)
            dark = np.sum(region_pixels < 0.2)
            medium_dark = np.sum(region_pixels < 0.3)
            print(
                f"    Very dark (< 0.1): {very_dark}/{len(region_pixels)} ({very_dark / len(region_pixels) * 100:.1f}%)"
            )
            print(
                f"    Dark (< 0.2): {dark}/{len(region_pixels)} ({dark / len(region_pixels) * 100:.1f}%)"
            )
            print(
                f"    Medium dark (< 0.3): {medium_dark}/{len(region_pixels)} ({medium_dark / len(region_pixels) * 100:.1f}%)"
            )

    print("\nCNR CALCULATIONS:")
    cnr_values = []
    tissue_to_background_cnrs = []

    for i in range(len(numpy_masks)):
        for j in range(i + 1, len(numpy_masks)):
            fg = img[numpy_masks[i] > 0]
            bg = img[numpy_masks[j] > 0]

            if len(fg) == 0 or len(bg) == 0:
                continue

            fg_mean, fg_std = np.mean(fg), np.std(fg)
            bg_mean, bg_std = np.mean(bg), np.std(bg)

            # Calculate CNR components
            mean_diff = abs(fg_mean - bg_mean)
            combined_std = np.sqrt(fg_std**2 + bg_std**2)
            cnr_linear = mean_diff / combined_std if combined_std > 0 else 0
            cnr_db = 10 * np.log10(cnr_linear) if cnr_linear > 0 else -np.inf

            # Identify what type of comparison this is
            roi_i_type = "TISSUE" if i < 2 else "BACKGROUND"
            roi_j_type = "TISSUE" if j < 2 else "BACKGROUND"
            comparison_type = f"{roi_i_type} vs {roi_j_type}"

            print(f"  ROI {i} vs ROI {j} ({comparison_type}):")
            print(f"    Mean difference: {mean_diff:.4f}")
            print(f"    Combined noise: {combined_std:.4f}")
            print(f"    CNR linear: {cnr_linear:.4f}")
            print(f"    CNR dB: {cnr_db:.4f}")

            # Track tissue-to-background CNRs separately
            # if (i < 2 and j >= 2) or (i >= 2 and j < 2):
            # tissue_to_background_cnrs.append(cnr_db)
            # print(f"    *** This is a TISSUE-BACKGROUND comparison ***")

            if len(numpy_masks) == 2:
                tissue_to_background_cnrs.append(
                    cnr_db
                )  # The only comparison is tissue vs background
            elif (i < 2 and j >= 2) or (i >= 2 and j < 2):
                tissue_to_background_cnrs.append(cnr_db)
            cnr_values.append(cnr_db)

    # Summary analysis
    print("\nSUMMARY ANALYSIS:")
    final_cnr = float(np.mean(cnr_values)) if cnr_values else 0.0
    tissue_bg_cnr = (
        float(np.mean(tissue_to_background_cnrs)) if tissue_to_background_cnrs else 0.0
    )

    print(f"  Overall average CNR: {final_cnr:.4f} dB")
    print(f"  Tissue-to-Background CNR: {tissue_bg_cnr:.4f} dB")
    print(
        f"  Number of tissue-background comparisons: {len(tissue_to_background_cnrs)}"
    )

    # Diagnose potential issues
    print("\nDIAGNOSIS:")

    # Check if background ROIs are actually dark
    bg_roi_means = [roi_stats[i]["mean"] for i in range(2, len(roi_stats))]
    avg_bg_mean = np.mean(bg_roi_means)

    if avg_bg_mean > 0.4:
        print(
            f"  ⚠️  ISSUE: Background ROIs have high intensity (avg={avg_bg_mean:.3f})"
        )
        print("     Background ROIs might be hitting tissue instead of dark regions!")
    elif avg_bg_mean < 0.2:
        print(
            f"  ✓ Background ROIs properly measuring dark regions (avg={avg_bg_mean:.3f})"
        )
    else:
        print(f"  ? Background ROIs measuring medium intensity (avg={avg_bg_mean:.3f})")

    # Check tissue vs background separation
    tissue_roi_means = [roi_stats[i]["mean"] for i in range(0, 2)]
    avg_tissue_mean = np.mean(tissue_roi_means)
    contrast_ratio = avg_tissue_mean / avg_bg_mean if avg_bg_mean > 0 else float("inf")

    print(f"  Tissue mean: {avg_tissue_mean:.3f}")
    print(f"  Background mean: {avg_bg_mean:.3f}")
    print(f"  Contrast ratio: {contrast_ratio:.2f}")

    if contrast_ratio < 1.5:
        print("  ⚠️  ISSUE: Low contrast ratio! Tissue and background too similar.")
    elif contrast_ratio > 3.0:
        print("  ✓ Good contrast ratio between tissue and background")

    print(f"{'=' * 60}\n")

    return final_cnr


def calculate_percentile_cnr(img):
    """Use percentiles instead of fixed ROIs"""
    # Get brightest 20% as tissue
    tissue_threshold = np.percentile(img, 90)
    tissue_pixels = img[img >= tissue_threshold]

    # Get darkest 20% as background
    bg_threshold = np.percentile(img, 30)
    bg_pixels = img[img <= bg_threshold]

    # Calculate CNR
    tissue_mean, tissue_std = np.mean(tissue_pixels), np.std(tissue_pixels)
    bg_mean, bg_std = np.mean(bg_pixels), np.std(bg_pixels)

    mean_diff = abs(tissue_mean - bg_mean)
    combined_std = np.sqrt(tissue_std**2 + bg_std**2)
    cnr = 10 * np.log10(mean_diff / combined_std) if combined_std > 0 else 0

    print(
        f"Percentile CNR: tissue={tissue_mean:.3f}, bg={bg_mean:.3f}, CNR={cnr:.2f} dB"
    )
    return cnr


# Modified calculate_cnr function to use the debug version
def calculate_cnr(img, roi_masks=None, debug_name="Image"):
    """Use the comprehensive debug version"""
    return calculate_cnr_with_full_debug(img, roi_masks, debug_name)
    # return calculate_percentile_cnr(img)
