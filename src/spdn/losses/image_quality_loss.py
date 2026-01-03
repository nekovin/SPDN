from utils import (
    calculate_psnr,
    calculate_ssim,
    calculate_snr,
    calculate_cnr,
    calculate_enl,
    auto_select_roi,
    calculate_cnr_whole,
    calculate_epi,
)
import numpy as np


def evaluate_image_quality(output_image, target_image, original_image=None):
    """
    Evaluate the quality of a denoised image using multiple metrics.

    Args:
        output_image: The denoised image (numpy array)
        target_image: The target/reference image (numpy array)
        original_image: The original noisy image, if available (numpy array)

    Returns:
        Dictionary of image quality metrics
    """
    # Ensure images are in the correct format (remove any extra dimensions)
    output_image = np.squeeze(output_image)
    target_image = np.squeeze(target_image)

    metrics = {}

    # Basic image quality metrics
    metrics["psnr"] = calculate_psnr(output_image, target_image)
    metrics["ssim"] = calculate_ssim(output_image, target_image)

    # If original image is available, calculate improvement metrics
    if original_image is not None:
        original_image = np.squeeze(original_image)

        # Calculate SNR improvement
        snr_original = calculate_snr(original_image)
        snr_output = calculate_snr(output_image)
        metrics["snr"] = snr_output - snr_original
        metrics["snr_value"] = snr_output

        # Try to automatically select ROIs for CNR calculation
        roi_masks = auto_select_roi(output_image)

        if len(roi_masks) >= 2:
            cnr_original = calculate_cnr(original_image, roi_masks[0], roi_masks[1])
            cnr_output = calculate_cnr(output_image, roi_masks[0], roi_masks[1])
            metrics["cnr"] = cnr_output - cnr_original
            metrics["cnr_value"] = cnr_output
        else:
            # Fallback to whole image CNR if ROI selection fails
            cnr_original = calculate_cnr_whole(original_image)
            cnr_output = calculate_cnr_whole(output_image)
            metrics["cnr"] = cnr_output - cnr_original
            metrics["cnr_value"] = cnr_output

        # Calculate ENL if we have at least one ROI
        if len(roi_masks) > 0:
            enl_original = calculate_enl(original_image, roi_masks[0])
            enl_output = calculate_enl(output_image, roi_masks[0])
            metrics["enl"] = enl_output - enl_original
            metrics["enl_value"] = enl_output

        # Edge preservation
        metrics["epi"] = calculate_epi(original_image, output_image)
    else:
        # If no original image, just calculate individual metrics
        metrics["snr_value"] = calculate_snr(output_image)
        metrics["cnr_value"] = calculate_cnr_whole(output_image)
        # Skip EPI which requires the original image

    # Print metrics summary
    print("\n--- Image Quality Metrics ---")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print("----------------------------\n")

    return metrics


def accumulate_metrics(image_metrics, epoch_metrics, metrics_count=0):
    """
    Accumulate image quality metrics for epoch averaging.

    Args:
        image_metrics: Dictionary of metrics for a single image
        epoch_metrics: Dictionary of accumulated metrics for the epoch
        metrics_count: Current count of metrics accumulated

    Returns:
        tuple: (updated_epoch_metrics, updated_metrics_count)
    """
    # Create a copy of epoch_metrics to avoid modifying the original
    updated_metrics = epoch_metrics.copy() if epoch_metrics else {}

    # Add valid metrics to the accumulated totals
    for metric_name, value in image_metrics.items():
        if not np.isnan(value) and not np.isinf(value):
            updated_metrics[metric_name] = updated_metrics.get(metric_name, 0) + value

    # Increment the count of valid metric sets
    updated_count = metrics_count + 1

    return updated_metrics, updated_count


def get_avg_metrics(epoch_metrics, metrics_count, epoch, mode):
    avg_metrics = {}
    if metrics_count > 0:
        for name, total in epoch_metrics.items():
            avg_metrics[name] = total / metrics_count

        print(f"\n=== Average {mode.capitalize()} Metrics for Epoch {epoch + 1} ===")
        for name, value in avg_metrics.items():
            print(f"{name}: {value:.4f}")
        print("=" * 50)
