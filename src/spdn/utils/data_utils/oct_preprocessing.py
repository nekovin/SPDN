import numpy as np
import cv2


def _compute_decorrelation(oct1, oct2):
    numerator = (oct1 - oct2) ** 2
    denominator = oct1**2 + oct2**2

    epsilon = 1e-6
    decorrelation = numerator / (denominator + epsilon)

    return decorrelation


def compute_decorrelation(oct1, oct2):
    """
    Simple pairwise decorrelation for two frames only.
    Equivalent to SSADA with N=2.

    Args:
        oct1, oct2: 2D numpy arrays (amplitude frames)

    Returns:
        decorrelation: 2D array of decorrelation values
    """
    # Compute correlation coefficient
    numerator = oct1 * oct2
    denominator = 0.5 * (oct1**2 + oct2**2)

    # Avoid division by zero
    epsilon = 1e-8
    correlation = numerator / (denominator + epsilon)

    # Return decorrelation (1 - correlation)
    decorrelation = 1.0 - correlation

    return decorrelation


def compute_decorrelation(oct1, oct2):
    """
    Compute SSADA decorrelation from two OCT amplitude frames.

    Args:
        oct1, oct2: 2D numpy arrays (amplitude frames)

    Returns:
        decorrelation: 2D array of decorrelation values
    """
    # Since we only have 2 frames, we can compute directly
    A_n = oct1
    A_n1 = oct2

    # Compute correlation coefficient
    numerator = A_n * A_n1
    denominator = 0.5 * (A_n**2 + A_n1**2)

    # Avoid division by zero
    epsilon = 1e-8
    correlation = numerator / (denominator + epsilon)

    # Return decorrelation (1 - correlation)
    decorrelation = 1.0 - correlation

    return decorrelation


def remove_speckle_noise(image, min_size=5):
    if len(image.shape) > 2:
        if image.shape[2] == 1:
            image_2d = image[:, :, 0]
        else:
            image_2d = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        image_2d = image

    binary = (image_2d > 0).astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )

    mask = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            mask[labels == i] = 1

    if len(image.shape) > 2:
        for c in range(image.shape[2]):
            image[:, :, c] = image[:, :, c] * mask
    else:
        image = image * mask

    return image


def octa_preprocessing(preprocessed_data, n_neighbours=1, threshold=20):
    n_scans = len(preprocessed_data)
    octa_images = []

    for i in range(n_neighbours, n_scans - n_neighbours):
        decorrelation_values = []
        center_scan = preprocessed_data[i]

        for j in range(i - n_neighbours, i):
            neighbor_scan = preprocessed_data[j]
            decorr = compute_decorrelation(center_scan, neighbor_scan)
            decorrelation_values.append(decorr)

        for j in range(i + 1, i + n_neighbours + 1):
            neighbor_scan = preprocessed_data[j]
            decorr = compute_decorrelation(center_scan, neighbor_scan)
            decorrelation_values.append(decorr)

        if decorrelation_values:
            avg_decorrelation = np.mean(np.stack(decorrelation_values), axis=0)
            thresholded_octa = threshold_octa(avg_decorrelation, center_scan, threshold)
            octa_images.append(thresholded_octa)

    return octa_images


def threshold_octa(octa, oct, threshold):
    background_mask = oct < np.percentile(oct, threshold)

    if np.sum(background_mask) > 0:
        background_mean = np.mean(oct[background_mask])
        background_std = np.std(oct[background_mask])

        intensity_threshold = background_mean + 2 * background_std
    else:
        intensity_threshold = np.percentile(oct, threshold)

    signal_mask = np.clip((oct - intensity_threshold) / (background_std * 2), 0, 1)

    thresholded_octa = octa * signal_mask

    return thresholded_octa
