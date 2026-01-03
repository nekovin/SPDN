import cv2
import torch
import numpy as np


def normalize_image(np_img):
    if np_img.max() > 0:
        foreground_mask = np_img > 0.01
        if foreground_mask.any():
            fg_min = np_img[foreground_mask].min()
            fg_max = np_img[foreground_mask].max()

            if fg_max > fg_min:
                np_img[foreground_mask] = (np_img[foreground_mask] - fg_min) / (
                    fg_max - fg_min
                )

    np_img[np_img < 0.01] = 0
    return np_img


def normalize_image_np(img):
    min_val = np.min(img)
    max_val = np.max(img)

    if max_val > min_val:
        normalized = (img - min_val) / (max_val - min_val)
    else:
        normalized = np.zeros_like(img)

    return normalized


def normalize_image_torch(img):
    # Get min and max values
    min_val = torch.min(img)
    max_val = torch.max(img)

    # Normalize to [0,1] range
    if max_val > min_val:
        normalized = (img - min_val) / (max_val - min_val)
    else:
        normalized = torch.zeros_like(img)

    return normalized


def standard_preprocessing(oct_volume):
    preprocessed = []

    for i, img in enumerate(oct_volume):
        resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

        resized = normalize_image_np(resized)

        resized = resized[:, :, np.newaxis]

        preprocessed.append(resized)

    return np.array(preprocessed)
