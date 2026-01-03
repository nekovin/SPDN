import torch
from utils.metrics import display_metrics, display_grouped_metrics

from evaluation.evaluate_pfn import evaluate_progressssive_fusion_unet
from evaluation.evaluate_n2_baselines import evaluate_n2, evaluate_n2_with_ssm
import numpy as np

# import rando
import matplotlib.pyplot as plt


def plot_images(images, metrics_df):
    cols = 4
    rows = len(images) // cols + (len(images) % cols > 0)
    fig, ax = plt.subplots(rows, cols, figsize=(20, 10))
    for i, (key, image) in enumerate(images.items()):
        # ax[i].imshow(image, cmap='gray')
        # ax[i].set_title(key)
        # ax[i].axis('off')
        ax[i // 4, i % 4].imshow(image, cmap="gray")
        ax[i // 4, i % 4].set_title(key)
        ax[i // 4, i % 4].axis("off")

    plt.show()


def get_evaluative_image():
    def resize_image(image, new_size):
        from skimage.transform import resize

        return resize(image, new_size, anti_aliasing=True)

    import os
    from skimage import io

    sdoct1_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
    sdoct1_data = {}
    os.listdir(sdoct1_path)
    for patient in os.listdir(sdoct1_path):
        patient_path = sdoct1_path + "/" + patient
        avg_path = patient_path + f"/{patient}_Averaged Image.tif"
        raw_path = patient_path + f"/{patient}_Raw Image.tif"
        try:
            sdoct1_data[patient] = {}
            sdoct1_data[patient]["avg"] = io.imread(avg_path)
            sdoct1_data[patient]["raw"] = io.imread(raw_path)
        except FileNotFoundError:
            print(f"File not found: {avg_path} or {raw_path}")
            continue

    raw = sdoct1_data["1"]["raw"]
    avg = sdoct1_data["1"]["avg"]

    raw = resize_image(raw, (256, 256))
    avg = resize_image(avg, (256, 256))

    if isinstance(raw, np.ndarray):
        raw = torch.from_numpy(raw).float().unsqueeze(0).unsqueeze(0)

    if isinstance(avg, np.ndarray):
        avg = torch.from_numpy(avg).float().unsqueeze(0).unsqueeze(0)

    return raw, avg


def evaluate_single_image():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    image, reference = get_evaluative_image()

    metrics = {}
    denoised_images = {
        "original": image.cpu().numpy()[0][0],
        "avg": reference.cpu().numpy()[0][0],
    }

    metrics, denoised_images = evaluate_n2(metrics, denoised_images, image, reference)

    prog_metrics, prog_image = evaluate_progressssive_fusion_unet(
        image, reference, device
    )
    metrics["pfn"] = prog_metrics
    denoised_images["pfn"] = prog_image

    metrics, denoised_images = evaluate_n2_with_ssm(
        metrics, denoised_images, image, reference
    )

    metrics_df = display_metrics(metrics)

    display_grouped_metrics(metrics)

    plot_images(denoised_images, metrics_df)
