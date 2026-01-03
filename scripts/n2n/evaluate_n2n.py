import matplotlib.pyplot as plt
from ssm.evaluation import evaluate_baseline, evaluate_ssm_constraint
from spdn.utils import load_sdoct_dataset, display_metrics, normalize_image_torch
from spdn.utils.paths import get_sdoct_path
import os
import random
from spdn.utils.config import get_config
import numpy as np


def standard_preprocessing_single_img(img):
    # img = normalize_image_np(img)
    img = normalize_image_torch(img)

    return np.array(img)


def main():
    config_path = os.getenv("N2_CONFIG_PATH")

    config = get_config(config_path)

    n_patients = config["training"]["n_patients"]
    n_images_per_patient = config["training"]["n_images_per_patient"]

    override_config = {
        "eval": {
            "ablation": f"patient_count/{n_patients}_patients/{n_images_per_patient}_images",
            "n_patients": n_patients,
        }
    }

    all_metrics = {}

    sdoct_path = get_sdoct_path()
    dataset = load_sdoct_dataset(sdoct_path)

    sample = random.choice(list(dataset.keys()))
    raw_image = dataset[sample]["raw"]
    reference = dataset[sample]["avg"]

    raw_image = standard_preprocessing_single_img(raw_image)
    reference = standard_preprocessing_single_img(reference)

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(raw_image[0][0], cmap="gray")
    ax[0].set_title("Raw Image")
    ax[1].imshow(reference[0][0], cmap="gray")
    ax[1].set_title("Reference Image")
    plt.show()

    n2n_metrics, n2n_denoised = evaluate_baseline(
        raw_image, reference, "n2n", config_path, override_config=override_config
    )
    n2n_ssm_metrics, n2n_ssm_denoised = evaluate_ssm_constraint(
        raw_image, reference, "n2n", config_path, override_config=override_config
    )
    n2n_metrics_last, n2n_denoised_last = evaluate_baseline(
        raw_image,
        reference,
        "n2n",
        config_path,
        override_config=override_config,
        last=True,
    )
    n2n_ssm_metrics_last, n2n_ssm_denoised_last = evaluate_ssm_constraint(
        raw_image,
        reference,
        "n2n",
        config_path,
        override_config=override_config,
        last=True,
    )
    metrics = {}
    metrics["n2n"] = n2n_metrics
    metrics["n2n_ssm"] = n2n_ssm_metrics
    all_metrics["n2n"] = n2n_metrics
    all_metrics["n2n_ssm"] = n2n_ssm_metrics
    display_metrics(metrics)
    metrics_last = {}
    metrics_last["n2n"] = n2n_metrics_last
    metrics_last["n2n_ssm"] = n2n_ssm_metrics_last
    all_metrics["n2n_last"] = n2n_metrics_last
    all_metrics["n2n_ssm_last"] = n2n_ssm_metrics_last
    display_metrics(metrics_last)
    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    ax[0][0].imshow(n2n_denoised, cmap="gray")
    ax[0][0].set_title("N2N Denoised")
    ax[0][1].imshow(n2n_ssm_denoised, cmap="gray")
    ax[0][1].set_title("N2N SSM Denoised")
    ax[1][0].imshow(n2n_denoised_last, cmap="gray")
    ax[1][0].set_title("N2N Denoised Last")
    ax[1][1].imshow(n2n_ssm_denoised_last, cmap="gray")
    ax[1][1].set_title("N2N SSM Denoised Last")
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
