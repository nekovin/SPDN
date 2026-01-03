import matplotlib.pyplot as plt
from spdn.evaluation import (
    evaluate_baseline,
    evaluate_ssm_constraint,
    evaluate_progressssive_fusion_unet,
)
from spdn.utils.paths import get_sdoct_path
from spdn.utils import load_sdoct_dataset, display_metrics, display_grouped_metrics
from tqdm import tqdm
import torch
import os


def main():
    override_config = {
        "eval": {"ablation": "patient_count/10_patients", "n_patients": 10}
    }

    all_metrics = {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sdoct_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
    dataset = load_sdoct_dataset(sdoct_path)

    for patient_id, patient_data in tqdm(dataset.items(), desc="Evaluating patients"):
        raw_image = patient_data["raw"].to(device)
        reference = patient_data["avg"].to(device)[0][0]
        break

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(raw_image.cpu().numpy()[0][0], cmap="gray")
    ax[0].set_title("Raw Image")
    ax[1].imshow(reference.cpu().numpy(), cmap="gray")
    ax[1].set_title("Reference Image")
    plt.show()

    config_path = os.getenv("N2_CONFIG_PATH")

    n2n_metrics, n2n_denoised = evaluate_baseline(
        raw_image, reference, "n2n", config_path, override_config=override_config
    )
    n2n_ssm_metrics, n2n_ssm_denoised = evaluate_ssm_constraint(
        raw_image, reference, "n2n", config_path, override_config=override_config
    )
    metrics = {}
    metrics["n2n"] = n2n_metrics
    metrics["n2n_ssm"] = n2n_ssm_metrics
    all_metrics["n2n"] = n2n_metrics
    all_metrics["n2n_ssm"] = n2n_ssm_metrics
    display_metrics(metrics)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(n2n_denoised, cmap="gray")
    ax[0].set_title("N2N Denoised")
    ax[1].imshow(n2n_ssm_denoised, cmap="gray")
    ax[1].set_title("N2N SSM Denoised")
    plt.show()

    n2v_metrics, n2v_denoised = evaluate_baseline(
        raw_image, reference, "n2v", config_path, override_config=override_config
    )
    n2v_ssm_metrics, n2v_ssm_denoised = evaluate_ssm_constraint(
        raw_image, reference, "n2v", config_path, override_config=override_config
    )
    metrics = {}
    metrics["n2v"] = n2v_metrics
    metrics["n2v_ssm"] = n2v_ssm_metrics
    all_metrics["n2v"] = n2v_metrics
    all_metrics["n2v_ssm"] = n2v_ssm_metrics
    display_metrics(metrics)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(n2v_denoised, cmap="gray")
    ax[0].set_title("N2V Denoised")
    ax[1].imshow(n2v_ssm_denoised, cmap="gray")
    ax[1].set_title("N2V SSM Denoised")
    plt.show()

    n2s_metrics, n2s_denoised = evaluate_baseline(
        raw_image, reference, "n2s", config_path, override_config=override_config
    )
    n2s_ssm_metrics, n2s_ssm_denoised = evaluate_ssm_constraint(
        raw_image, reference, "n2s", config_path, override_config=override_config
    )
    metrics = {}
    metrics["n2s"] = n2s_metrics
    metrics["n2s_ssm"] = n2s_ssm_metrics
    all_metrics["n2s"] = n2s_metrics
    all_metrics["n2s_ssm"] = n2s_ssm_metrics
    display_metrics(metrics)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(n2s_denoised, cmap="gray")
    ax[0].set_title("N2S Denoised")
    ax[1].imshow(n2s_ssm_denoised, cmap="gray")
    ax[1].set_title("N2S SSM Denoised")
    plt.show()

    prog_metrics, prog_image = evaluate_progressssive_fusion_unet(
        raw_image, reference, device
    )

    metrics = {}
    metrics["pfn"] = prog_metrics
    display_grouped_metrics(metrics)

    all_metrics["pfn"] = prog_metrics

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))

    ax[0].imshow(prog_image, cmap="gray")

    ax[0].set_title("Progressive Fusion Denoised")
    plt.show()

    display_grouped_metrics(all_metrics)
    display_metrics(all_metrics)


if __name__ == "__main__":
    main()
