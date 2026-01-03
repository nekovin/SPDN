import matplotlib.pyplot as plt
from ssm.evaluation import evaluate_baseline, evaluate_ssm_constraint
from spdn.utils.paths import get_sdoct_path
from spdn.utils import load_sdoct_dataset, display_metrics
import torch
import os
import random
from spdn.utils.config import get_config


def main():
    config_path = os.getenv("N2_CONFIG_PATH")

    config = get_config(config_path)

    n_patients = config["training"]["n_patients"]

    override_config = {
        "eval": {
            "ablation": f"patient_count/{n_patients}_patients",
            "n_patients": n_patients,
        }
    }

    all_metrics = {}

    "cuda" if torch.cuda.is_available() else "cpu"

    sdoct_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
    dataset = load_sdoct_dataset(sdoct_path)

    # random sample
    sample = random.choice(list(dataset.keys()))
    raw_image = dataset[sample]["raw"]
    reference = dataset[sample]["avg"][0][0]

    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].imshow(raw_image.cpu().numpy()[0][0], cmap="gray")
    ax[0].set_title("Raw Image")
    ax[1].imshow(reference.cpu().numpy(), cmap="gray")
    ax[1].set_title("Reference Image")
    plt.show()

    fig, ax = plt.subplots(3, 2, figsize=(15, 15))

    metrics = {}
    metrics_last = {}
    metrics_best = {}
    all_metrics = {}

    # Best checkpoints
    try:
        n2v_metrics, n2v_denoised = evaluate_baseline(
            raw_image, reference, "n2v", config_path, override_config=override_config
        )
        metrics["n2v"] = n2v_metrics
        all_metrics["n2v"] = n2v_metrics
        ax[0][0].imshow(n2v_denoised, cmap="gray")
        ax[0][0].set_title("N2V Denoised")
    except Exception as e:
        print(f"Error evaluating n2v: {e}")
        n2v_metrics = None
        n2v_denoised = None

    try:
        n2v_ssm_metrics, n2v_ssm_denoised = evaluate_ssm_constraint(
            raw_image, reference, "n2v", config_path, override_config=override_config
        )
        metrics["n2v_ssm"] = n2v_ssm_metrics
        all_metrics["n2v_ssm"] = n2v_ssm_metrics
        ax[0][1].imshow(n2v_ssm_denoised, cmap="gray")
        ax[0][1].set_title("N2V SSM Denoised")
    except Exception as e:
        print(f"Error evaluating n2v ssm: {e}")
        n2v_ssm_metrics = None
        n2v_ssm_denoised = None

    # Last checkpoints
    try:
        n2v_metrics_last, n2v_denoised_last = evaluate_baseline(
            raw_image,
            reference,
            "n2v",
            config_path,
            override_config=override_config,
            last=True,
        )
        metrics_last["n2v"] = n2v_metrics_last
        all_metrics["n2v_last"] = n2v_metrics_last
        ax[1][0].imshow(n2v_denoised_last, cmap="gray")
        ax[1][0].set_title("N2V Denoised Last")
    except Exception as e:
        print(f"Error evaluating n2v last: {e}")
        n2v_metrics_last = None
        n2v_denoised_last = None

    try:
        n2v_ssm_metrics_last, n2v_ssm_denoised_last = evaluate_ssm_constraint(
            raw_image,
            reference,
            "n2v",
            config_path,
            override_config=override_config,
            last=True,
        )
        metrics_last["n2v_ssm"] = n2v_ssm_metrics_last
        all_metrics["n2v_ssm_last"] = n2v_ssm_metrics_last
        ax[1][1].imshow(n2v_ssm_denoised_last, cmap="gray")
        ax[1][1].set_title("N2V SSM Denoised Last")
    except Exception as e:
        print(f"Error evaluating n2v ssm last: {e}")
        n2v_ssm_metrics_last = None
        n2v_ssm_denoised_last = None

    # Best metrics checkpoints
    try:
        n2v_metrics_best, n2v_denoised_best = evaluate_baseline(
            raw_image,
            reference,
            "n2v",
            config_path,
            override_config=override_config,
            best=True,
        )
        metrics_best["n2v"] = n2v_metrics_best
        all_metrics["n2v_best"] = n2v_metrics_best
        ax[2][0].imshow(n2v_denoised_best, cmap="gray")
        ax[2][0].set_title("N2V Denoised Best")
    except Exception as e:
        print(f"Error evaluating n2v best: {e}")
        n2v_metrics_best = None
        n2v_denoised_best = None

    try:
        n2v_ssm_metrics_best, n2v_ssm_denoised_best = evaluate_ssm_constraint(
            raw_image,
            reference,
            "n2v",
            config_path,
            override_config=override_config,
            best=True,
        )
        metrics_best["n2v_ssm"] = n2v_ssm_metrics_best
        all_metrics["n2v_ssm_best"] = n2v_ssm_metrics_best
        ax[2][1].imshow(n2v_ssm_denoised_best, cmap="gray")
        ax[2][1].set_title("N2V SSM Denoised Best")
    except Exception as e:
        print(f"Error evaluating n2v ssm best: {e}")
        n2v_ssm_metrics_best = None
        n2v_ssm_denoised_best = None

    display_metrics(metrics)
    display_metrics(metrics_last)
    display_metrics(metrics_best)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
