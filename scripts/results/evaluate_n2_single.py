import matplotlib.pyplot as plt
from spdn.evaluate.evaluate import evaluate_baseline, evaluate_ssm_constraint
from spdn.utils import load_sdoct_dataset, display_metrics
from spdn.utils.paths import get_ssm_checkpoint
import torch
import os
from spdn.utils.config import get_config
from spdn.data import get_paired_loaders
from spdn.evaluate.evaluate import load_model
from spdn.models.spdn.spdn_attention import SPDNAttention

device = "cuda" if torch.cuda.is_available() else "cpu"


def normalise_sample(raw_image, reference):
    """
    sample = random.choice(list(dataset.keys()))
    raw_image = dataset[sample]["raw"][0][0]
    raw_image = raw_image.cpu().numpy()
    print(f"Raw image shape: {raw_image.shape}")
    resized = cv2.resize(raw_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    print(f"Resized image shape: {resized.shape}")
    resized = normalize_image_np(resized)
    #raw_image = resized.to(device)
    raw_image = torch.from_numpy(resized).float()
    raw_image = raw_image.unsqueeze(0).unsqueeze(0)
    raw_image = raw_image.to(device)
    """
    import cv2
    from spdn.utils import normalize_image_np

    # Normalise the raw image
    raw_image = raw_image.cpu().numpy()
    resized = cv2.resize(raw_image, (256, 256), interpolation=cv2.INTER_LINEAR)
    resized = normalize_image_np(resized)
    raw_image = torch.from_numpy(resized).float()
    raw_image = raw_image.unsqueeze(0).unsqueeze(0)
    raw_image = raw_image.to(device)

    # Normalise the reference image
    reference = reference.cpu().numpy()
    resized_ref = cv2.resize(reference, (256, 256), interpolation=cv2.INTER_LINEAR)
    resized_ref = normalize_image_np(resized_ref)
    reference = torch.from_numpy(resized_ref).float()
    reference = reference.unsqueeze(0).unsqueeze(0)
    reference = reference.to(device)

    return raw_image, reference


def main(method=None, soct=True):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if soct:
        sdoct_path = (
            r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
        )
        dataset = load_sdoct_dataset(sdoct_path)

        # random sample
        # sample = random.choice(list(dataset.keys()))
        sample = "13"
        raw_image = dataset[sample]["raw"][0][0]
        reference = dataset[sample]["avg"][0][0]
        raw_image, reference = normalise_sample(raw_image, reference)
    else:
        config["training"]["n_images_per_patient"]
        batch_size = config["training"]["batch_size"]
        start = 1
        train_loader, val_loader = get_paired_loaders(start, 2, 5, batch_size)
        print(f"Train loader size: {len(train_loader.dataset)}")

        # Get actual tensor data, not shape
        batch_data = next(iter(train_loader))
        sample_batch = batch_data[0]  # Get input tensor batch
        print(f"Sample batch shape: {sample_batch.shape}")

        # Get first sample from batch
        raw_image = sample_batch[0]  # Shape: (channels, height, width)
        reference = sample_batch[0]  # Using same as reference

        # Add batch dimension
        raw_image = raw_image.unsqueeze(0).to(
            device
        )  # Shape: (1, channels, height, width)
        reference = reference.unsqueeze(0).to(device)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    for a in ax:
        a.axis("off")
    ax[0].imshow(raw_image.cpu().numpy()[0][0], cmap="gray")
    ax[1].imshow(reference.cpu().numpy()[0][0], cmap="gray")
    plt.tight_layout()
    plt.show()

    # save figure
    fig.savefig("../../results/raw_reference.png")

    fig, ax = plt.subplots(1, 2, figsize=(15, 15))

    metrics = {}
    all_metrics = {}

    config = get_config(config_path, override_config)

    config["training"]["method"] = method

    verbose = config["training"]["verbose"]

    speckle_module = SPDNAttention().to(device)
    try:
        print("Loading ssm model from checkpoint...")
        ssm_checkpoint_path = get_ssm_checkpoint()
        ssm_checkpoint = torch.load(ssm_checkpoint_path, map_location=device)
        speckle_module.load_state_dict(ssm_checkpoint["model_state_dict"])
        speckle_module.to(device)
        config["speckle_module"]["alpha"]
        print("SPDN model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Starting training from scratch.")
        raise e

    speckle_module.eval()
    speckle_output = speckle_module(raw_image)["flow_component"]
    speckle_output = speckle_output.detach().cpu()
    print(speckle_output)
    plt.figure(figsize=(10, 5))
    plt.imshow(speckle_output.cpu().numpy()[0][0], cmap="gray")
    plt.show()

    # Best checkpoints
    try:
        base_model, checkpoint = load_model(config, verbose, last=True, best=False)
        n2v_metrics, n2v_denoised = evaluate_baseline(
            raw_image, reference, method, base_model
        )
        print(
            f"Raw - min: {raw_image.min():.4f}, max: {raw_image.max():.4f}, mean: {raw_image.mean():.4f}"
        )
        print(
            f"Denoised - min: {n2v_denoised.min():.4f}, max: {n2v_denoised.max():.4f}, mean: {n2v_denoised.mean():.4f}"
        )

        metrics[f"{method}"] = n2v_metrics
        all_metrics[f"{method}"] = n2v_metrics
        print(f"Base: {checkpoint['epoch']}")
        ax[0].imshow(n2v_denoised, cmap="gray")
    except Exception as e:
        print(f"Error evaluating n2v: {e}")
        n2v_metrics = None
        n2v_denoised = raw_image.cpu().numpy()[0][0]

    try:
        config["speckle_module"]["use"] = True
        spdn_model, spdn_checkpoint = load_model(config, verbose, last=True, best=False)

        n2v_ssm_metrics, n2v_ssm_denoised = evaluate_ssm_constraint(
            raw_image, reference, method, spdn_model
        )
        print(
            f"Raw - min: {raw_image.min():.4f}, max: {raw_image.max():.4f}, mean: {raw_image.mean():.4f}"
        )
        print(
            f"Denoised - min: {n2v_ssm_denoised.min():.4f}, max: {n2v_ssm_denoised.max():.4f}, mean: {n2v_ssm_denoised.mean():.4f}"
        )
        metrics[f"{method}_ssm"] = n2v_ssm_metrics
        all_metrics[f"{method}_ssm"] = n2v_ssm_metrics
        print(f"SPDN: {spdn_checkpoint['epoch']}")
        ax[1].imshow(n2v_ssm_denoised, cmap="gray")
    except Exception as e:
        print(f"Error evaluating n2v ssm: {e}")
        n2v_ssm_metrics = None
        n2v_ssm_denoised = raw_image.cpu().numpy()[0][0]

    display_metrics(metrics)

    fig.tight_layout()
    plt.show()

    # plt.savefig(f"../../results/{method}.png")\
    fig.savefig(f"../../results/{method}.png")

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    for a in ax:
        a.axis("off")
    ax[0].imshow(n2v_denoised, cmap="gray")
    ax[1].imshow(n2v_ssm_denoised, cmap="gray")
    plt.tight_layout()
    plt.show()

    ###

    print("Comparing image patches...")
    print(f"Raw image shape: {raw_image.shape}")
    print(f"Reference image shape: {reference.shape}")
    print(
        f"N2V denoised shape: {n2v_denoised.shape if n2v_denoised is not None else 'N/A'}"
    )

    def compare_image_patches(img1, img2, figsize=(15, 8)):
        """
        Split two images into 3x3 patches and visualize comparison

        Args:
            img1, img2: Input images (H, W) or (H, W, C)
            figsize: Figure size for matplotlib
        """
        # Ensure images are same size
        assert img1.shape == img2.shape, "Images must have same dimensions"

        h, w = img1.shape[:2]
        patch_h, patch_w = h // 3, w // 3

        fig, axes = plt.subplots(3, 6, figsize=figsize)
        fig.suptitle("Image Patch Comparison (Left: Image 1, Right: Image 2)")

        for i in range(3):
            for j in range(3):
                # Extract patches
                y_start, y_end = i * patch_h, (i + 1) * patch_h
                x_start, x_end = j * patch_w, (j + 1) * patch_w

                patch1 = img1[y_start:y_end, x_start:x_end]
                patch2 = img2[y_start:y_end, x_start:x_end]

                # Plot patch from image 1
                axes[i][j * 2].imshow(patch1, cmap="gray")
                axes[i][j * 2].set_title(f"Img1 P{i}{j}")
                axes[i][j * 2].axis("off")

                # Plot patch from image 2
                axes[i][j * 2 + 1].imshow(patch2, cmap="gray")
                axes[i][j * 2 + 1].set_title(f"Img2 P{i}{j}")
                axes[i][j * 2 + 1].axis("off")

        plt.tight_layout()
        plt.show()

    # Usage example:
    compare_image_patches(raw_image[0][0].cpu().numpy(), n2v_denoised)
    compare_image_patches(raw_image[0][0].cpu().numpy(), n2v_ssm_denoised)

    def export_results_to_overleaf_table(metrics1, metrics2):
        table = (
            "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
        )
        table += f"Metric & {str(method).upper()} & {str(method).upper()}-SPDN \\\\\n\\hline\n"

        for m in zip(metrics1.keys(), metrics2.keys()):
            # if metric not a number, skip

            metric_name = m[0]
            if metric_name not in [
                "psnr",
                "ssim",
                "snr",
                "snr_change",
                "cnr",
                "cnr_change",
                "enl",
                "enl_change",
                "epi",
            ]:
                continue
            metric1 = float(metrics1[metric_name])
            metric2 = float(metrics2[metric_name])
            # metric_values = f"{metric1:.4f} & {metric2:.4f}"
            if metric1 > metric2:
                metric_values = f"\\textbf{{{metric1:.4f}}} & {metric2:.4f}"
            else:
                metric_values = f"{metric1:.4f} & \\textbf{{{metric2:.4f}}}"

            # Add to table
            table += f"{str(metric_name).upper()} & {metric_values} \\\\\n"

        # table += f"{metric_name} & {metric_values} & {metric_values} \\\\\n"

        table += "\\hline\n\\end{tabular}\n\\caption{Comparison of "
        # {method} and {method}}-SPDN Performance Metrics}\n\\end{table}"
        table += f"{method} and {method}-SPDN Performance Metrics"
        table += "}\n\\end{table}"

        table += "\n\label{tab:"
        table += f"{method}_comparison"
        table += "}\n"
        print(table)

    print(metrics)
    export_results_to_overleaf_table(metrics[method], metrics[f"{method}_ssm"])

    return metrics


if __name__ == "__main__":
    main()
