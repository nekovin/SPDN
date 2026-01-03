import matplotlib.pyplot as plt
from spdn.evaluate.evaluate import evaluate_baseline, evaluate_ssm_constraint
from spdn.utils import load_sdoct_dataset, display_metrics
from tqdm import tqdm
import torch
import os
import numpy as np
from spdn.utils.config import get_config
from spdn.data import get_paired_loaders
from spdn.evaluate.evaluate import load_model
from scipy.stats import ttest_rel, wilcoxon
from scipy.io import loadmat

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set matplotlib style for prettier figures
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 11,
        "figure.titlesize": 16,
    }
)


def normalise_sample(raw_image, reference):
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


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1))
        / (n1 + n2 - 2)
    )
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def perform_statistical_analysis(aggregated_metrics, method):
    """Perform comprehensive statistical analysis"""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)

    base_method = method
    spdn_method = f"{method}_ssm"

    if base_method not in aggregated_metrics or spdn_method not in aggregated_metrics:
        print("Insufficient data for statistical analysis")
        return

    # Statistical results storage
    stat_results = {}

    print(f"\nComparing {base_method.upper()} vs {spdn_method.upper()}")
    print("-" * 60)

    # LaTeX table for statistical results
    stat_table = "\\begin{table}[htbp]\n\\centering\n"
    stat_table += "\\begin{tabular}{|l|c|c|c|c|c|c|}\n\\hline\n"
    stat_table += "Metric & Mean Diff & t-stat & p-value & Effect Size & Wilcoxon p & Significance \\\\\n\\hline\n"

    for metric in ["psnr", "ssim", "snr", "cnr", "enl", "epi"]:
        if (
            metric in aggregated_metrics[base_method]
            and metric in aggregated_metrics[spdn_method]
        ):
            base_values = np.array(aggregated_metrics[base_method][metric]["values"])
            spdn_values = np.array(aggregated_metrics[spdn_method][metric]["values"])

            if len(base_values) != len(spdn_values) or len(base_values) < 3:
                print(f"Skipping {metric}: insufficient paired samples")
                continue

            # Paired t-test
            t_stat, t_p_value = ttest_rel(spdn_values, base_values)

            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                w_stat, w_p_value = wilcoxon(spdn_values, base_values)
            except ValueError:
                w_p_value = np.nan

            # Effect size (Cohen's d)
            cohens_d = calculate_cohens_d(spdn_values, base_values)

            # Mean difference
            mean_diff = np.mean(spdn_values) - np.mean(base_values)

            # Significance classification
            if t_p_value < 0.001:
                significance = "***"
            elif t_p_value < 0.01:
                significance = "**"
            elif t_p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"

            # Effect size interpretation
            if abs(cohens_d) < 0.2:
                effect_interpretation = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interpretation = "small"
            elif abs(cohens_d) < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"

            # Store results
            stat_results[metric] = {
                "mean_diff": mean_diff,
                "t_stat": t_stat,
                "t_p_value": t_p_value,
                "w_p_value": w_p_value,
                "cohens_d": cohens_d,
                "effect_interpretation": effect_interpretation,
                "significance": significance,
                "base_mean": np.mean(base_values),
                "spdn_mean": np.mean(spdn_values),
                "base_std": np.std(base_values),
                "spdn_std": np.std(spdn_values),
            }

            # Print detailed results
            print(f"\n{metric.upper()}:")
            print(
                f"  {base_method.upper()}: {np.mean(base_values):.4f} ± {np.std(base_values):.4f}"
            )
            print(
                f"  {spdn_method.upper()}: {np.mean(spdn_values):.4f} ± {np.std(spdn_values):.4f}"
            )
            print(f"  Mean Difference: {mean_diff:.4f}")
            print(
                f"  Paired t-test: t = {t_stat:.3f}, p = {t_p_value:.6f} {significance}"
            )
            print(
                f"  Effect Size (Cohen's d): {cohens_d:.3f} ({effect_interpretation})"
            )
            if not np.isnan(w_p_value):
                print(f"  Wilcoxon test: p = {w_p_value:.6f}")

            # Add to LaTeX table
            stat_table += f"{metric.upper()} & {mean_diff:.3f} & {t_stat:.3f} & {t_p_value:.3f} & {cohens_d:.3f} & "
            if not np.isnan(w_p_value):
                stat_table += f"{w_p_value:.3f} & {significance} \\\\\n"
            else:
                stat_table += f"-- & {significance} \\\\\n"

    stat_table += "\\hline\n\\end{tabular}\n"
    stat_table += f"\\caption{{Statistical comparison between {base_method.upper()} and {spdn_method.upper()}. *** p<0.001, ** p<0.01, * p<0.05, ns = not significant}}\n"
    stat_table += f"\\label{{{{tab:{method}_statistical_analysis}}}}\n\\end{{table}}"

    print("\n" + "=" * 60)
    print("SUMMARY OF STATISTICAL FINDINGS")
    print("=" * 60)

    significant_improvements = []
    significant_degradations = []

    for metric, results in stat_results.items():
        if results["significance"] != "ns":
            if results["mean_diff"] > 0:
                significant_improvements.append(
                    f"{metric.upper()} (p={results['t_p_value']:.3f}, d={results['cohens_d']:.2f})"
                )
            else:
                significant_degradations.append(
                    f"{metric.upper()} (p={results['t_p_value']:.3f}, d={results['cohens_d']:.2f})"
                )

    print(f"\nSignificant improvements with {spdn_method.upper()}:")
    if significant_improvements:
        for improvement in significant_improvements:
            print(f"  • {improvement}")
    else:
        print("  None")

    print(f"\nSignificant degradations with {spdn_method.upper()}:")
    if significant_degradations:
        for degradation in significant_degradations:
            print(f"  • {degradation}")
    else:
        print("  None")

    print("\nLaTeX Statistical Table:")
    print(stat_table)

    # Create statistical visualization
    create_statistical_plots(stat_results, method)

    return stat_results


def create_statistical_plots(stat_results, method):
    """Create visualizations for statistical analysis"""

    # 1. Effect sizes plot
    metrics = list(stat_results.keys())
    effect_sizes = [stat_results[metric]["cohens_d"] for metric in metrics]
    p_values = [stat_results[metric]["t_p_value"] for metric in metrics]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Effect sizes
    colors = ["red" if es < 0 else "green" for es in effect_sizes]
    ax1.bar(range(len(metrics)), effect_sizes, color=colors, alpha=0.7)
    ax1.set_xlabel("Metrics")
    ax1.set_ylabel("Cohen's d (Effect Size)")
    ax1.set_title(f"Effect Sizes: {method.upper()}-SPDN vs {method.upper()}")
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels([m.upper() for m in metrics], rotation=45)
    ax1.axhline(y=0, color="black", linestyle="-", alpha=0.3)
    ax1.axhline(y=0.2, color="gray", linestyle="--", alpha=0.5, label="Small effect")
    ax1.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Medium effect")
    ax1.axhline(y=0.8, color="gray", linestyle="--", alpha=0.5, label="Large effect")
    ax1.axhline(y=-0.2, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(y=-0.5, color="gray", linestyle="--", alpha=0.5)
    ax1.axhline(y=-0.8, color="gray", linestyle="--", alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # P-values
    log_p_values = [-np.log10(p) for p in p_values]
    colors2 = ["red" if p >= 0.05 else "green" for p in p_values]
    ax2.bar(range(len(metrics)), log_p_values, color=colors2, alpha=0.7)
    ax2.set_xlabel("Metrics")
    ax2.set_ylabel("-log10(p-value)")
    ax2.set_title("Statistical Significance")
    ax2.set_xticks(range(len(metrics)))
    ax2.set_xticklabels([m.upper() for m in metrics], rotation=45)
    ax2.axhline(
        y=-np.log10(0.05), color="red", linestyle="--", alpha=0.7, label="p = 0.05"
    )
    ax2.axhline(
        y=-np.log10(0.01), color="orange", linestyle="--", alpha=0.7, label="p = 0.01"
    )
    ax2.axhline(
        y=-np.log10(0.001), color="green", linestyle="--", alpha=0.7, label="p = 0.001"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 2. Before/After comparison with confidence intervals
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, metric in enumerate(metrics[:6]):
        if i >= 6:
            break

        results = stat_results[metric]

        # Data for plotting
        categories = [f"{method.upper()}", f"{method.upper()}-SPDN"]
        means = [results["base_mean"], results["spdn_mean"]]
        stds = [results["base_std"], results["spdn_std"]]

        # Bar plot with error bars
        axes[i].bar(
            categories,
            means,
            yerr=stds,
            capsize=5,
            color=["steelblue", "darkorange"],
            alpha=0.8,
        )

        axes[i].set_title(
            f"{metric.upper()}\n(p={results['t_p_value']:.3f}, d={results['cohens_d']:.2f})"
        )
        axes[i].set_ylabel(metric.upper())

        # Add significance annotation
        if results["significance"] != "ns":
            y_max = max(means) + max(stds) * 1.1
            axes[i].annotate(
                results["significance"],
                xy=(0.5, y_max),
                xytext=(0.5, y_max * 1.05),
                ha="center",
                va="bottom",
                fontsize=14,
                fontweight="bold",
            )

        axes[i].grid(True, alpha=0.3)

    plt.suptitle(
        "Metric Comparison with Statistical Significance",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.show()


def main(method=None, soct=True, chiu=False):
    config_path = os.getenv("N2_CONFIG_PATH")
    config = get_config(config_path)
    n_patients = config["training"]["n_patients"]

    override_config = {
        "eval": {
            "ablation": f"patient_count/{n_patients}_patients",
            "n_patients": n_patients,
        }
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if soct:
        sdoct_path = (
            r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
        )
        dataset = load_sdoct_dataset(sdoct_path)
        samples = list(dataset.keys())
    elif chiu:
        mat_path = r"C:\Datasets\OCTData\2011_IOVS_Chiu\Chiu_IOVS_2011\Reproducibility Study\Patient1_Degree00.mat"
        dataset = loadmat(mat_path)
        samples = list(dataset["images"])
    else:
        n_images_per_patient = config["training"]["n_images_per_patient"]
        batch_size = config["training"]["batch_size"]
        start = 1
        train_loader, val_loader = get_paired_loaders(start, 2, 5, batch_size)
        print(f"Train loader size: {len(train_loader.dataset)}")

        batch_data = next(iter(train_loader))
        sample_batch = batch_data[0]
        raw_image = sample_batch[0].unsqueeze(0).to(device)
        reference = sample_batch[0].unsqueeze(0).to(device)
        samples = ["single_sample"]  # For consistency

    # Load models once
    config = get_config(config_path, override_config)
    config["training"]["method"] = method
    verbose = config["training"]["verbose"]

    base_model, checkpoint = load_model(config, verbose, last=False, best=False)
    config["speckle_module"]["use"] = True
    spdn_model, checkpoint = load_model(config, verbose, last=False, best=False)

    # Accumulate metrics across all samples
    all_sample_metrics = []
    sample_images = []

    print(f"Evaluating {len(samples)} samples...")

    for sample_idx, sample in enumerate(tqdm(samples, desc="Processing samples")):
        if soct:
            raw_image = dataset[sample]["raw"][0][0]
            reference = dataset[sample]["avg"][0][0]
            raw_image, reference = normalise_sample(raw_image, reference)

        # Store sample for visualization
        if sample_idx < 6:  # Store first 6 for grid display
            sample_images.append(
                {
                    "raw": raw_image.cpu().numpy()[0][0],
                    "reference": reference.cpu().numpy()[0][0],
                    "sample_id": sample,
                }
            )

        sample_metrics = {}

        # Evaluate baseline
        try:
            n2v_metrics, n2v_denoised = evaluate_baseline(
                raw_image, reference, method, base_model
            )
            sample_metrics[f"{method}"] = n2v_metrics
            if sample_idx < 6:
                sample_images[-1]["n2v_denoised"] = n2v_denoised
        except Exception as e:
            print(f"Error evaluating {method} for sample {sample}: {e}")
            continue

        # Evaluate with SSM
        try:
            n2v_ssm_metrics, n2v_ssm_denoised = evaluate_ssm_constraint(
                raw_image, reference, method, spdn_model
            )
            sample_metrics[f"{method}_ssm"] = n2v_ssm_metrics
            if sample_idx < 6:
                sample_images[-1]["n2v_ssm_denoised"] = n2v_ssm_denoised
        except Exception as e:
            print(f"Error evaluating {method}_ssm for sample {sample}: {e}")
            continue

        all_sample_metrics.append(sample_metrics)

    # Aggregate metrics across all samples
    aggregated_metrics = {}
    if all_sample_metrics:
        for method_key in all_sample_metrics[0].keys():
            method_metrics = {}
            for metric_name in all_sample_metrics[0][method_key].keys():
                if metric_name in ["psnr", "ssim", "snr", "cnr", "enl", "epi"]:
                    values = [
                        sample[method_key][metric_name]
                        for sample in all_sample_metrics
                        if method_key in sample and metric_name in sample[method_key]
                    ]
                    if values:
                        method_metrics[metric_name] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                            "values": values,
                        }
            aggregated_metrics[method_key] = method_metrics

    # Create prettier visualizations
    # 1. Sample grid (first 6 samples)
    if sample_images:
        fig, axes = plt.subplots(3, 6, figsize=(20, 12))
        fig.suptitle("Sample Results Overview", fontsize=16, fontweight="bold")

        for i, sample_data in enumerate(sample_images[:6]):
            if i >= 6:
                break

            # Raw image
            axes[0, i].imshow(sample_data["raw"], cmap="gray")
            axes[0, i].set_title(f"Sample {sample_data['sample_id']}\nRaw", fontsize=10)
            axes[0, i].axis("off")

            # Method denoised
            if "n2v_denoised" in sample_data:
                axes[1, i].imshow(sample_data["n2v_denoised"], cmap="gray")
                axes[1, i].set_title(f"{method.upper()}", fontsize=10)
                axes[1, i].axis("off")

            # Method + SSM denoised
            if "n2v_ssm_denoised" in sample_data:
                axes[2, i].imshow(sample_data["n2v_ssm_denoised"], cmap="gray")
                axes[2, i].set_title(f"{method.upper()}-SPDN", fontsize=10)
                axes[2, i].axis("off")

        plt.tight_layout()
        plt.show()

    def export_results_to_overleaf_table(metrics1, metrics2):
        table = (
            "\\begin{table}[htbp]\n\\centering\n\\begin{tabular}{|c|c|c|}\n\\hline\n"
        )
        table += f"Metric & {str(method).upper()} & {str(method).upper()}-SPDN \\\\\n\\hline\n"

        for metric_name in ["psnr", "ssim", "snr", "cnr", "enl", "epi"]:
            if metric_name in metrics1 and metric_name in metrics2:
                metric1 = float(metrics1[metric_name])
                metric2 = float(metrics2[metric_name])

                if metric1 > metric2:
                    metric_values = f"\\textbf{{{metric1:.4f}}} & {metric2:.4f}"
                else:
                    metric_values = f"{metric1:.4f} & \\textbf{{{metric2:.4f}}}"

                table += f"{str(metric_name).upper()} & {metric_values} \\\\\n"

        table += "\\hline\n\\end{tabular}\n"
        table += f"\\caption{{Comparison of {method} and {method}-SPDN Performance Metrics (Mean across all samples)}}\n"
        table += f"\\label{{tab:{method}_comparison}}\n\\end{table}"

        print("\nLaTeX Table:")
        print(table)

    # 2. Metrics comparison plot
    if aggregated_metrics:
        metrics_to_plot = ["psnr", "ssim", "snr", "cnr"]
        available_metrics = [
            m for m in metrics_to_plot if m in aggregated_metrics[f"{method}"]
        ]

        if available_metrics:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(
                "Performance Metrics Comparison", fontsize=16, fontweight="bold"
            )
            axes = axes.flatten()

            for i, metric in enumerate(available_metrics[:4]):
                if i >= 4:
                    break

                base_values = aggregated_metrics[f"{method}"][metric]["values"]
                ssm_values = aggregated_metrics[f"{method}_ssm"][metric]["values"]

                x = np.arange(len(base_values))
                width = 0.35

                axes[i].bar(
                    x - width / 2,
                    base_values,
                    width,
                    label=f"{method.upper()}",
                    alpha=0.8,
                    color="steelblue",
                )
                axes[i].bar(
                    x + width / 2,
                    ssm_values,
                    width,
                    label=f"{method.upper()}-SPDN",
                    alpha=0.8,
                    color="darkorange",
                )

                axes[i].set_title(f"{metric.upper()} Comparison", fontweight="bold")
                axes[i].set_xlabel("Sample Index")
                axes[i].set_ylabel(metric.upper())
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

    # Display aggregated metrics
    if aggregated_metrics:
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS ACROSS ALL SAMPLES")
        print("=" * 60)

        summary_metrics = {}
        for method_key, metrics in aggregated_metrics.items():
            summary_metrics[method_key] = {k: v["mean"] for k, v in metrics.items()}

        display_metrics(summary_metrics)

        # Export to LaTeX table
        export_results_to_overleaf_table(
            summary_metrics[method], summary_metrics[f"{method}_ssm"]
        )

        # Perform statistical analysis
        stat_results = perform_statistical_analysis(aggregated_metrics, method)

    return aggregated_metrics, stat_results if "stat_results" in locals() else None


if __name__ == "__main__":
    main()
