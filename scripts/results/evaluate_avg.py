import os
import numpy as np
import torch
import pandas as pd
from skimage import io
from skimage.transform import resize
from tqdm import tqdm
import matplotlib.pyplot as plt

from utils.metrics import display_grouped_metrics

from evaluation.evaluate_pfn import evaluate_progressssive_fusion_unet
from evaluation.evaluate_n2_baselines import evaluate_n2, evaluate_n2_with_ssm


def plot_images(images, metrics_df=None):
    """
    Plot multiple images in a grid with appropriate handling for single rows.

    Args:
        images: Dictionary of images with their labels as keys
        metrics_df: Optional metrics DataFrame for titles (not used currently)
    """
    cols = 4
    rows = len(images) // cols + (len(images) % cols > 0)
    fig, ax = plt.subplots(rows, cols, figsize=(20, 5 * rows))

    # Handle the case when there's only one row
    if rows == 1:
        ax = np.array([ax])  # Convert to 1D array if it's not already
        ax = ax.reshape(1, -1)  # Reshape to 2D array with one row

    # Handle the case when there's only one image
    if len(images) == 1:
        key, image = next(iter(images.items()))
        ax[0, 0].imshow(image, cmap="gray")
        ax[0, 0].set_title(key)
        ax[0, 0].axis("off")
    else:
        for i, (key, image) in enumerate(images.items()):
            row, col = i // cols, i % cols
            if row < rows and col < cols:  # Check bounds
                ax[row, col].imshow(image, cmap="gray")
                ax[row, col].set_title(key)
                ax[row, col].axis("off")

    # Turn off any unused subplots
    for i in range(len(images), rows * cols):
        row, col = i // cols, i % cols
        ax[row, col].axis("off")

    plt.tight_layout()
    plt.show()


def load_sdoct_dataset(dataset_path, target_size=(256, 256)):
    """Load all images from the SDOCT dataset and resize them.

    Args:
        dataset_path: Path to the SDOCT dataset directory
        target_size: Size to resize images to

    Returns:
        Dictionary containing patient data with raw and averaged images
    """
    sdoct_data = {}
    patients = os.listdir(dataset_path)

    print(f"Loading SDOCT dataset from {dataset_path}")
    i = 0
    for patient in tqdm(patients, desc="Loading patients"):
        patient_path = os.path.join(dataset_path, patient)
        avg_path = os.path.join(patient_path, f"{patient}_Averaged Image.tif")
        raw_path = os.path.join(patient_path, f"{patient}_Raw Image.tif")

        try:
            # Check if both files exist
            if not os.path.exists(avg_path) or not os.path.exists(raw_path):
                print(f"Missing files for patient {patient}")
                continue

            # Load and resize images
            raw_img = io.imread(raw_path)
            avg_img = io.imread(avg_path)

            # Normalize images to 0-1 range if needed
            if raw_img.max() > 1.0:
                raw_img = raw_img / 255.0
            if avg_img.max() > 1.0:
                avg_img = avg_img / 255.0

            # Resize images
            raw_img = resize(raw_img, target_size, anti_aliasing=True)
            avg_img = resize(avg_img, target_size, anti_aliasing=True)

            # Convert to PyTorch tensors
            raw_tensor = torch.from_numpy(raw_img).float().unsqueeze(0).unsqueeze(0)
            avg_tensor = torch.from_numpy(avg_img).float().unsqueeze(0).unsqueeze(0)

            # Store in dictionary
            sdoct_data[patient] = {
                "raw": raw_tensor,
                "avg": avg_tensor,
                "raw_np": raw_img,
                "avg_np": avg_img,
            }

            i += 1
            if i == 5:
                sdoct_data

        except Exception as e:
            print(f"Error processing patient {patient}: {e}")

    print(f"Successfully loaded {len(sdoct_data)} patients")
    return sdoct_data


def evaluate_all_models(dataset, device="cuda"):
    """Evaluate all models on the entire dataset.

    Args:
        dataset: Dictionary containing patient data
        device: Device to run evaluation on

    Returns:
        DataFrame with evaluation metrics for all models and patients
    """
    all_metrics = []

    for patient_id, patient_data in tqdm(dataset.items(), desc="Evaluating patients"):
        raw_image = patient_data["raw"].to(device)
        reference = patient_data["avg"].to(device)

        # Initialize metrics for this patient
        patient_metrics = {"patient_id": patient_id}

        # Store original images
        denoised_images = {
            "original": patient_data["raw_np"],
            "avg": patient_data["avg_np"],
        }

        n2_metrics, denoised_images = evaluate_n2(
            patient_metrics.copy(), denoised_images, raw_image, reference
        )

        pfn_metrics, pfn_image = evaluate_progressssive_fusion_unet(
            raw_image, reference, device
        )
        n2_metrics["pfn"] = pfn_metrics
        denoised_images["pfn"] = pfn_image

        final_metrics, _ = evaluate_n2_with_ssm(
            n2_metrics.copy(), denoised_images, raw_image, reference
        )

        all_metrics.append(final_metrics)

    metrics_df = pd.DataFrame(all_metrics)
    return metrics_df, denoised_images


def compute_average_metrics(metrics_df):
    """Compute average metrics across all patients.

    Args:
        metrics_df: DataFrame with metrics for all patients

    Returns:
        DataFrame with average metrics
    """
    # Check if metrics_df is a DataFrame
    if not isinstance(metrics_df, pd.DataFrame):
        # Convert to DataFrame if it's a list of dictionaries
        metrics_df = pd.DataFrame(metrics_df)

    # Initialize result dictionary
    avg_metrics = {}

    # Get all column names except patient_id
    if "patient_id" in metrics_df.columns:
        model_columns = [col for col in metrics_df.columns if col != "patient_id"]
    else:
        model_columns = metrics_df.columns

    for model in model_columns:
        if model in metrics_df.columns:
            # If the model column contains dictionaries, extract metrics from dictionaries
            if isinstance(metrics_df[model].iloc[0], dict):
                metrics_dict = metrics_df[model].tolist()
                metric_keys = set()
                for m in metrics_dict:
                    if m:
                        metric_keys.update(m.keys())

                avg_metrics[model] = {}
                for metric in metric_keys:
                    values = [m.get(metric) for m in metrics_dict if m and metric in m]
                    values = [v for v in values if isinstance(v, (int, float))]

                    if values:
                        avg_metrics[model][metric] = {
                            "mean": np.mean(values),
                            "std": np.std(values),
                        }
            # If the model column contains numeric values directly
            elif pd.api.types.is_numeric_dtype(metrics_df[model]):
                values = metrics_df[model].dropna().tolist()
                if values:
                    avg_metrics[model] = {
                        "mean": np.mean(values),
                        "std": np.std(values),
                    }

    return pd.DataFrame(avg_metrics)


def display_grouped_metrics(metrics_df):
    """Display grouped metrics with appropriate highlighting for best values."""

    if not isinstance(metrics_df, pd.DataFrame):
        metrics_df = pd.DataFrame(metrics_df)

    flattened_metrics = {}

    for col in metrics_df.columns:
        if col == "patient_id":
            flattened_metrics[col] = metrics_df[col]
            continue

        # Check if column contains dictionaries
        if metrics_df[col].apply(isinstance, args=(dict,)).any():
            # Get all possible metric keys
            all_keys = set()
            for entry in metrics_df[col].dropna():
                if isinstance(entry, dict):
                    all_keys.update(entry.keys())

            # Create separate columns for each metric
            for metric in all_keys:
                col_name = f"{col}_{metric}"
                flattened_metrics[col_name] = metrics_df[col].apply(
                    lambda x: x.get(metric) if isinstance(x, dict) else None
                )
        else:
            # Just copy the column if it doesn't contain dictionaries
            flattened_metrics[col] = metrics_df[col]

    # Create a new DataFrame with flattened metrics
    flat_df = pd.DataFrame(flattened_metrics)

    # Filter to only include numeric columns for styling
    numeric_cols = flat_df.select_dtypes(include=["number"]).columns

    # Define a function to highlight the maximum value in each numeric column
    def highlight_max(s):
        is_max = s == s.max()
        return ["font-weight: bold" if v else "" for v in is_max]

    # Apply styling only to numeric columns
    if len(numeric_cols) > 0:
        styled_df = flat_df.style.apply(highlight_max, subset=numeric_cols)
        return styled_df
    else:
        return flat_df


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


def display_grouped_metrics(metrics):
    """
    Display metrics grouped by model type with proper formatting and highlighting.

    Args:
        metrics: Dictionary containing metrics (potentially nested)

    Returns:
        Dictionary of DataFrames for each model group
    """
    import pandas as pd
    from IPython.display import display

    # Define base schemas to group by
    base_schemas = ["n2n", "n2v", "n2s", "pfn", "n2n_ssm", "n2v_ssm", "n2s_ssm"]

    # Initialize storage for grouped metrics
    schema_tables = {schema: {} for schema in base_schemas}
    other_metrics = {}

    # Process each metric key
    for metric_key, metric_value in metrics.items():
        # Skip non-metric keys
        if metric_key == "patient_id":
            continue

        # Find which base schema this metric belongs to
        matched_schema = None
        for schema in base_schemas:
            if schema in metric_key:
                matched_schema = schema
                break

        # Store metric in appropriate group
        if matched_schema:
            # Extract metric name (e.g., 'ssim' from 'n2n_ssim')
            parts = metric_key.split("_")
            if len(parts) > 1:
                metric_name = "_".join(parts[1:])  # For metrics like n2n_ssim
            else:
                metric_name = metric_key  # For the model name itself

            schema_tables[matched_schema][metric_name] = metric_value
        else:
            other_metrics[metric_key] = metric_value

    # Function to process nested dictionaries into flat data
    def flatten_metric_dict(metric_dict):
        flat_dict = {}
        for key, value in metric_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float)) and not pd.isna(sub_value):
                        flat_dict[f"{key}_{sub_key}"] = sub_value
            elif isinstance(value, (int, float)) and not pd.isna(value):
                flat_dict[key] = value
        return flat_dict

    # Display tables for each schema group
    all_dfs = {}

    for schema, data in schema_tables.items():
        if data:  # Only process if we have data
            # Flatten nested metrics
            flat_data = flatten_metric_dict(data)

            if flat_data:  # Ensure we have data after flattening
                # Convert to DataFrame
                df = pd.DataFrame([flat_data])

                # Custom styling function to highlight max values
                def highlight_max(s):
                    try:
                        is_max = s == s.max()
                        return ["font-weight: bold" if v else "" for v in is_max]
                    except:
                        return ["" for _ in range(len(s))]

                # Apply styling
                styled_df = df.style.apply(highlight_max, axis=1)

                print(f"\n--- {schema.upper()} Model Results ---")
                display(styled_df)
                all_dfs[schema] = df

    # Add other metrics if there are any
    if other_metrics:
        flat_other = flatten_metric_dict(other_metrics)
        if flat_other:
            df_other = pd.DataFrame([flat_other])
            print("\n--- Other Metrics ---")
            display(df_other)
            all_dfs["other"] = df_other

    return all_dfs


def display_avg_metrics(avg_metrics):
    """
    Display average metrics in a properly formatted table.
    This function extracts and flattens the mean values from nested dictionaries.

    Args:
        avg_metrics: Dictionary containing average metrics with nested means/stds
    """
    import pandas as pd
    from IPython.display import display

    # Extract mean values from nested dictionaries
    flat_metrics = {}

    for model, metrics in avg_metrics.items():
        if isinstance(metrics, dict):
            flat_metrics[model] = {}
            for metric_name, value in metrics.items():
                if isinstance(value, dict) and "mean" in value:
                    # Extract just the mean value for display
                    flat_metrics[model][metric_name] = value["mean"]
                else:
                    flat_metrics[model][metric_name] = value
        else:
            flat_metrics[model] = metrics

    # Convert to DataFrame for display
    df = pd.DataFrame(flat_metrics)

    # Define styling function that can safely handle numeric data
    def highlight_max(s):
        try:
            # Convert to numeric, coerce non-numeric to NaN
            s_numeric = pd.to_numeric(s, errors="coerce")
            # Only compare values that could be converted
            is_max = s_numeric == s_numeric.max()
            return ["font-weight: bold" if v else "" for v in is_max]
        except:
            # Return empty formatting if comparison fails
            return ["" for _ in range(len(s))]

    # Apply styling to numeric columns
    styled_df = df.style.apply(highlight_max, axis=1)
    display(styled_df)

    return df


from utils.config import get_config
from IPython.display import clear_output


def main(eval_override=None):
    if eval_override is None:
        eval_override = {
            "prog_config": {
                "": "",
            },
            "n2_eval": {
                "": "",
            },
        }

    eval_config = get_config(
        r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\eval.yaml",
        eval_override,
    )

    print(eval_config)
    eval_config["eval"]["sample"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sdoct_path = r"C:\Datasets\OCTData\boe-13-12-6357-d001\Sparsity_SDOCT_DATASET_2012"
    dataset = load_sdoct_dataset(sdoct_path)

    all_patient_metrics = {}

    patient = eval_config["data"]["patient"]

    first_patient = list(dataset.keys())[patient]
    visualization_images = {
        "original": dataset[first_patient]["raw_np"],
        "avg": dataset[first_patient]["avg_np"],
    }

    if eval_config["eval"]["sample"]:
        # Process only the first patient
        # print(f"Sampling single patient: {first_patient}")
        patient_data = dataset[first_patient]
        patient_id = first_patient

        raw_image = patient_data["raw"].to(device)
        reference = patient_data["avg"].to(device)[0][0]

        # Initialize metrics for this patient
        metrics = {}
        denoised_images = {
            "original": patient_data["raw_np"],
            "avg": patient_data["avg_np"],
        }

        n2_config_path = (
            r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"
        )
        try:
            metrics, denoised_images = evaluate_n2(
                metrics,
                denoised_images,
                n2_config_path,
                eval_override["n2_eval"],
                raw_image,
                reference,
            )
        except Exception as e:
            print(f"Error evaluating N2: {e}")
            raise e

        if not eval_config["exclude"]["pfn"]:
            pfn_config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\pfn_config.yaml"
            prog_metrics, prog_image = evaluate_progressssive_fusion_unet(
                raw_image,
                reference,
                device,
                pfn_config_path,
                eval_override["prog_config"],
            )
            metrics["pfn"] = prog_metrics
            denoised_images["pfn"] = prog_image

        metrics, denoised_images = evaluate_n2_with_ssm(
            metrics,
            denoised_images,
            n2_config_path,
            eval_override["n2_eval"],
            raw_image,
            reference,
        )

        visualization_images = denoised_images

        # Store metrics for this patient
        for key, value in metrics.items():
            all_patient_metrics[key] = [value]
    else:
        for patient_id, patient_data in tqdm(
            dataset.items(), desc="Evaluating patients"
        ):
            raw_image = patient_data["raw"].to(device)
            reference = patient_data["avg"].to(device)[0][0]

            # Initialize metrics for this patient
            metrics = {}
            denoised_images = {
                "original": patient_data["raw_np"],
                "avg": patient_data["avg_np"],
            }
            n2_config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\n2_config.yaml"
            try:
                metrics, denoised_images = evaluate_n2(
                    metrics,
                    denoised_images,
                    n2_config_path,
                    eval_override["n2_eval"],
                    raw_image,
                    reference,
                )
            except Exception as e:
                raise e

            if not eval_config["exclude"]["pfn"]:
                pfn_config_path = r"C:\Users\CL-11\OneDrive\Repos\OCTDenoisingFinal\configs\pfn_config.yaml"
                prog_metrics, prog_image = evaluate_progressssive_fusion_unet(
                    raw_image,
                    reference,
                    device,
                    pfn_config_path,
                    eval_override["prog_config"],
                )
                metrics["pfn"] = prog_metrics
                denoised_images["pfn"] = prog_image

            metrics, denoised_images = evaluate_n2_with_ssm(
                metrics,
                denoised_images,
                n2_config_path,
                eval_override["n2_eval"],
                raw_image,
                reference,
            )

            # Save first patient's denoised images for visualization
            if patient_id == first_patient:
                visualization_images = denoised_images

            # Aggregate metrics for overall statistics
            for key, value in metrics.items():
                if key not in all_patient_metrics:
                    all_patient_metrics[key] = []
                all_patient_metrics[key].append(value)

        clear_output(wait=True)

    metrics_df = pd.DataFrame(
        [
            {
                "patient_id": patient_id,
                **{
                    f"{k}_{m}": v
                    for k, vals in all_patient_metrics.items()
                    for i, val in enumerate(vals)
                    if i == idx
                    for m, v in (
                        val.items() if isinstance(val, dict) else [("value", val)]
                    )
                },
            }
            for idx, patient_id in enumerate(dataset.keys())
        ]
    )

    metrics_df.to_csv("sdoct_evaluation_results.csv", index=False)
    print("Results saved to sdoct_evaluation_results.csv")

    print("\nAverage Metrics Across All Patients:")
    avg_metrics = {}
    for key, values_list in all_patient_metrics.items():
        if all(isinstance(v, dict) for v in values_list):
            avg_metrics[key] = {}
            metric_keys = set()
            for d in values_list:
                metric_keys.update(d.keys())

            for metric_key in metric_keys:
                # Get all values for this metric, ensuring they're numeric
                numeric_values = []
                for d in values_list:
                    if (
                        metric_key in d
                        and isinstance(d[metric_key], (int, float))
                        and not np.isnan(d[metric_key])
                    ):
                        numeric_values.append(d[metric_key])

                if numeric_values:  # Only calculate if we have numeric values
                    avg_metrics[key][metric_key] = {
                        "mean": np.mean(numeric_values),
                        "std": np.std(numeric_values),
                    }
        elif all(isinstance(v, (int, float)) for v in values_list):
            # Handle direct numeric values
            numeric_values = [v for v in values_list if not np.isnan(v)]
            if numeric_values:
                avg_metrics[key] = np.mean(numeric_values)

    # Display overall average metrics using a modified display function
    display_avg_metrics(avg_metrics)

    # Display grouped metrics
    print("\nDetailed Metrics by Model Type:")
    display_grouped_metrics(avg_metrics)

    # Plot the sample images
    plot_images(visualization_images, metrics_df)


if __name__ == "__main__":
    main()
