import torch
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
from skimage import io
from skimage.transform import resize

from spdn.utils import normalize_image, normalize_image_np
from spdn.utils.eval_utils.metrics import evaluate_oct_denoising


def get_sample_image(dataloader, device):
    sample = next(iter(dataloader))
    image = sample[0].to(device)
    return image


def plot_sample(image, denoised, model, method):
    fig, ax = plt.subplots(1, 3, figsize=(12, 6))

    fig.suptitle(f"Model: {model.__class__.__name__} - {method}", fontsize=16)

    ax[0].imshow(image, cmap="gray")
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    normalised_denoised = normalize_image(denoised)

    ax[1].imshow(denoised, cmap="gray")
    ax[1].set_title("Denoised Image")
    ax[1].axis("off")

    ax[2].imshow(normalised_denoised, cmap="gray")
    ax[2].set_title("Normalized Denoised Image")
    ax[2].axis("off")

    plt.show()


def denoise_image(model, image, device):
    model.eval()
    with torch.no_grad():
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
            if len(image.shape) == 2:
                image = image.unsqueeze(0).unsqueeze(0)
        image = image.to(device)
        denoised_image = model(image)
    return denoised_image


load_dotenv()

device = os.getenv("DEVICE")
device


def evaluate(image, reference, model, method):
    # Convert image to tensor if it's a numpy array
    if isinstance(image, np.ndarray):
        image_tensor = torch.from_numpy(image).float()
        if len(image_tensor.shape) == 2:
            image_tensor = image_tensor.unsqueeze(0).unsqueeze(0)
        original_image = image
    else:
        image_tensor = image
        try:
            original_image = image.cpu().numpy()[0][0]
        except:
            original_image = image.cpu().numpy()

    denoised = denoise_image(model, image_tensor, device="cuda")

    # denoised = normalize_image(denoised)

    # denoised = denoised[-1]

    print(f"Image shape: {image_tensor.shape}")
    print(f"Denoised shape: {denoised.shape}")

    # Convert denoised to numpy
    if isinstance(denoised, torch.Tensor):
        denoised_np = denoised.cpu().numpy()
    else:
        denoised_np = denoised

    # Get the first image if it's batched
    if len(denoised_np.shape) > 2:
        if len(denoised_np.shape) == 4:  # [batch, channel, height, width]
            denoised_np = denoised_np[0, 0]
        elif (
            len(denoised_np.shape) == 3
        ):  # [channel, height, width] or [batch, height, width]
            denoised_np = denoised_np[0]

    denoised_np = normalize_image_np(denoised_np)

    # Handle reference (could be tensor or numpy)
    if isinstance(reference, torch.Tensor):
        reference_np = reference.cpu().numpy()
        if len(reference_np.shape) > 2:
            if len(reference_np.shape) == 4:  # [batch, channel, height, width]
                reference_np = reference_np[0, 0]
            elif (
                len(reference_np.shape) == 3
            ):  # [channel, height, width] or [batch, height, width]
                reference_np = reference_np[0]
    else:
        reference_np = reference
        if len(reference_np.shape) > 2:
            if len(reference_np.shape) == 4:  # [batch, channel, height, width]
                reference_np = reference_np[0, 0]
            elif (
                len(reference_np.shape) == 3
            ):  # [channel, height, width] or [batch, height, width]
                reference_np = reference_np[0]

    # Print shapes for debugging
    original_image = original_image.squeeze().squeeze()
    print(f"Original shape: {original_image.shape}")
    print(f"Denoised shape: {denoised_np.shape}")
    print(f"Reference shape: {reference_np.shape}")

    metrics = evaluate_oct_denoising(original_image, denoised_np, reference_np)

    return metrics, denoised_np


from spdn.utils.data_utils.standard_preprocessing import normalize_image


def load_sdoct_dataset(dataset_path, target_size=(256, 256)):
    sdoct_data = {}
    patients = os.listdir(dataset_path)

    print(f"Loading SDOCT dataset from {dataset_path}")
    for patient in tqdm(patients, desc="Loading patients"):
        patient_path = os.path.join(dataset_path, patient)
        avg_path = os.path.join(patient_path, f"{patient}_Averaged Image.tif")
        raw_path = os.path.join(patient_path, f"{patient}_Raw Image.tif")

        try:
            if not os.path.exists(avg_path) or not os.path.exists(raw_path):
                print(f"Missing files for patient {patient}")
                continue

            raw_img = io.imread(raw_path)
            avg_img = io.imread(avg_path)

            # Resize images
            raw_img = resize(raw_img, target_size, anti_aliasing=True)
            avg_img = resize(avg_img, target_size, anti_aliasing=True)

            raw_tensor = torch.from_numpy(raw_img).float().unsqueeze(0).unsqueeze(0)
            avg_tensor = torch.from_numpy(avg_img).float().unsqueeze(0).unsqueeze(0)

            sdoct_data[patient] = {
                "raw": raw_tensor,
                "avg": avg_tensor,
                "raw_np": raw_img,
                "avg_np": avg_img,
            }

        except Exception as e:
            print(f"Error processing patient {patient}: {e}")

    print(f"Successfully loaded {len(sdoct_data)} SDOCT patients")
    return sdoct_data


def load_duke_dataset(dataset_path, target_size=(256, 256)):
    sdoct_data = {}
    patients = os.listdir(dataset_path)

    print(f"Loading SDOCT dataset from {dataset_path}")
    for patient in tqdm(patients, desc="Loading patients"):
        patient_path = os.path.join(dataset_path, patient)
        avg_path = os.path.join(patient_path, f"{patient}_Averaged Image.tif")
        raw_path = os.path.join(patient_path, f"{patient}_Raw Image.tif")

        try:
            if not os.path.exists(avg_path) or not os.path.exists(raw_path):
                print(f"Missing files for patient {patient}")
                continue

            raw_img = io.imread(raw_path)
            avg_img = io.imread(avg_path)

            # Resize images
            raw_img = resize(raw_img, target_size, anti_aliasing=True)
            avg_img = resize(avg_img, target_size, anti_aliasing=True)

            raw_tensor = torch.from_numpy(raw_img).float().unsqueeze(0).unsqueeze(0)
            avg_tensor = torch.from_numpy(avg_img).float().unsqueeze(0).unsqueeze(0)

            sdoct_data[patient] = {
                "raw": raw_tensor,
                "avg": avg_tensor,
                "raw_np": raw_img,
                "avg_np": avg_img,
            }

        except Exception as e:
            print(f"Error processing patient {patient}: {e}")

    print(f"Successfully loaded {len(sdoct_data)} SDOCT patients")
    return sdoct_data
