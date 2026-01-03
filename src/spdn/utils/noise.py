import torch


def create_blind_spot_input_with_realistic_noise(image, mask):
    blind_input = image.clone()

    # Parameters for OCT-like noise (speckle)
    mean_level = image.mean()
    std_level = image.std() * 0.9  # Adjust based on your OCT noise level

    # Generate noise with speckle-like characteristics
    noise = torch.randn_like(image) * std_level + mean_level

    # Apply mask
    blind_input[mask.bool()] = noise[mask.bool()]
    return blind_input
