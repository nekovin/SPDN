import torch


def lognormal_consistency_loss(
    denoised, noisy, epsilon=1e-6
):  # based on noise distribution
    denoised_safe = torch.clamp(denoised, min=epsilon)
    noisy_safe = torch.clamp(noisy, min=epsilon)

    ratio = noisy_safe / denoised_safe

    ratio_safe = torch.clamp(ratio, min=epsilon, max=10.0)

    log_ratio = torch.log(ratio_safe)

    mu = torch.mean(log_ratio)
    sigma = torch.std(log_ratio)

    if torch.isnan(mu) or torch.isnan(sigma):
        return torch.tensor(0.0, device=denoised.device, requires_grad=True)

    expected_mu = 0.0
    expected_sigma = 0.5

    loss = torch.abs(mu - expected_mu) + torch.abs(sigma - expected_sigma)
    return loss


def lognormal_consistency_loss(denoised, noisy, epsilon=1e-6):
    """
    Physics-informed loss term that ensures denoised and noisy images
    maintain the expected log-normal relationship for OCT speckle.
    """
    denoised_safe = torch.clamp(denoised, min=epsilon)
    noisy_safe = torch.clamp(noisy, min=epsilon)

    ratio = noisy_safe / denoised_safe

    # Clamp ratio to reasonable range to prevent extreme values
    ratio_safe = torch.clamp(ratio, min=epsilon, max=10.0)

    # Log-transform the ratio which should follow normal distribution
    log_ratio = torch.log(ratio_safe)

    # For log-normal statistics, calculate parameters
    mu = torch.mean(log_ratio)
    sigma = torch.std(log_ratio)

    # Check for NaN values
    if torch.isnan(mu) or torch.isnan(sigma):
        return torch.tensor(0.0, device=denoised.device, requires_grad=True)

    # Expected values for log-normal OCT speckle
    expected_mu = 0.0  # Calibrate this
    expected_sigma = 0.5  # Calibrate this

    # Penalize deviation from expected log-normal statistics
    loss = torch.abs(mu - expected_mu) + torch.abs(sigma - expected_sigma)
    return loss
