import torch
import torch.nn as nn
import torch.nn.functional as F


def ssim_loss(img1, img2, window_size=11, size_average=True):
    C1, C2 = 0.01**2, 0.03**2  # Small stability constants

    # Compute means
    mu1 = F.avg_pool2d(img1, window_size, stride=1, padding=window_size // 2)
    mu2 = F.avg_pool2d(img2, window_size, stride=1, padding=window_size // 2)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2

    # Compute variances and covariance
    sigma1_sq = (
        F.avg_pool2d(img1**2, window_size, stride=1, padding=window_size // 2) - mu1_sq
    )
    sigma2_sq = (
        F.avg_pool2d(img2**2, window_size, stride=1, padding=window_size // 2) - mu2_sq
    )
    sigma12 = (
        F.avg_pool2d(img1 * img2, window_size, stride=1, padding=window_size // 2)
        - mu1_mu2
    )

    # SSIM calculation
    num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = num / den
    return 1 - ssim_map.mean() if size_average else 1 - ssim_map


# Total Variation (TV) Loss
def tv_loss(img):
    loss = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :])) + torch.mean(
        torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:])
    )
    return loss


# Combined Loss Function
class Noise2VoidLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.1, gamma=0.1):
        super(Noise2VoidLoss, self).__init__()
        self.alpha = alpha  # Weight for MSE loss
        self.beta = beta  # Weight for SSIM loss
        self.gamma = gamma  # Weight for TV loss
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse = self.mse(pred, target)
        ssim = ssim_loss(pred, target)
        tv = tv_loss(pred)
        return self.alpha * mse + self.beta * ssim + self.gamma * tv


def noise_smoothness_loss(noise_component, background_mask):
    """Encourage smoothness in the noise component"""
    # Extract noise in background regions (where mask = 1)
    background_noise = noise_component * background_mask

    # Calculate local variation within background regions
    # This penalizes high-frequency components in the noise
    diff_x = background_noise[:, :, :, 1:] - background_noise[:, :, :, :-1]
    diff_y = background_noise[:, :, 1:, :] - background_noise[:, :, :-1, :]

    # Sum of squared differences (L2 norm)
    smoothness_loss = torch.mean(diff_x**2) + torch.mean(diff_y**2)

    return smoothness_loss


def structure_separation_loss(noise_component, flow_component):
    """Ensure the noise component doesn't contain structural information"""
    # We want the noise and flow components to be uncorrelated
    # Calculate correlation coefficient between noise and flow
    noise_flat = noise_component.view(noise_component.size(0), -1)
    flow_flat = flow_component.view(flow_component.size(0), -1)

    # Normalize both components
    noise_norm = (noise_flat - torch.mean(noise_flat, dim=1, keepdim=True)) / torch.std(
        noise_flat, dim=1, keepdim=True
    )
    flow_norm = (flow_flat - torch.mean(flow_flat, dim=1, keepdim=True)) / torch.std(
        flow_flat, dim=1, keepdim=True
    )

    # Calculate cosine similarity (correlation)
    correlation = torch.sum(noise_norm * flow_norm, dim=1) / (noise_norm.size(1) - 1)

    # We want to minimize the absolute correlation
    return torch.mean(torch.abs(correlation))


def noise_distribution_regularization(noise_component, background_mask):
    """Encourage noise to follow expected statistical distribution"""
    # Extract background noise values
    bg_noise_values = noise_component[background_mask > 0.5]

    if bg_noise_values.numel() == 0:
        return torch.tensor(0.0, device=noise_component.device)

    # For speckle noise, often Rayleigh distribution is appropriate
    # Here we'll use a simple approach to match first and second moments

    # Calculate current statistics
    mean_noise = torch.mean(bg_noise_values)
    std_noise = torch.std(bg_noise_values)

    # Target statistics for noise (can be determined empirically)
    # For Rayleigh: mean ≈ σ√(π/2), std ≈ σ√(2-π/2)
    target_mean = 0.2  # Example value
    target_std = 0.15  # Example value

    # Penalize deviation from target statistics
    return torch.abs(mean_noise - target_mean) + torch.abs(std_noise - target_std)


def local_coherence_loss(noise_component, patch_size=7):
    """Encourage noise to have local coherence"""
    # Calculate local patch statistics
    unfold = nn.Unfold(kernel_size=patch_size, stride=1, padding=patch_size // 2)
    patches = unfold(noise_component)
    patches = patches.view(
        noise_component.size(0), -1, noise_component.size(2), noise_component.size(3)
    )

    # Calculate variance within each patch
    patch_mean = torch.mean(patches, dim=1, keepdim=True)
    patch_var = torch.mean((patches - patch_mean) ** 2, dim=1)

    # We want low variance within patches (locally smooth)
    return torch.mean(patch_var)


def structural_correlation_loss(noise_component, target):
    """
    Penalize correlation between the noise component and structural information in target
    """
    # We want the noise component to be uncorrelated with the target structures
    # Flatten the tensors for correlation calculation
    noise_flat = noise_component.view(noise_component.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Normalize to zero mean and unit variance for proper correlation measurement
    noise_norm = (noise_flat - torch.mean(noise_flat, dim=1, keepdim=True)) / (
        torch.std(noise_flat, dim=1, keepdim=True) + 1e-8
    )
    target_norm = (target_flat - torch.mean(target_flat, dim=1, keepdim=True)) / (
        torch.std(target_flat, dim=1, keepdim=True) + 1e-8
    )

    # Calculate correlation coefficient
    correlation = torch.sum(noise_norm * target_norm, dim=1) / (noise_norm.size(1) - 1)

    # Minimize the absolute correlation
    return torch.mean(torch.abs(correlation))


def flow_structure_preservation_loss(
    flow_component, original_image, target=None, threshold=0.15
):
    """
    Preserves flow-related speckle patterns while allowing removal of noise-only speckle.

    Args:
        flow_component: Model's predicted flow component
        original_image: Original OCT input image
        target: Optional target OCTA image (if available)
        threshold: Intensity threshold for identifying potential flow regions
    """
    # 1. Identify potential flow regions using temporal variance or intensity patterns
    if target is not None:
        # If we have target OCTA, use it to identify flow regions
        flow_regions = (target > threshold).float()
    else:
        # Otherwise use heuristics to estimate flow regions
        # Technique: Look for structured speckle with temporal correlation
        variance = torch.var(original_image, dim=0, keepdim=True)
        flow_regions = (variance > variance.mean() * 1.5).float()

    # 2. Calculate flow pattern coherence loss
    # We want to preserve spatial coherence in flow regions
    sobel_x = (
        torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(flow_component.device)
    )
    sobel_y = (
        torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        .view(1, 1, 3, 3)
        .to(flow_component.device)
    )

    # Calculate gradients of original and flow component
    orig_grad_x = F.conv2d(original_image, sobel_x, padding=1)
    orig_grad_y = F.conv2d(original_image, sobel_y, padding=1)
    flow_grad_x = F.conv2d(flow_component, sobel_x, padding=1)
    flow_grad_y = F.conv2d(flow_component, sobel_y, padding=1)

    # Calculate gradient magnitude
    orig_grad_mag = torch.sqrt(orig_grad_x**2 + orig_grad_y**2 + 1e-6)
    flow_grad_mag = torch.sqrt(flow_grad_x**2 + flow_grad_y**2 + 1e-6)

    # In flow regions, preserve gradient structure
    flow_structure_loss = F.l1_loss(
        flow_grad_mag * flow_regions, orig_grad_mag * flow_regions, reduction="sum"
    ) / (flow_regions.sum() + 1e-6)

    # 3. Add spectral constraint for flow patterns
    # Optional: Add FFT-based constraint to preserve spectral characteristics of flow

    return flow_structure_loss


def custom_loss(
    flow_component, noise_component, batch_inputs, batch_targets, loss_parameters, debug
):
    mse = nn.MSELoss(reduction="none")

    foreground_mask = (batch_targets > 0.03).float()
    background_mask = 1.0 - foreground_mask

    """
    plt.figure(figsize=(10, 5))
    plt.imshow(foreground_mask[0, 0].cpu().numpy(), cmap='gray')
    plt.title("Foreground Mask")
    plt.show()
    """

    # not very elegant just mse loss between masked flow and target
    pixel_wise_loss = mse(flow_component, batch_targets)
    foreground_loss = (pixel_wise_loss * foreground_mask).sum() / (
        foreground_mask.sum() + 1e-8
    )
    background_loss = torch.mean((flow_component * background_mask) ** 2) * 2.0

    # possible non edge points dominate the edge points, maybe need to mask,, the background might be dominating

    horizontal_kernel = torch.tensor(
        [[[[1, -1]]]], dtype=batch_targets.dtype, device=batch_targets.device
    )
    vertical_kernel = torch.tensor(
        [[[[1], [-1]]]], dtype=batch_targets.dtype, device=batch_targets.device
    )
    diagonal1_kernel = torch.tensor(
        [[[[1, 0], [0, -1]]]], dtype=batch_targets.dtype, device=batch_targets.device
    )
    diagonal2_kernel = torch.tensor(
        [[[[0, 1], [-1, 0]]]], dtype=batch_targets.dtype, device=batch_targets.device
    )

    h_edges_target = torch.abs(
        torch.nn.functional.conv2d(batch_targets, horizontal_kernel, padding=(0, 1))
    )
    h_edges_flow = torch.abs(
        torch.nn.functional.conv2d(flow_component, horizontal_kernel, padding=(0, 1))
    )
    h_edge_loss = mse(h_edges_flow, h_edges_target).mean()

    v_edges_target = torch.abs(
        torch.nn.functional.conv2d(batch_targets, vertical_kernel, padding=(1, 0))
    )
    v_edges_flow = torch.abs(
        torch.nn.functional.conv2d(flow_component, vertical_kernel, padding=(1, 0))
    )
    v_edge_loss = mse(v_edges_flow, v_edges_target).mean()

    d1_edges_target = torch.abs(
        torch.nn.functional.conv2d(batch_targets, diagonal1_kernel, padding=(1, 1))
    )
    d1_edges_flow = torch.abs(
        torch.nn.functional.conv2d(flow_component, diagonal1_kernel, padding=(1, 1))
    )
    d1_edge_loss = mse(d1_edges_flow, d1_edges_target).mean()

    d2_edges_target = torch.abs(
        torch.nn.functional.conv2d(batch_targets, diagonal2_kernel, padding=(1, 1))
    )
    d2_edges_flow = torch.abs(
        torch.nn.functional.conv2d(flow_component, diagonal2_kernel, padding=(1, 1))
    )
    d2_edge_loss = mse(d2_edges_flow, d2_edges_target).mean()

    edge_loss = (
        1.5 * h_edge_loss + 1.5 * v_edge_loss + 0.8 * d1_edge_loss + 0.8 * d2_edge_loss
    ) / 4.0

    continuity_kernel = torch.ones((1, 1, 3, 3), device=batch_targets.device) / 9.0
    local_avg = torch.nn.functional.conv2d(flow_component, continuity_kernel, padding=1)
    discontinuity_penalty = (
        torch.mean(((flow_component - local_avg) * foreground_mask) ** 2) * 2.0
    )

    # flow_preservation_loss = flow_structure_preservation_loss(flow_component, batch_inputs, batch_targets)

    if not loss_parameters:
        alpha, beta, gamma, delta = 1.0, 1.0, 1.0, 1.0
    else:
        alpha = loss_parameters.get("alpha", 1.0)
        beta = loss_parameters.get("beta", 1.0)
        gamma = loss_parameters.get("gamma", 1.0)
        delta = loss_parameters.get("delta", 1.0)

    total_loss = (
        alpha * foreground_loss  # Focus on vessel regions
        + beta * background_loss  # Strong background suppression
        + gamma
        * edge_loss  # Edge preservation     # NEW: Explicitly penalize unwanted circular patterns
        + delta * discontinuity_penalty  # NEW: Penalize discontinuities in vessels
        # + 1.0 * flow_preservation_loss
    )

    print(
        f"Foreground Loss: {foreground_loss.item()}, Background Loss: {background_loss.item()}, Edge Loss: {edge_loss.item()}, Discontinuity Penalty: {discontinuity_penalty.item()}"
    )

    return total_loss
