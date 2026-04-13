"""Loss helpers for JanusFlow-Art."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_flow_matching_loss(
    predicted_velocity: torch.Tensor,
    target_velocity: torch.Tensor,
    *,
    loss_type: str = "mse",
) -> torch.Tensor:
    """Compute the main JanusFlow-Art flow loss."""

    if loss_type == "l1":
        return F.l1_loss(predicted_velocity, target_velocity)
    return F.mse_loss(predicted_velocity, target_velocity)


def compute_art_alignment_loss(
    predicted_embedding: torch.Tensor,
    target_embedding: torch.Tensor,
) -> torch.Tensor:
    """Align generated intermediate features to style embeddings."""

    predicted = F.normalize(predicted_embedding, dim=-1)
    target = F.normalize(target_embedding, dim=-1)
    return 1.0 - (predicted * target).sum(dim=-1).mean()


def compute_style_classification_loss(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    ignore_index: int = 0,
) -> torch.Tensor:
    """Compute a label-supervised style classification regularizer."""

    return F.cross_entropy(logits, target_ids, ignore_index=ignore_index)


def compute_texture_aux_loss(
    predicted_texture_embedding: torch.Tensor,
    target_texture_embedding: torch.Tensor,
) -> torch.Tensor:
    """Encourage decoder-side texture features to match texture-style features."""

    return F.mse_loss(
        F.normalize(predicted_texture_embedding, dim=-1),
        F.normalize(target_texture_embedding, dim=-1),
    )


def _depthwise_filter2d(tensor: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    channels = tensor.shape[1]
    weight = kernel.to(device=tensor.device, dtype=tensor.dtype).view(1, 1, 3, 3)
    weight = weight.repeat(channels, 1, 1, 1)
    return F.conv2d(tensor, weight, padding=1, groups=channels)


def _gaussian_kernel2d(
    kernel_size: int,
    sigma: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    kernel_1d = torch.exp(-(coords * coords) / max(2.0 * sigma * sigma, 1.0e-6))
    kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1.0e-6)
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d / kernel_2d.sum().clamp_min(1.0e-6)


def gaussian_blur2d(
    tensor: torch.Tensor,
    *,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Apply a depthwise Gaussian blur to a BCHW tensor."""

    kernel = _gaussian_kernel2d(
        kernel_size,
        sigma,
        device=tensor.device,
        dtype=tensor.dtype,
    ).view(1, 1, kernel_size, kernel_size)
    weight = kernel.repeat(tensor.shape[1], 1, 1, 1)
    padding = kernel_size // 2
    return F.conv2d(tensor, weight, padding=padding, groups=tensor.shape[1])


def compute_laplacian_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match local second-order edge/detail structure in latent space."""

    kernel = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        device=predicted.device,
        dtype=predicted.dtype,
    )
    return F.l1_loss(
        _depthwise_filter2d(predicted, kernel),
        _depthwise_filter2d(target, kernel),
    )


def compute_sobel_gradient_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match first-order local edge energy in latent space."""

    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=predicted.device,
        dtype=predicted.dtype,
    )
    sobel_y = sobel_x.t()
    pred_x = _depthwise_filter2d(predicted, sobel_x)
    pred_y = _depthwise_filter2d(predicted, sobel_y)
    target_x = _depthwise_filter2d(target, sobel_x)
    target_y = _depthwise_filter2d(target, sobel_y)
    return 0.5 * (F.l1_loss(pred_x, target_x) + F.l1_loss(pred_y, target_y))


def compute_fft_high_frequency_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    min_radius: float = 0.35,
) -> torch.Tensor:
    """Match high-frequency spectral magnitude in latent space."""

    pred_fft = torch.fft.rfft2(predicted.float(), norm="ortho").abs()
    target_fft = torch.fft.rfft2(target.float(), norm="ortho").abs()
    height, width = predicted.shape[-2:]
    fft_width = width // 2 + 1
    y = torch.linspace(0.0, 1.0, height, device=predicted.device)[:, None]
    x = torch.linspace(0.0, 1.0, fft_width, device=predicted.device)[None, :]
    mask = (torch.sqrt(x * x + y * y) >= min_radius).to(dtype=pred_fft.dtype)
    diff = (pred_fft - target_fft).abs() * mask.view(1, 1, height, fft_width)
    return diff.sum() / mask.sum().clamp_min(1.0) / max(predicted.shape[0] * predicted.shape[1], 1)


def compute_low_frequency_consistency_loss(
    predicted: torch.Tensor,
    reference: torch.Tensor,
    *,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """Match only low-frequency structure between tuned and base predictions."""

    predicted_blur = gaussian_blur2d(predicted, kernel_size=kernel_size, sigma=sigma)
    reference_blur = gaussian_blur2d(reference, kernel_size=kernel_size, sigma=sigma)
    return F.mse_loss(predicted_blur, reference_blur)


def compute_texture_statistics_orientation_loss(
    orientation_logits: torch.Tensor,
    orientation_target: torch.Tensor,
) -> torch.Tensor:
    """Supervise per-location dominant orientation bins."""

    return F.cross_entropy(orientation_logits, orientation_target)


def compute_texture_statistics_band_loss(
    band_logits: torch.Tensor,
    band_target: torch.Tensor,
) -> torch.Tensor:
    """Match multi-band texture energy targets."""

    band_probs = torch.softmax(band_logits, dim=1)
    normalized_target = band_target / band_target.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    return F.l1_loss(band_probs, normalized_target)


def compute_texture_statistics_density_loss(
    density_map: torch.Tensor,
    density_target: torch.Tensor,
) -> torch.Tensor:
    """Match local texture-density targets."""

    return F.l1_loss(torch.sigmoid(density_map), density_target)


def compute_texture_statistics_pigment_loss(
    pigment_map: torch.Tensor,
    pigment_target: torch.Tensor,
) -> torch.Tensor:
    """Match local pigment-thickness proxy targets."""

    return F.l1_loss(torch.sigmoid(pigment_map), pigment_target)


def compute_basis_usage_entropy_loss(basis_usage_entropy: torch.Tensor) -> torch.Tensor:
    """Regularize basis usage toward lower entropy and sparser local routing."""

    return basis_usage_entropy


def compute_stroke_field_orientation_loss(
    theta: torch.Tensor,
    angle_target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align predicted stroke orientation with axial proxy orientation."""

    predicted_angle = torch.tanh(theta) * torch.pi
    delta = predicted_angle - angle_target
    loss_map = 1.0 - torch.cos(2.0 * delta)
    if weight is None:
        return loss_map.mean()
    weighted = loss_map * weight
    return weighted.sum() / weight.sum().clamp_min(1.0e-6)


def compute_stroke_field_length_loss(
    length: torch.Tensor,
    length_target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match predicted stroke-length proxy to coherent-edge target."""

    predicted_length = torch.sigmoid(length) * 0.85 + 0.25
    loss_map = (predicted_length - length_target).abs()
    if weight is None:
        return loss_map.mean()
    weighted = loss_map * weight
    return weighted.sum() / weight.sum().clamp_min(1.0e-6)


def compute_stroke_field_width_loss(
    width: torch.Tensor,
    width_target: torch.Tensor,
    *,
    weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Match predicted stroke-width proxy to local coherence target."""

    predicted_width = torch.sigmoid(width) * 0.22 + 0.08
    loss_map = (predicted_width - width_target).abs()
    if weight is None:
        return loss_map.mean()
    weighted = loss_map * weight
    return weighted.sum() / weight.sum().clamp_min(1.0e-6)


def compute_stroke_field_alpha_loss(
    alpha: torch.Tensor,
    alpha_target: torch.Tensor,
) -> torch.Tensor:
    """Match predicted stroke occupancy/intensity to local texture proxy."""

    return F.l1_loss(torch.sigmoid(alpha), alpha_target)


def compute_stroke_field_objectness_loss(
    objectness: torch.Tensor,
    objectness_target: torch.Tensor,
) -> torch.Tensor:
    """Match explicit stroke occupancy/objectness to a sharper proxy target."""

    return F.l1_loss(torch.sigmoid(objectness), objectness_target)


def compute_stroke_field_blank_suppression_loss(
    objectness: torch.Tensor,
    blank_region_target: torch.Tensor,
) -> torch.Tensor:
    """Penalize stroke occupancy inside low-support blank regions."""

    return (torch.sigmoid(objectness) * blank_region_target).mean()


def compute_stroke_field_support_ceiling_loss(
    objectness: torch.Tensor,
    support_target: torch.Tensor,
) -> torch.Tensor:
    """Penalize objectness that exceeds the local support envelope."""

    return torch.relu(torch.sigmoid(objectness) - support_target).mean()


def compute_stroke_field_prototype_loss(
    prototype_logits: torch.Tensor,
    prototype_target: torch.Tensor,
) -> torch.Tensor:
    """Supervise per-location prototype selection against nearest proxy anchor."""

    return F.cross_entropy(prototype_logits, prototype_target)


def compute_stroke_occupancy_bce_loss(
    occupancy_logits: torch.Tensor,
    occupancy_target: torch.Tensor,
) -> torch.Tensor:
    """Supervise sparse-anchor occupancy logits against proxy occupancy targets."""

    return F.binary_cross_entropy_with_logits(occupancy_logits, occupancy_target)


def compute_stroke_anchor_outside_support_loss(
    anchor_map: torch.Tensor,
    support_target: torch.Tensor,
) -> torch.Tensor:
    """Penalize selected anchors that fall outside the allowed support envelope."""

    outside = anchor_map * (1.0 - support_target)
    return outside.sum() / anchor_map.sum().clamp_min(1.0e-6)


def compute_stroke_anchor_sparsity_loss(anchor_map: torch.Tensor) -> torch.Tensor:
    """Keep sparse-anchor activation compact instead of filling the whole grid."""

    return anchor_map.mean()


def compute_stroke_anchor_small_component_loss(
    anchor_map: torch.Tensor,
    small_component_mask: torch.Tensor,
) -> torch.Tensor:
    """Penalize anchors that land on support fragments below the minimum area."""

    invalid = anchor_map * small_component_mask
    return invalid.sum() / anchor_map.sum().clamp_min(1.0e-6)


def compute_stroke_anchor_component_overflow_loss(component_overflow_score: torch.Tensor) -> torch.Tensor:
    """Penalize excess component-local anchor demand beyond allowed capacity."""

    return component_overflow_score.mean()


def compute_stroke_anchor_inside_exclusion_loss(
    anchor_map: torch.Tensor,
    exclusion_map: torch.Tensor,
) -> torch.Tensor:
    """Penalize selected anchors that still fall inside exclusion regions."""

    invalid = anchor_map * exclusion_map
    return invalid.sum() / anchor_map.sum().clamp_min(1.0e-6)


def compute_stroke_exclusion_overlap_loss(
    occupancy_logits: torch.Tensor,
    exclusion_map: torch.Tensor,
) -> torch.Tensor:
    """Penalize dense occupancy mass inside portrait/head exclusion regions."""

    return (torch.sigmoid(occupancy_logits) * exclusion_map).mean()


def compute_slot_anchor_outside_support_loss(
    slot_valid_probs: torch.Tensor,
    slot_support_mass: torch.Tensor,
) -> torch.Tensor:
    """Penalize valid slots that do not gather enough support evidence."""

    invalid_mass = slot_valid_probs * (1.0 - slot_support_mass.clamp(0.0, 1.0))
    return invalid_mass.mean()


def compute_slot_anchor_inside_exclusion_loss(
    slot_valid_probs: torch.Tensor,
    slot_exclusion_mass: torch.Tensor,
) -> torch.Tensor:
    """Penalize valid slots whose attention mass falls into exclusion regions."""

    invalid_mass = slot_valid_probs * slot_exclusion_mass.clamp(0.0, 1.0)
    return invalid_mass.mean()


def compute_slot_component_conflict_loss(slot_component_conflict_score: torch.Tensor) -> torch.Tensor:
    """Penalize excess valid-slot demand beyond component-local capacity."""

    return slot_component_conflict_score.mean()


def compute_slot_count_sparsity_loss(slot_valid_probs: torch.Tensor) -> torch.Tensor:
    """Keep the explicit slot set sparse instead of activating every slot."""

    return slot_valid_probs.mean()
