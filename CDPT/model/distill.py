"""
Knowledge distillation losses for CDPT (encoder feature align, denoise MI / KL).
"""

import torch
import torch.nn.functional as F


def encoder_feature_distillation_loss(
    student_attention_values: list,
    teacher_attention_values: list,
) -> torch.Tensor:
    """MSE between last-layer encoder attention features."""
    return F.mse_loss(
        student_attention_values[-1], teacher_attention_values[-1], reduction="mean"
    )


def mi_distill_loss(
    student_output: torch.Tensor,
    teacher_output: torch.Tensor,
    scoring_fn: str = "cosine",
) -> torch.Tensor:
    """
    Mutual-information style distillation via InfoNCE on flattened batch features.

    Args:
        student_output: [B, A, T, D]
        teacher_output: [B, A, T, D]
        scoring_fn: 'mse' or 'cosine'
    """
    B = student_output.shape[0]
    student_flat = student_output.view(B, -1)
    teacher_flat = teacher_output.view(B, -1)

    if scoring_fn == "mse":
        mse_matrix = -torch.cdist(student_flat, teacher_flat, p=2) ** 2 / student_flat.shape[1]
        scores = mse_matrix
    elif scoring_fn == "cosine":
        student_norm = student_flat / torch.norm(student_flat, dim=1, keepdim=True)
        teacher_norm = teacher_flat / torch.norm(teacher_flat, dim=1, keepdim=True)
        scores = torch.mm(student_norm, teacher_norm.T)
    else:
        raise ValueError(f"Invalid scoring function: {scoring_fn}. Choose 'mse' or 'cosine'")

    mi_losses = -torch.log_softmax(scores, dim=1)[range(B), range(B)]
    return mi_losses.mean()


def denoise_kl_distill_loss(
    student_denoised_normalized: torch.Tensor,
    teacher_denoised_normalized: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """KL(student || teacher) over softmax of flattened normalized denoised actions."""
    student_softmax = F.softmax(student_denoised_normalized.flatten(2), dim=-1)
    teacher_softmax = F.softmax(teacher_denoised_normalized.flatten(2), dim=-1)
    return F.kl_div(
        torch.log(student_softmax + eps),
        teacher_softmax,
        reduction="batchmean",
    )
