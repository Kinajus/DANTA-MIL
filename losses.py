"""Loss functions for DANTA-MIL."""

from __future__ import annotations

import torch
from torch import Tensor


def nll_survival_loss(
    hazards: Tensor,
    survival: Tensor | None,
    targets: Tensor,
    censorship: Tensor,
    alpha: float = 0.0,
    eps: float = 1e-7,
) -> Tensor:
    """Discrete-time negative log-likelihood survival loss.

    ``censorship`` follows the common computational pathology convention:
    ``0`` means event observed and ``1`` means censored.
    """

    batch_size = targets.shape[0]
    targets = targets.view(batch_size, 1).long()
    censorship = censorship.view(batch_size, 1).float()
    if survival is None:
        survival = torch.cumprod(1.0 - hazards, dim=1)

    survival_padded = torch.cat([torch.ones_like(censorship), survival], dim=1)
    uncensored = -(
        1.0 - censorship
    ) * (
        torch.log(torch.gather(survival_padded, 1, targets).clamp(min=eps))
        + torch.log(torch.gather(hazards, 1, targets).clamp(min=eps))
    )
    censored = -censorship * torch.log(
        torch.gather(survival_padded, 1, targets + 1).clamp(min=eps)
    )
    loss = censored + uncensored
    loss = (1.0 - alpha) * loss + alpha * uncensored
    return loss.mean()


class NLLSurvivalLoss:
    """Callable wrapper for ``nll_survival_loss``."""

    def __init__(self, alpha: float = 0.0) -> None:
        self.alpha = alpha

    def __call__(
        self,
        hazards: Tensor,
        survival: Tensor | None,
        targets: Tensor,
        censorship: Tensor,
    ) -> Tensor:
        return nll_survival_loss(
            hazards=hazards,
            survival=survival,
            targets=targets,
            censorship=censorship,
            alpha=self.alpha,
        )
