"""DANTA-MIL model components aligned with the current manuscript.

The trainable pipeline is:

RSRP -> DNCI -> bag-level attention pooling -> task-specific prediction head.

The frozen patch encoder is executed offline and is intentionally not part of
this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass
class DANTAConfig:
    """Configuration for :class:`DANTAMIL`.

    The manuscript configuration uses ``latent_dim=512``,
    ``requested_neighbors=6``, ``dropout=0.25``, two logits for binary
    classification, and four intervals for overall-survival prediction.
    """

    input_dim: int = 1024
    latent_dim: int = 512
    n_classes: int = 2
    num_intervals: int = 4
    requested_neighbors: int = 6
    dropout: float = 0.25
    task: str = "classification"

    def __post_init__(self) -> None:
        if self.task not in {"classification", "survival"}:
            raise ValueError("task must be 'classification' or 'survival'")
        if self.input_dim < 1 or self.latent_dim < 1:
            raise ValueError("input_dim and latent_dim must be positive")
        if self.n_classes < 2:
            raise ValueError("classification requires at least two logits")
        if self.num_intervals != 4:
            raise ValueError("the manuscript survival model uses exactly four intervals")
        if self.requested_neighbors < 1:
            raise ValueError("requested_neighbors must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class DANTAOutput:
    """Model output and optional instance-level analysis tensors.

    All tensors retain an explicit batch dimension. A two-dimensional input
    bag is interpreted as a batch of size one.
    """

    logits: Tensor
    probabilities: Tensor | None = None
    hazards: Tensor | None = None
    survival: Tensor | None = None
    risk: Tensor | None = None
    attention_weights: Tensor | None = None
    query_features: Tensor | None = None
    key_features: Tensor | None = None
    neighbor_indices: Tensor | None = None
    neighbor_mask: Tensor | None = None
    first_stage_weights: Tensor | None = None
    second_stage_weights: Tensor | None = None
    instance_mask: Tensor | None = None


@dataclass
class LocalizationOutput:
    """Decision-aligned binary-classification localization decomposition."""

    scores: Tensor
    contributions: Tensor
    attribution_mass: Tensor
    reconstructed_logit_contrast: Tensor
    logit_contrast: Tensor
    reconstruction_error: Tensor


def initialize_weights(module: nn.Module) -> None:
    """Apply Xavier-normal initialization once to each trainable linear layer."""

    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RoleSpecificReembeddingProjector(nn.Module):
    """Role-Specific Re-embedding Projector (RSRP).

    A shared ``Linear -> ReLU -> Dropout`` projection is followed by two
    independent affine maps that create query and key representations.
    """

    def __init__(self, input_dim: int = 1024, latent_dim: int = 512, dropout: float = 0.25) -> None:
        super().__init__()
        self.shared_projection = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.query_projection = nn.Linear(latent_dim, latent_dim)
        self.key_projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, features: Tensor) -> tuple[Tensor, Tensor]:
        shared = self.shared_projection(features)
        return self.query_projection(shared), self.key_projection(shared)


class DirectedNeighborhoodConstructorIntegrator(nn.Module):
    """Directed Neighborhood Constructor--Integrator (DNCI).

    DNCI contains no trainable mappings. It forms the dense scaled query--key
    matrix, masks self-interactions, selects exact row-wise non-self Top-K
    neighborhoods, applies the manuscript's two weighting stages, and adds the
    aggregated key representation to the source query.
    """

    def __init__(self, requested_neighbors: int = 6) -> None:
        super().__init__()
        if requested_neighbors < 1:
            raise ValueError("requested_neighbors must be >= 1")
        self.requested_neighbors = requested_neighbors

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        mask: Tensor | None = None,
        return_details: bool = False,
    ) -> tuple[Tensor, Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        squeeze_batch = False
        if queries.ndim == 2:
            queries = queries.unsqueeze(0)
            keys = keys.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
            squeeze_batch = True
        if queries.ndim != 3 or keys.shape != queries.shape:
            raise ValueError(
                "queries and keys must have identical [B, I, D] or [I, D] shapes"
            )

        batch_size, n_instances, latent_dim = queries.shape
        if mask is None:
            mask = torch.ones(
                batch_size,
                n_instances,
                dtype=torch.bool,
                device=queries.device,
            )
        else:
            mask = mask.to(device=queries.device, dtype=torch.bool)
            if mask.shape != (batch_size, n_instances):
                raise ValueError(
                    f"Expected mask shape {(batch_size, n_instances)}, got {tuple(mask.shape)}"
                )

        valid_counts = mask.sum(dim=1)
        if torch.any(valid_counts < 2):
            raise ValueError(
                "DNCI requires at least two valid instances per bag; the manuscript "
                "does not define an empty non-self neighborhood for I=1"
            )

        max_neighbors = min(
            self.requested_neighbors,
            int((valid_counts - 1).max().item()),
        )
        contextualized_batches: list[Tensor] = []
        index_batches: list[Tensor] = []
        neighbor_mask_batches: list[Tensor] = []
        alpha_batches: list[Tensor] = []
        beta_batches: list[Tensor] = []

        for batch_index in range(batch_size):
            valid_positions = torch.nonzero(mask[batch_index], as_tuple=False).flatten()
            query_valid = queries[batch_index].index_select(0, valid_positions)
            key_valid = keys[batch_index].index_select(0, valid_positions)
            bag_instances = query_valid.shape[0]
            k_eff = min(self.requested_neighbors, bag_instances - 1)

            scores = torch.matmul(query_valid, key_valid.transpose(0, 1)) / sqrt(latent_dim)
            self_mask = torch.eye(bag_instances, dtype=torch.bool, device=scores.device)
            scores = scores.masked_fill(self_mask, -torch.inf)
            selected_scores, selected_local_indices = torch.topk(scores, k=k_eff, dim=-1)
            selected_keys = key_valid[selected_local_indices]

            stable_selected_scores = selected_scores - selected_scores.amax(
                dim=-1,
                keepdim=True,
            )
            alpha = F.softmax(stable_selected_scores, dim=-1)
            mixed = (
                alpha.unsqueeze(-1) * selected_keys
                + (1.0 - alpha).unsqueeze(-1) * query_valid.unsqueeze(1)
            )
            transformed = torch.tanh(mixed)

            key_aware_scores = torch.einsum("ikd,ikd->ik", selected_keys, transformed)
            stable_key_aware_scores = key_aware_scores - key_aware_scores.amax(
                dim=-1,
                keepdim=True,
            )
            beta = F.softmax(stable_key_aware_scores, dim=-1)
            neighborhood = torch.sum(beta.unsqueeze(-1) * selected_keys, dim=1)
            contextualized_valid = neighborhood + query_valid

            contextualized = torch.zeros_like(queries[batch_index]).index_copy(
                0,
                valid_positions,
                contextualized_valid,
            )
            contextualized_batches.append(contextualized)

            if return_details:
                indices = torch.full(
                    (n_instances, max_neighbors),
                    -1,
                    dtype=torch.long,
                    device=queries.device,
                )
                valid_neighbor_mask = torch.zeros(
                    n_instances,
                    max_neighbors,
                    dtype=torch.bool,
                    device=queries.device,
                )
                alpha_padded = queries.new_zeros((n_instances, max_neighbors))
                beta_padded = queries.new_zeros((n_instances, max_neighbors))
                selected_global_indices = valid_positions[selected_local_indices]
                indices[valid_positions, :k_eff] = selected_global_indices
                valid_neighbor_mask[valid_positions, :k_eff] = True
                alpha_padded[valid_positions, :k_eff] = alpha
                beta_padded[valid_positions, :k_eff] = beta
                index_batches.append(indices)
                neighbor_mask_batches.append(valid_neighbor_mask)
                alpha_batches.append(alpha_padded)
                beta_batches.append(beta_padded)

        contextualized_output = torch.stack(contextualized_batches)
        if return_details:
            neighbor_indices = torch.stack(index_batches)
            neighbor_mask = torch.stack(neighbor_mask_batches)
            first_stage_weights = torch.stack(alpha_batches)
            second_stage_weights = torch.stack(beta_batches)
        else:
            neighbor_indices = neighbor_mask = first_stage_weights = second_stage_weights = None

        if squeeze_batch:
            contextualized_output = contextualized_output.squeeze(0)
            if return_details:
                neighbor_indices = neighbor_indices.squeeze(0)
                neighbor_mask = neighbor_mask.squeeze(0)
                first_stage_weights = first_stage_weights.squeeze(0)
                second_stage_weights = second_stage_weights.squeeze(0)

        return (
            contextualized_output,
            neighbor_indices,
            neighbor_mask,
            first_stage_weights,
            second_stage_weights,
        )


class BagLevelAttentionPooling(nn.Module):
    """Non-gated bag-level attention pooling with hidden dimension ``D``.

    The returned tensors always retain an explicit batch dimension.
    """

    def __init__(self, latent_dim: int = 512) -> None:
        super().__init__()
        self.attention_network = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh(),
            nn.Linear(latent_dim, 1),
        )

    def forward(self, features: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if features.ndim == 2:
            features = features.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("features must have shape [B, I, D] or [I, D]")

        scores = self.attention_network(features).transpose(1, 2)
        if mask is not None:
            mask = mask.to(device=features.device, dtype=torch.bool)
            if mask.shape != features.shape[:2]:
                raise ValueError(
                    f"Expected mask shape {tuple(features.shape[:2])}, got {tuple(mask.shape)}"
                )
            if torch.any(mask.sum(dim=1) == 0):
                raise ValueError("attention pooling cannot process an empty bag")
            scores = scores.masked_fill(~mask.unsqueeze(1), -torch.inf)
        attention_weights = F.softmax(scores, dim=-1)
        pooled = torch.bmm(attention_weights, features).squeeze(1)
        return pooled, attention_weights


class DANTAMIL(nn.Module):
    """Directed Asymmetric Non-self Top-K Aggregation MIL model."""

    def __init__(self, config: DANTAConfig) -> None:
        super().__init__()
        self.config = config
        self.rsrp = RoleSpecificReembeddingProjector(
            input_dim=config.input_dim,
            latent_dim=config.latent_dim,
            dropout=config.dropout,
        )
        self.dnci = DirectedNeighborhoodConstructorIntegrator(
            requested_neighbors=config.requested_neighbors,
        )
        self.attention_pooling = BagLevelAttentionPooling(config.latent_dim)
        output_dim = config.n_classes if config.task == "classification" else config.num_intervals
        self.prediction_head = nn.Linear(config.latent_dim, output_dim)
        self.apply(initialize_weights)

    def forward(
        self,
        features: Tensor,
        mask: Tensor | None = None,
        return_attention: bool = False,
        return_neighborhood_details: bool = False,
    ) -> DANTAOutput:
        if features.ndim == 2:
            features = features.unsqueeze(0)
            if mask is not None and mask.ndim == 1:
                mask = mask.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError("features must have shape [B, I, d] or [I, d]")
        if features.shape[-1] != self.config.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.config.input_dim}, "
                f"got {features.shape[-1]}"
            )

        if mask is None:
            mask = torch.ones(
                features.shape[:2],
                dtype=torch.bool,
                device=features.device,
            )
        else:
            mask = mask.to(device=features.device, dtype=torch.bool)

        projection_dtype = self.rsrp.shared_projection[0].weight.dtype
        queries, keys = self.rsrp(features.to(dtype=projection_dtype))
        (
            contextualized,
            neighbor_indices,
            neighbor_mask,
            first_stage_weights,
            second_stage_weights,
        ) = self.dnci(
            queries,
            keys,
            mask=mask,
            return_details=return_neighborhood_details,
        )
        pooled, attention_weights = self.attention_pooling(contextualized, mask=mask)
        logits = self.prediction_head(pooled)

        probabilities = hazards = survival = risk = None
        if self.config.task == "classification":
            probabilities = F.softmax(logits, dim=1)
        else:
            hazards = torch.sigmoid(logits)
            survival = torch.cumprod(1.0 - hazards, dim=1)
            risk = -torch.sum(survival, dim=1)

        include_attention = return_attention or return_neighborhood_details
        return DANTAOutput(
            logits=logits,
            probabilities=probabilities,
            hazards=hazards,
            survival=survival,
            risk=risk,
            attention_weights=attention_weights if include_attention else None,
            query_features=queries if return_neighborhood_details else None,
            key_features=keys if return_neighborhood_details else None,
            neighbor_indices=neighbor_indices,
            neighbor_mask=neighbor_mask,
            first_stage_weights=first_stage_weights,
            second_stage_weights=second_stage_weights,
            instance_mask=mask if return_neighborhood_details else None,
        )

    def decision_aligned_localization(
        self,
        output: DANTAOutput,
        *,
        positive_class: int = 1,
        negative_class: int = 0,
    ) -> LocalizationOutput:
        """Compute the manuscript's additive binary localization score.

        ``output`` must come from a classification forward pass with
        ``return_neighborhood_details=True``.
        """

        if self.config.task != "classification":
            raise ValueError("decision-aligned localization requires a classification model")
        if self.config.n_classes != 2:
            raise ValueError(
                "the manuscript's decision-aligned localization is defined only "
                "for binary classification"
            )
        required = (
            output.attention_weights,
            output.query_features,
            output.key_features,
            output.neighbor_indices,
            output.neighbor_mask,
            output.second_stage_weights,
            output.instance_mask,
        )
        if any(value is None for value in required):
            raise ValueError("run forward with return_neighborhood_details=True")

        if not 0 <= positive_class < self.config.n_classes:
            raise ValueError("positive_class is out of range")
        if not 0 <= negative_class < self.config.n_classes:
            raise ValueError("negative_class is out of range")
        if positive_class == negative_class:
            raise ValueError("positive_class and negative_class must differ")

        weight_contrast = (
            self.prediction_head.weight[positive_class]
            - self.prediction_head.weight[negative_class]
        )
        bias_contrast = (
            self.prediction_head.bias[positive_class]
            - self.prediction_head.bias[negative_class]
        )
        contrast_norm = torch.linalg.vector_norm(weight_contrast)
        if bool(contrast_norm == 0):
            raise ValueError("classification weight contrast has zero norm")

        attention = output.attention_weights.squeeze(1)
        queries = output.query_features
        keys = output.key_features
        indices = output.neighbor_indices
        valid_neighbors = output.neighbor_mask
        beta = output.second_stage_weights
        instance_mask = output.instance_mask

        direct = attention * torch.einsum("bid,d->bi", queries, weight_contrast)
        contributions = direct.clone()
        attribution_mass = attention.clone()
        key_contrast = torch.einsum("bid,d->bi", keys, weight_contrast)

        for batch_index in range(output.logits.shape[0]):
            safe_indices = indices[batch_index].clamp_min(0)
            neighbor_mass = (
                attention[batch_index].unsqueeze(-1)
                * beta[batch_index]
                * valid_neighbors[batch_index]
            )
            neighbor_contribution = neighbor_mass * key_contrast[batch_index][safe_indices]
            contributions[batch_index].scatter_add_(
                0,
                safe_indices.flatten(),
                neighbor_contribution.flatten(),
            )
            attribution_mass[batch_index].scatter_add_(
                0,
                safe_indices.flatten(),
                neighbor_mass.flatten(),
            )

        contributions = contributions * instance_mask
        attribution_mass = attribution_mass * instance_mask
        valid_mass = attribution_mass[instance_mask]
        if not torch.isfinite(valid_mass).all() or torch.any(valid_mass <= 0):
            raise FloatingPointError(
                "localization attribution mass must be finite and positive for every patch"
            )
        scores = torch.zeros_like(contributions)
        scores[instance_mask] = contributions[instance_mask] / (
            attribution_mass[instance_mask] * contrast_norm
        )
        reconstructed = contributions.sum(dim=1) + bias_contrast
        logit_contrast = output.logits[:, positive_class] - output.logits[:, negative_class]
        reconstruction_error = reconstructed - logit_contrast
        return LocalizationOutput(
            scores=scores,
            contributions=contributions,
            attribution_mass=attribution_mass,
            reconstructed_logit_contrast=reconstructed,
            logit_contrast=logit_contrast,
            reconstruction_error=reconstruction_error,
        )
