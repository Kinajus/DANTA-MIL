"""Training and patient-level evaluation utilities for DANTA-MIL."""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from losses import NLLSurvivalLoss
from metrics import classification_metrics, concordance_index
from model import DANTAMIL


def seed_everything(seed: int = 2026) -> None:
    """Set the manuscript's deterministic random seed by default."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values from a collated batch to a device."""

    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def build_optimizer(
    model: nn.Module,
    name: str = "Adam",
    lr: float = 2e-4,
    weight_decay: float = 1e-5,
) -> torch.optim.Optimizer:
    """Build the optimizer; manuscript runs use Adam."""

    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    epochs: int = 200,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build a learning-rate scheduler; manuscript runs use cosine annealing."""

    name = str(name).lower()
    if name in {"none", "null", ""}:
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=max(1, epochs // 2),
            gamma=0.1,
        )
    raise ValueError(f"Unsupported scheduler: {name}")


def train_one_epoch(
    model: DANTAMIL,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    task: str,
    amp: bool = False,
) -> float:
    """Train for one epoch and return mean unweighted bag-level loss."""

    model.train()
    scaler = GradScaler(enabled=amp and device.type == "cuda")
    classification_loss = nn.CrossEntropyLoss()
    survival_loss = NLLSurvivalLoss(alpha=0.0)
    total_loss = 0.0

    for batch in tqdm(loader, desc="train", leave=False):
        batch = move_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp and device.type == "cuda"):
            output = model(batch["features"], mask=batch["mask"])
            if task == "classification":
                loss = classification_loss(output.logits, batch["label"])
            else:
                loss = survival_loss(
                    hazards=output.hazards,
                    survival=output.survival,
                    targets=batch["survival_bin"],
                    censorship=batch["censorship"],
                )
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += float(loss.detach().cpu())
    return total_loss / max(1, len(loader))


def aggregate_classification_by_patient(
    patient_ids: Sequence[str],
    labels: Sequence[int],
    probabilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Average WSI class-probability vectors within each patient."""

    probabilities = np.asarray(probabilities, dtype=float)
    lengths_match = len(patient_ids) == len(labels) == len(probabilities)
    if probabilities.ndim != 2 or not lengths_match:
        raise ValueError("patient IDs, labels, and probability rows must have matching lengths")

    records: dict[str, dict[str, Any]] = {}
    for patient_id, label, probability in zip(patient_ids, labels, probabilities, strict=True):
        patient_id = str(patient_id)
        if patient_id not in records:
            records[patient_id] = {"label": int(label), "probabilities": []}
        elif records[patient_id]["label"] != int(label):
            raise ValueError(f"Patient {patient_id} has inconsistent classification labels")
        records[patient_id]["probabilities"].append(probability)

    patient_labels = np.asarray([record["label"] for record in records.values()], dtype=int)
    patient_probabilities = np.stack(
        [np.mean(record["probabilities"], axis=0) for record in records.values()]
    )
    return patient_labels, patient_probabilities


def aggregate_survival_by_patient(
    patient_ids: Sequence[str],
    risks: Sequence[float],
    event_times: Sequence[float],
    event_observed: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average WSI bag-level risk scores within each patient."""

    if not (len(patient_ids) == len(risks) == len(event_times) == len(event_observed)):
        raise ValueError("survival aggregation inputs must have matching lengths")

    records: dict[str, dict[str, Any]] = {}
    for patient_id, risk, event_time, observed in zip(
        patient_ids,
        risks,
        event_times,
        event_observed,
        strict=True,
    ):
        patient_id = str(patient_id)
        if patient_id not in records:
            records[patient_id] = {
                "risks": [],
                "event_time": float(event_time),
                "event_observed": int(observed),
            }
        else:
            record = records[patient_id]
            time_differs = record["event_time"] != float(event_time)
            status_differs = record["event_observed"] != int(observed)
            if time_differs or status_differs:
                raise ValueError(f"Patient {patient_id} has inconsistent survival outcomes")
        records[patient_id]["risks"].append(float(risk))

    patient_risks = np.asarray(
        [np.mean(record["risks"]) for record in records.values()],
        dtype=float,
    )
    patient_times = np.asarray(
        [record["event_time"] for record in records.values()],
        dtype=float,
    )
    patient_observed = np.asarray(
        [record["event_observed"] for record in records.values()],
        dtype=int,
    )
    return patient_risks, patient_times, patient_observed


@torch.no_grad()
def evaluate_classification(
    model: DANTAMIL,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate after arithmetic-mean probability aggregation by patient."""

    model.eval()
    patient_ids: list[str] = []
    labels: list[int] = []
    probability_batches: list[np.ndarray] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        patient_ids.extend(map(str, batch["patient_id"]))
        batch = move_to_device(batch, device)
        output = model(batch["features"], mask=batch["mask"])
        labels.extend(batch["label"].detach().cpu().tolist())
        probability_batches.append(output.probabilities.detach().cpu().numpy())
    patient_labels, patient_probabilities = aggregate_classification_by_patient(
        patient_ids,
        labels,
        np.concatenate(probability_batches, axis=0),
    )
    return classification_metrics(patient_labels, patient_probabilities)


@torch.no_grad()
def evaluate_survival(
    model: DANTAMIL,
    loader: DataLoader,
    *,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate Harrell C-index after arithmetic-mean risk aggregation by patient."""

    model.eval()
    patient_ids: list[str] = []
    risks: list[float] = []
    event_times: list[float] = []
    event_observed: list[int] = []
    for batch in tqdm(loader, desc="eval", leave=False):
        patient_ids.extend(map(str, batch["patient_id"]))
        batch = move_to_device(batch, device)
        output = model(batch["features"], mask=batch["mask"])
        risks.extend(output.risk.detach().cpu().tolist())
        event_times.extend(batch["event_time"].detach().cpu().tolist())
        event_observed.extend((1.0 - batch["censorship"]).detach().cpu().int().tolist())
    patient_risks, patient_times, patient_observed = aggregate_survival_by_patient(
        patient_ids,
        risks,
        event_times,
        event_observed,
    )
    return {
        "c_index": concordance_index(
            event_times=patient_times,
            risks=patient_risks,
            event_observed=patient_observed,
        )
    }


def save_checkpoint(
    path: str | Path,
    *,
    model: DANTAMIL,
    epoch: int,
    metric_name: str,
    metric_value: float,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    label_map: dict[Any, int] | None = None,
    survival_boundaries: Sequence[float] | None = None,
) -> None:
    """Save model state together with fold-specific inference metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_name": "DANTA-MIL",
        "implementation_version": "0.2.0",
        "epoch": epoch,
        "metric_name": metric_name,
        "metric_value": metric_value,
        "model_state": model.state_dict(),
        "model_config": model.config.__dict__,
        "config": config,
        "label_map": label_map,
        "survival_boundaries": (
            list(map(float, survival_boundaries)) if survival_boundaries is not None else None
        ),
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    torch.save(payload, path)


def append_metrics_csv(path: str | Path, row: dict[str, Any]) -> None:
    """Append one metrics row to a CSV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
