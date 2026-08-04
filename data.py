"""Dataset utilities for manifest-based pathology feature bags."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset


FEATURE_KEYS = ("features", "feats", "embeddings", "x", "bag")


def normalize_label_key(value: Any) -> Any:
    """Normalize manifest labels before looking them up in a label map."""

    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return int(value)
    return str(value)


def read_manifest(manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a CSV manifest or return a copy of an existing dataframe."""

    if isinstance(manifest, pd.DataFrame):
        return manifest.copy()
    manifest = Path(manifest)
    if manifest.suffix.lower() in {".tsv", ".txt"}:
        return pd.read_csv(manifest, sep="\t")
    return pd.read_csv(manifest)


def resolve_feature_path(path: str | Path, root: str | Path | None = None) -> Path:
    """Resolve a feature path relative to a root directory when needed."""

    path = Path(path)
    if path.is_absolute() or root is None:
        return path
    return Path(root) / path


def load_feature_tensor(path: str | Path, map_location: str = "cpu") -> Tensor:
    """Load a cached patch-feature tensor from a ``.pt`` file."""

    obj = torch.load(path, map_location=map_location)
    if isinstance(obj, dict):
        for key in FEATURE_KEYS:
            if key in obj:
                obj = obj[key]
                break
        else:
            raise KeyError(f"No feature tensor key found in {path}; tried {FEATURE_KEYS}")
    if not torch.is_tensor(obj):
        obj = torch.as_tensor(obj)
    obj = obj.float()
    if obj.ndim == 1:
        obj = obj.unsqueeze(0)
    if obj.ndim == 3 and obj.shape[0] == 1:
        obj = obj.squeeze(0)
    if obj.ndim != 2:
        raise ValueError(f"Feature tensor at {path} must have shape [I, d], got {tuple(obj.shape)}")
    if obj.shape[0] < 2:
        raise ValueError(
            f"Feature tensor at {path} has I={obj.shape[0]}; DNCI requires at least two instances"
        )
    return obj


def build_label_map(values: pd.Series) -> dict[Any, int]:
    """Build a stable mapping for string or mixed labels."""

    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise ValueError("Classification labels must be complete and non-empty")
    cleaned = [normalize_label_key(value) for value in values.tolist()]
    if not cleaned:
        raise ValueError("Classification manifest must contain at least one label")
    if all(isinstance(value, int) for value in cleaned):
        unique = sorted(set(cleaned))
        return {value: index for index, value in enumerate(unique)}
    labels = sorted({str(value) for value in cleaned})
    return {label: index for index, label in enumerate(labels)}


def encode_label(value: Any, label_map: dict[Any, int]) -> int:
    """Encode a label using the canonical numeric-or-string map representation."""

    normalized = normalize_label_key(value)
    if normalized in label_map:
        return int(label_map[normalized])
    string_key = str(normalized)
    if string_key in label_map:
        return int(label_map[string_key])
    raise ValueError(f"Label {value!r} is absent from the configured label map")


def _unique_patient_outcomes(
    df: pd.DataFrame,
    *,
    patient_id_column: str,
    event_time_column: str,
    event_observed_column: str,
) -> pd.DataFrame:
    required = {patient_id_column, event_time_column, event_observed_column}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Survival manifest is missing columns: {missing}")

    outcomes = df[[patient_id_column, event_time_column, event_observed_column]].copy()
    outcomes[event_time_column] = pd.to_numeric(outcomes[event_time_column], errors="raise")
    observed_numeric = pd.to_numeric(
        outcomes[event_observed_column],
        errors="raise",
    )
    outcomes[event_observed_column] = observed_numeric
    if outcomes[[patient_id_column, event_time_column, event_observed_column]].isna().any().any():
        raise ValueError("Survival patient ID, time, and event status must be complete")
    if outcomes[patient_id_column].astype(str).str.strip().eq("").any():
        raise ValueError("patient_id must not be empty")
    if not np.isfinite(outcomes[event_time_column].to_numpy(dtype=float)).all():
        raise ValueError("event_time must be finite")
    if not np.isfinite(observed_numeric.to_numpy(dtype=float)).all():
        raise ValueError("event_observed must be finite")
    if not observed_numeric.isin([0, 1]).all():
        raise ValueError("event_observed must contain only 0 and 1")
    outcomes[event_observed_column] = observed_numeric.astype(int)
    if (outcomes[event_time_column] < 0).any():
        raise ValueError("event_time must be non-negative")

    consistency = outcomes.groupby(patient_id_column, sort=False).agg(
        time_values=(event_time_column, "nunique"),
        event_values=(event_observed_column, "nunique"),
    )
    inconsistent = consistency[(consistency.time_values != 1) | (consistency.event_values != 1)]
    if not inconsistent.empty:
        raise ValueError(
            "All WSIs from a patient must share one survival outcome; inconsistent patients: "
            + ", ".join(map(str, inconsistent.index[:10]))
        )
    return outcomes.drop_duplicates(patient_id_column, keep="first").reset_index(drop=True)


def derive_survival_interval_boundaries(
    train_df: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
    event_time_column: str = "event_time",
    event_observed_column: str = "event_observed",
) -> tuple[float, float, float]:
    """Derive the manuscript's 25/50/75% boundaries from training deaths only.

    Patients, rather than WSI rows, are the analytical unit. NumPy's
    ``method='linear'`` is Hyndman--Fan type 7. Validation outcomes must never
    be passed to this function.
    """

    outcomes = _unique_patient_outcomes(
        train_df,
        patient_id_column=patient_id_column,
        event_time_column=event_time_column,
        event_observed_column=event_observed_column,
    )
    death_times = outcomes.loc[
        outcomes[event_observed_column] == 1,
        event_time_column,
    ].to_numpy(dtype=float)
    if death_times.size < 4:
        raise ValueError("At least four unique training patients with observed deaths are required")
    boundaries = np.quantile(death_times, [0.25, 0.50, 0.75], method="linear")
    if not np.all(np.diff(boundaries) > 0):
        raise ValueError(
            "Training-death quantiles are not strictly increasing; four non-empty intervals "
            "cannot be constructed"
        )
    return tuple(float(value) for value in boundaries)


def validate_survival_patient_outcomes(
    df: pd.DataFrame,
    *,
    patient_id_column: str = "patient_id",
    event_time_column: str = "event_time",
    event_observed_column: str = "event_observed",
) -> None:
    """Validate that every WSI from a patient carries one complete outcome."""

    _unique_patient_outcomes(
        df,
        patient_id_column=patient_id_column,
        event_time_column=event_time_column,
        event_observed_column=event_observed_column,
    )


def validate_censorship_consistency(
    df: pd.DataFrame,
    *,
    event_observed_column: str = "event_observed",
    censorship_column: str = "censorship",
) -> None:
    """Require ``censorship == 1 - event_observed`` when both are supplied."""

    if censorship_column not in df.columns:
        return
    if event_observed_column not in df.columns:
        raise ValueError(
            f"'{event_observed_column}' is required to validate '{censorship_column}'"
        )
    observed = pd.to_numeric(df[event_observed_column], errors="raise")
    censorship = pd.to_numeric(df[censorship_column], errors="raise")
    if observed.isna().any() or censorship.isna().any():
        raise ValueError("event_observed and censorship must be complete")
    if not np.isfinite(observed.to_numpy(dtype=float)).all() or not np.isfinite(
        censorship.to_numpy(dtype=float)
    ).all():
        raise ValueError("event_observed and censorship must be finite")
    if not observed.isin([0, 1]).all() or not censorship.isin([0, 1]).all():
        raise ValueError("event_observed and censorship must contain only 0 and 1")
    inconsistent = censorship.astype(int) != 1 - observed.astype(int)
    if inconsistent.any():
        raise ValueError("censorship must equal 1 - event_observed for every WSI row")


def assign_survival_bins(
    df: pd.DataFrame,
    boundaries: Sequence[float],
    *,
    event_time_column: str = "event_time",
    output_column: str = "survival_bin",
) -> pd.DataFrame:
    """Apply fixed training-derived boundaries using right-boundary assignment.

    Returned bins are zero-based (0--3) for direct PyTorch indexing. An
    observation exactly equal to a boundary is assigned to the interval on its
    right, matching the manuscript's ``[b_r, b_{r+1})`` convention.
    """

    boundary_array = np.asarray(boundaries, dtype=float)
    valid_boundaries = (
        boundary_array.shape == (3,)
        and np.isfinite(boundary_array).all()
        and np.all(np.diff(boundary_array) > 0)
    )
    if not valid_boundaries:
        raise ValueError("boundaries must contain three finite strictly increasing values")
    result = df.copy()
    times = pd.to_numeric(result[event_time_column], errors="raise").to_numpy(dtype=float)
    if not np.isfinite(times).all() or (times < 0).any():
        raise ValueError("event_time must contain complete finite non-negative values")
    result[output_column] = np.searchsorted(boundary_array, times, side="right").astype(int)
    return result


class FeatureBagDataset(Dataset):
    """One cached WSI or TMA-core feature bag per dataset item."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        *,
        root: str | Path | None = None,
        task: str = "classification",
        feature_column: str = "feature_path",
        id_column: str = "slide_id",
        patient_id_column: str = "patient_id",
        label_column: str = "label",
        event_time_column: str = "event_time",
        event_observed_column: str = "event_observed",
        censorship_column: str = "censorship",
        survival_bin_column: str = "survival_bin",
        label_map: dict[Any, int] | None = None,
    ) -> None:
        self.df = read_manifest(manifest).reset_index(drop=True)
        self.root = root
        self.task = task
        self.feature_column = feature_column
        self.id_column = id_column
        self.patient_id_column = patient_id_column
        self.label_column = label_column
        self.event_time_column = event_time_column
        self.event_observed_column = event_observed_column
        self.censorship_column = censorship_column
        self.survival_bin_column = survival_bin_column
        self.label_map = label_map

        if task not in {"classification", "survival"}:
            raise ValueError("task must be 'classification' or 'survival'")
        if feature_column not in self.df.columns:
            raise ValueError(f"Manifest is missing feature column: {feature_column}")
        if id_column not in self.df.columns:
            raise ValueError(f"Manifest is missing image ID column: {id_column}")
        if task == "survival" and survival_bin_column not in self.df.columns:
            raise ValueError("survival bins must be assigned from the training partition first")
        if task == "survival":
            validate_survival_patient_outcomes(
                self.df,
                patient_id_column=patient_id_column,
                event_time_column=event_time_column,
                event_observed_column=event_observed_column,
            )
            validate_censorship_consistency(
                self.df,
                event_observed_column=event_observed_column,
                censorship_column=censorship_column,
            )

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        slide_id = str(row[self.id_column])
        patient_id = (
            str(row[self.patient_id_column])
            if self.patient_id_column in self.df.columns
            else slide_id
        )
        feature_path = resolve_feature_path(row[self.feature_column], self.root)
        item: dict[str, Any] = {
            "slide_id": slide_id,
            "patient_id": patient_id,
            "features": load_feature_tensor(feature_path),
        }

        if self.task == "classification":
            value = row[self.label_column]
            if self.label_map is not None:
                item["label"] = encode_label(value, self.label_map)
            else:
                item["label"] = int(value)
        else:
            item["event_time"] = float(row[self.event_time_column])
            if self.censorship_column in self.df.columns:
                censorship = int(row[self.censorship_column])
            else:
                censorship = 0 if int(row[self.event_observed_column]) == 1 else 1
            item["censorship"] = censorship
            item["survival_bin"] = int(row[self.survival_bin_column])
        return item


def collate_bags(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length bags and create a boolean instance mask."""

    if not batch:
        raise ValueError("Cannot collate an empty batch")
    dims = {item["features"].shape[1] for item in batch}
    if len(dims) != 1:
        raise ValueError(f"All bags in a batch must share feature dim, got {sorted(dims)}")

    batch_size = len(batch)
    max_instances = max(item["features"].shape[0] for item in batch)
    feature_dim = batch[0]["features"].shape[1]
    features = torch.zeros(batch_size, max_instances, feature_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_instances, dtype=torch.bool)

    for index, item in enumerate(batch):
        bag = item["features"]
        n_instances = bag.shape[0]
        features[index, :n_instances] = bag
        mask[index, :n_instances] = True

    output: dict[str, Any] = {
        "slide_id": [item["slide_id"] for item in batch],
        "patient_id": [item["patient_id"] for item in batch],
        "features": features,
        "mask": mask,
    }
    if "label" in batch[0]:
        output["label"] = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    if "survival_bin" in batch[0]:
        output["event_time"] = torch.tensor(
            [item["event_time"] for item in batch],
            dtype=torch.float32,
        )
        output["censorship"] = torch.tensor(
            [item["censorship"] for item in batch],
            dtype=torch.float32,
        )
        output["survival_bin"] = torch.tensor(
            [item["survival_bin"] for item in batch],
            dtype=torch.long,
        )
    return output
