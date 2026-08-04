"""YAML-driven DANTA-MIL training entrypoint."""

from __future__ import annotations

import argparse
import math
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from torch.utils.data import DataLoader

from data import (
    FeatureBagDataset,
    assign_survival_bins,
    build_label_map,
    collate_bags,
    derive_survival_interval_boundaries,
    normalize_label_key,
    read_manifest,
    validate_censorship_consistency,
    validate_survival_patient_outcomes,
)
from model import DANTAConfig, DANTAMIL
from training import (
    append_metrics_csv,
    build_optimizer,
    build_scheduler,
    evaluate_classification,
    evaluate_survival,
    save_checkpoint,
    seed_everything,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DANTA-MIL from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to a YAML config.")
    parser.add_argument("--fold", type=int, default=None, help="Override the validation fold.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=None,
        help="Override the repeated-CV repetition index used for generated folds.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Build data/model and exit.")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _validate_patient_partition(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    patient_id_column: str,
) -> None:
    if train_df.empty or val_df.empty:
        raise ValueError("Train/validation split is empty. Check split or fold settings.")
    train_patients = set(train_df[patient_id_column].astype(str))
    val_patients = set(val_df[patient_id_column].astype(str))
    overlap = sorted(train_patients.intersection(val_patients))
    if overlap:
        raise ValueError(
            "Patients cannot contribute WSIs to both partitions; overlapping IDs: "
            + ", ".join(overlap[:10])
        )


def split_dataframe(
    df: pd.DataFrame,
    *,
    task: str,
    split_cfg: dict[str, Any],
    label_column: str,
    patient_id_column: str,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create or read a patient-level train/validation partition."""

    if task not in {"classification", "survival"}:
        raise ValueError("task must be 'classification' or 'survival'")
    if patient_id_column not in df.columns:
        raise ValueError(
            f"Manifest must contain patient ID column '{patient_id_column}' so all WSIs "
            "from one patient remain in one partition"
        )
    patient_ids = df[patient_id_column]
    if patient_ids.isna().any() or patient_ids.astype(str).str.strip().eq("").any():
        raise ValueError(f"Patient ID column '{patient_id_column}' contains missing or empty IDs")
    if task == "classification":
        if label_column not in df.columns:
            raise ValueError(f"Classification manifest is missing label column: {label_column}")
        labels = df[label_column]
        if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
            raise ValueError("Classification labels must be complete and non-empty")
        canonical_labels = labels.map(lambda value: str(normalize_label_key(value)))
        patient_labels = canonical_labels.groupby(df[patient_id_column], sort=False).nunique()
        if (patient_labels != 1).any():
            inconsistent = patient_labels[patient_labels != 1].index.astype(str).tolist()
            raise ValueError(
                "All WSIs from a patient must share one classification label; inconsistent IDs: "
                + ", ".join(inconsistent[:10])
            )

    split_column = split_cfg.get("split_column", "split")
    fold_column = split_cfg.get("fold_column", "fold")
    fold = int(split_cfg.get("fold", 0))
    repeat = int(split_cfg.get("repeat", 0))
    generate_folds = bool(split_cfg.get("generate_folds", False))

    validation_ids_file = split_cfg.get("validation_ids_file")
    if validation_ids_file:
        split_path = Path(
            str(validation_ids_file).format(
                fold=fold,
                repeat=repeat,
            )
        )
        if not split_path.is_file():
            raise FileNotFoundError(f"Validation-ID split file not found: {split_path}")
        split_frame = pd.read_csv(split_path)
        split_id_column = str(split_cfg.get("split_id_column", "ID"))
        if split_id_column not in split_frame.columns:
            raise ValueError(
                f"Split file {split_path} is missing ID column '{split_id_column}'"
            )
        split_ids = split_frame[split_id_column].astype(str).str.strip()
        if split_ids.empty or split_ids.eq("").any() or split_ids.duplicated().any():
            raise ValueError(
                f"Split file {split_path} must contain unique, non-empty validation IDs"
            )
        manifest_ids = df[patient_id_column].astype(str).str.strip()
        missing_ids = sorted(set(split_ids).difference(set(manifest_ids)))
        if missing_ids:
            raise ValueError(
                f"Split file {split_path} contains IDs absent from the manifest: "
                + ", ".join(missing_ids[:10])
            )
        if "slice_count" in split_frame.columns:
            expected_counts = pd.Series(
                pd.to_numeric(split_frame["slice_count"], errors="raise").to_numpy(),
                index=split_ids,
            ).astype(int)
            observed_counts = manifest_ids[manifest_ids.isin(set(split_ids))].value_counts()
            mismatched = expected_counts[expected_counts != observed_counts.reindex(expected_counts.index)]
            if not mismatched.empty:
                raise ValueError(
                    f"Split file {split_path} slice_count does not match the manifest for IDs: "
                    + ", ".join(map(str, mismatched.index[:10]))
                )
        validation_set = set(split_ids)
        train_df = df[~manifest_ids.isin(validation_set)].copy()
        val_df = df[manifest_ids.isin(validation_set)].copy()
        _validate_patient_partition(train_df, val_df, patient_id_column)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    if not generate_folds and split_column in df.columns:
        split_values = df[split_column].astype(str).str.lower()
        train_values = {str(value).lower() for value in split_cfg.get("train_values", ["train"])}
        val_values = {
            str(value).lower()
            for value in split_cfg.get("val_values", ["val", "valid", "validation", "test"])
        }
        train_df = df[split_values.isin(train_values)].copy()
        val_df = df[split_values.isin(val_values)].copy()
        _validate_patient_partition(train_df, val_df, patient_id_column)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    if not generate_folds and fold_column in df.columns:
        train_df = df[df[fold_column].astype(int) != fold].copy()
        val_df = df[df[fold_column].astype(int) == fold].copy()
        _validate_patient_partition(train_df, val_df, patient_id_column)
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True)

    patient_frame = df.drop_duplicates(patient_id_column, keep="first").reset_index(drop=True)
    n_splits = int(split_cfg.get("n_splits", 5))
    if repeat < 0:
        raise ValueError("repeat must be non-negative")
    if not 0 <= fold < n_splits:
        raise ValueError(f"fold must be in [0, {n_splits - 1}]")
    indices = list(range(len(patient_frame)))
    if task == "classification":
        splitter = RepeatedStratifiedKFold(
            n_splits=n_splits,
            n_repeats=repeat + 1,
            random_state=seed,
        )
        labels = patient_frame[label_column].map(
            lambda value: str(normalize_label_key(value))
        ).to_numpy()
        splits = list(splitter.split(indices, labels))
    else:
        splitter = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=repeat + 1,
            random_state=seed,
        )
        splits = list(splitter.split(indices))
    split_index = repeat * n_splits + fold
    train_patient_indices, val_patient_indices = splits[split_index]
    train_patients = set(
        patient_frame.iloc[train_patient_indices][patient_id_column].astype(str)
    )
    val_patients = set(patient_frame.iloc[val_patient_indices][patient_id_column].astype(str))
    patient_strings = df[patient_id_column].astype(str)
    train_df = df[patient_strings.isin(train_patients)].copy()
    val_df = df[patient_strings.isin(val_patients)].copy()
    _validate_patient_partition(train_df, val_df, patient_id_column)
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    task = config.get("task", "classification")
    if task not in {"classification", "survival"}:
        raise ValueError("task must be 'classification' or 'survival'")
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("train", {})
    split_cfg = config.get("split", {})
    if args.fold is not None:
        split_cfg["fold"] = args.fold
    if args.repeat is not None:
        split_cfg["repeat"] = args.repeat

    seed = int(train_cfg.get("seed", 2026))
    seed_everything(seed)

    dataframe = read_manifest(data_cfg["manifest"])
    label_column = data_cfg.get("label_column", "label")
    patient_id_column = data_cfg.get("patient_id_column", "patient_id")
    train_df, val_df = split_dataframe(
        dataframe,
        task=task,
        split_cfg=split_cfg,
        label_column=label_column,
        patient_id_column=patient_id_column,
        seed=seed,
    )

    survival_boundaries: tuple[float, float, float] | None = None
    if task == "survival":
        event_time_column = data_cfg.get("event_time_column", "event_time")
        event_observed_column = data_cfg.get("event_observed_column", "event_observed")
        survival_bin_column = data_cfg.get("survival_bin_column", "survival_bin")
        validate_survival_patient_outcomes(
            train_df,
            patient_id_column=patient_id_column,
            event_time_column=event_time_column,
            event_observed_column=event_observed_column,
        )
        validate_survival_patient_outcomes(
            val_df,
            patient_id_column=patient_id_column,
            event_time_column=event_time_column,
            event_observed_column=event_observed_column,
        )
        validate_censorship_consistency(
            train_df,
            event_observed_column=event_observed_column,
            censorship_column=data_cfg.get("censorship_column", "censorship"),
        )
        validate_censorship_consistency(
            val_df,
            event_observed_column=event_observed_column,
            censorship_column=data_cfg.get("censorship_column", "censorship"),
        )
        survival_boundaries = derive_survival_interval_boundaries(
            train_df,
            patient_id_column=patient_id_column,
            event_time_column=event_time_column,
            event_observed_column=event_observed_column,
        )
        train_df = assign_survival_bins(
            train_df,
            survival_boundaries,
            event_time_column=event_time_column,
            output_column=survival_bin_column,
        )
        val_df = assign_survival_bins(
            val_df,
            survival_boundaries,
            event_time_column=event_time_column,
            output_column=survival_bin_column,
        )

    label_map = build_label_map(dataframe[label_column]) if task == "classification" else None
    n_classes = int(model_cfg.get("n_classes", len(label_map) if label_map else 2))
    if label_map is not None and n_classes != len(label_map):
        raise ValueError(
            f"Configured n_classes={n_classes} does not match {len(label_map)} manifest labels"
        )
    danta_config = DANTAConfig(
        input_dim=int(model_cfg.get("input_dim", 1024)),
        latent_dim=int(model_cfg.get("latent_dim", 512)),
        n_classes=n_classes,
        num_intervals=int(model_cfg.get("num_intervals", 4)),
        requested_neighbors=int(model_cfg.get("requested_neighbors", 6)),
        dropout=float(model_cfg.get("dropout", 0.25)),
        task=task,
    )

    dataset_kwargs = {
        "root": data_cfg.get("root"),
        "task": task,
        "feature_column": data_cfg.get("feature_column", "feature_path"),
        "id_column": data_cfg.get("id_column", "slide_id"),
        "patient_id_column": patient_id_column,
        "label_column": label_column,
        "event_time_column": data_cfg.get("event_time_column", "event_time"),
        "event_observed_column": data_cfg.get("event_observed_column", "event_observed"),
        "censorship_column": data_cfg.get("censorship_column", "censorship"),
        "survival_bin_column": data_cfg.get("survival_bin_column", "survival_bin"),
        "label_map": label_map,
    }
    train_set = FeatureBagDataset(train_df, **dataset_kwargs)
    val_set = FeatureBagDataset(val_df, **dataset_kwargs)

    batch_size = int(train_cfg.get("batch_size", 1))
    if batch_size != 1:
        raise ValueError("The manuscript training protocol fixes batch_size=1")
    num_workers = int(train_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_bags,
    )

    device = torch.device(train_cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = DANTAMIL(danta_config).to(device)
    optimizer = build_optimizer(
        model,
        name=train_cfg.get("optimizer", "Adam"),
        lr=float(train_cfg.get("lr", 2e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-5)),
    )
    epochs = int(train_cfg.get("epochs", 200))
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    scheduler = build_scheduler(optimizer, train_cfg.get("scheduler", "cosine"), epochs=epochs)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    repeat = int(split_cfg.get("repeat", 0))
    fold = int(split_cfg.get("fold", 0))
    output_dir = (
        Path(train_cfg.get("output_dir", "runs"))
        / config.get("name", "danta")
        / f"repeat_{repeat}"
        / f"fold_{fold}"
        / timestamp
    )
    metrics_path = output_dir / "metrics.csv"
    best_path = output_dir / "best_model.pt"
    metric_name = train_cfg.get("best_metric", "auc" if task == "classification" else "c_index")
    patience = int(train_cfg.get("early_stopping_patience", 20))
    required_metric = "auc" if task == "classification" else "c_index"
    if metric_name != required_metric:
        raise ValueError(
            f"The manuscript checkpoint criterion for task={task!r} is {required_metric!r}"
        )
    if patience < 1:
        raise ValueError("early_stopping_patience must be at least 1")
    amp = bool(train_cfg.get("amp", False))

    parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"[DANTA-MIL] task={task} train_wsis={len(train_set)} val_wsis={len(val_set)} "
        f"device={device} parameters={parameter_count:,}"
    )
    print(f"[DANTA-MIL] output_dir={output_dir}")
    if survival_boundaries is not None:
        print(f"[DANTA-MIL] training-derived survival boundaries={survival_boundaries}")
    if args.dry_run:
        return

    best_metric = float("-inf")
    epochs_without_improvement = 0
    for epoch in range(epochs):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device=device,
            task=task,
            amp=amp,
        )
        if scheduler is not None:
            scheduler.step()
        metrics = (
            evaluate_classification(model, val_loader, device=device)
            if task == "classification"
            else evaluate_survival(model, val_loader, device=device)
        )
        row = {"epoch": epoch, "train_loss": train_loss, **metrics}
        append_metrics_csv(metrics_path, row)

        current = metrics.get(metric_name, float("nan"))
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} "
            + " ".join(f"{key}={value:.5f}" for key, value in metrics.items())
        )
        if not math.isfinite(current):
            raise RuntimeError(
                f"Validation {metric_name} is not finite; check validation outcomes/classes"
            )
        if current > best_metric:
            best_metric = current
            epochs_without_improvement = 0
            save_checkpoint(
                best_path,
                model=model,
                epoch=epoch,
                metric_name=metric_name,
                metric_value=best_metric,
                config=config,
                optimizer=optimizer,
                scheduler=scheduler,
                label_map=label_map,
                survival_boundaries=survival_boundaries,
            )
            print(f"[DANTA-MIL] saved best checkpoint: {best_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(
                    f"[DANTA-MIL] early stopping after {patience} epochs without "
                    f"validation {metric_name} improvement"
                )
                break


if __name__ == "__main__":
    main()
