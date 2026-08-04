# DANTA-MIL

Author implementation accompanying the manuscript:

**DANTA-MIL: Directed Asymmetric Non-self Top-$K$ Aggregation for Task-Adaptive Re-embedding of Digital Pathology Features**

DANTA-MIL is a multiple instance learning framework for task-adaptive re-embedding of frozen digital-pathology patch features. It uses a shared nonlinear projection, separate query and key mappings, exact row-wise non-self Top-$K$ neighborhood construction, two-stage neighbor weighting, a residual update, and attention-based bag pooling.

The current repository contains the core DANTA-MIL implementation, training entry point, example task configurations, and the patient/WSI-level cross-validation assignments used for the public cohorts. The model is trained from pre-extracted patch-feature bags rather than directly from whole-slide image pixels.

## Repository contents

| Path | Description |
| --- | --- |
| `model.py` | DANTA-MIL model, including the Role-Specific Re-embedding Projector (RSRP), Directed Neighborhood Constructor--Integrator (DNCI), attention pooling, and task-specific prediction heads. |
| `data.py` | Manifest loading, cached feature-bag loading, survival-interval construction, and dataset utilities. |
| `losses.py` | Censoring-aware discrete-time survival loss. |
| `metrics.py` | Classification metrics and Harrell's concordance index. |
| `training.py` | Training, patient-level prediction aggregation, validation, deterministic seeding, and checkpoint serialization. |
| `train.py` | Command-line training entry point. |
| `configs/` | Example YAML configurations for classification and overall-survival experiments. |
| `split/` | Fixed patient/WSI-level cross-validation assignments for the public-cohort experiments. |
| `requirements.txt` | Direct Python dependencies of the core training implementation. |

## Experimental environment

The manuscript experiments were conducted with:

- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1
- one NVIDIA GeForce RTX 4090 GPU with 24 GB VRAM
- FP32 and a batch size of one for the reported model-stage resource benchmark

Install the direct dependencies of the core implementation from the repository root:

```bash
pip install -r requirements.txt
```

For the closest match to the reported environment, install PyTorch 2.1.0 with a CUDA 12.1-compatible build before installing the remaining dependencies. Exact driver and system-level packages may differ across machines.

## Data access

The public-cohort experiments use:

- CAMELYON-16, available from the official CAMELYON-16 challenge data provider.
- The Cancer Genome Atlas (TCGA) diagnostic pathology images and clinical data, available through the [NCI Genomic Data Commons](https://gdc.cancer.gov/access-data/gdc-data-portal).

The private SMUZH-NSCLC cohort is not distributed in this repository because it is subject to institutional, ethical, and privacy restrictions. These restrictions do not apply to the DANTA-MIL source code, public-cohort split files, or public-cohort model configurations.

Raw WSIs, cached patch embeddings, TCGA clinical files, and the private external-cohort data are not redistributed here. Users are responsible for obtaining the public datasets from their official providers and complying with the corresponding terms of use.

## WSI preprocessing and patch-feature extraction

WSI tissue segmentation, patch-coordinate extraction, and background/hole exclusion use the public [CLAM](https://github.com/mahmoodlab/CLAM) preprocessing pipeline rather than a duplicated copy of CLAM in this repository.

The reference CLAM revision used for the instructions below is:

```text
Repository: https://github.com/mahmoodlab/CLAM
Commit: 53e2409d4a8189c682c173382964a85f114f923c
```

Obtain the pinned CLAM revision with:

```bash
git clone https://github.com/mahmoodlab/CLAM.git
cd CLAM
git checkout 53e2409d4a8189c682c173382964a85f114f923c
```

The preprocessing settings used in the manuscript were:

- target magnification: $20\times$
- patch size: $512\times512$ pixels
- step size: 512 pixels
- non-overlapping patches
- tissue segmentation and exclusion of blank background regions and holes using CLAM

The CLAM fast patching entry point can be called as follows after selecting the WSI pyramid level corresponding to $20\times$ magnification and an appropriate CLAM segmentation preset for the source cohort:

```bash
python create_patches_fp.py \
  --source DATA_DIRECTORY \
  --save_dir PATCH_RESULTS_DIRECTORY \
  --patch_size 512 \
  --step_size 512 \
  --patch_level LEVEL_CORRESPONDING_TO_20X \
  --preset DATASET_SPECIFIC_PRESET.csv \
  --seg --patch
```

The generated coordinate HDF5 files and `process_list_autogen.csv` should be retained so that the applied slide-specific segmentation and patching parameters remain auditable. The pyramid level must be selected from the scan metadata; it should not be inferred solely from the level number because scanners may expose different pyramids.

Patch embeddings were extracted offline with frozen encoders:

| Encoder | Reference identifier | Stored feature dimension |
| --- | --- | ---: |
| ResNet-50 | ImageNet-1K-pretrained ResNet-50 | 1024 |
| PLIP | `vinid/plip` | 512 |
| UNI | `MahmoodLab/UNI` | 1024 |
| CONCH | `MahmoodLab/CONCH` | 512 |

CLAM provides `extract_features_fp.py` and documented support for ResNet-50, UNI, and CONCH feature extraction. PLIP should be loaded from the identifier above using its corresponding image preprocessing. Regardless of encoder, DANTA-MIL expects one PyTorch feature tensor per pathology image with shape `[I, d]`, where `I` is the retained patch count and `d` is the stored encoder dimension.

Feature extraction outputs should be organized so that each manifest row can resolve its `feature_path`, for example:

```text
features/
  DATASET_NAME/
    ENCODER_NAME/
      pt_files/
        SLIDE_ID.pt
```

CLAM is third-party software and remains subject to its own license and usage terms. Model weights for PLIP, UNI, and CONCH are likewise subject to their respective access and license conditions.

## Analytical manifests

Training consumes a CSV manifest specified by `data.manifest` in the YAML configuration. Each pathology image corresponds to one row. Multiple images from one patient therefore appear as multiple rows with the same `patient_id`.

Classification manifests require at least:

| Column | Meaning |
| --- | --- |
| `slide_id` | Unique WSI or TMA-core image identifier. |
| `patient_id` | Patient identifier used for partitioning and patient-level aggregation. |
| `feature_path` | Path to the cached `[I, d]` PyTorch feature tensor. |
| `label` | Task-specific class label. |

Overall-survival manifests require at least:

| Column | Meaning |
| --- | --- |
| `slide_id` | Unique pathology image identifier. |
| `patient_id` | Patient identifier used for partitioning and aggregation. |
| `feature_path` | Path to the cached `[I, d]` PyTorch feature tensor. |
| `event_time` | Observed follow-up or event time in the task's stated time unit. |
| `event_observed` | `1` for an observed death and `0` for a censored observation. |

For survival experiments, discrete-time interval boundaries are derived only from the training partition and then applied unchanged to the corresponding held-out partition. Do not precompute interval boundaries from the full cohort.

Public analytical manifests should be constructed from the public data releases using the inclusion, exclusion, label, and outcome definitions reported in the manuscript. Local absolute feature paths should be replaced with paths appropriate for the user's installation.

## Fixed cross-validation assignments

The `split/` directory contains the fixed public-cohort assignments used in the study:

| Cohort and task | Assignment scheme |
| --- | --- |
| CAMELYON-16 diagnosis | Three repetitions of stratified three-fold cross-validation. Each WSI belongs to one held-out fold per repetition. CAMELYON-16 contains one WSI per patient in this analysis. |
| TCGA-BRCA IDC versus ILC subtype | Five patient-level folds. |
| TCGA-NSCLC LUAD versus LUSC subtype | Five patient-level folds. |
| TCGA-LUAD four-class pathologic stage | Five patient-level folds. |
| TCGA-LUSC four-class pathologic stage | Five patient-level folds. |
| TCGA-BLCA overall survival | Five patient-level folds. |
| TCGA-LUAD overall survival | Five patient-level folds. |
| TCGA-LUSC overall survival | Five patient-level folds. |

Each split CSV contains `ID` and `slice_count`. `ID` is the held-out patient identifier for TCGA and the held-out WSI identifier for CAMELYON-16. The training partition is the complement of those IDs in the corresponding analytical manifest. The loader verifies that no patient occurs in both partitions and, when `slice_count` is present, checks the expected image count against the manifest.

The same task-specific split files were used for DANTA-MIL, all comparator methods, and all frozen feature extractors.

## Training DANTA-MIL

Run commands from the repository root. Example commands for the included configurations are:

### CAMELYON-16 classification

```bash
python train.py \
  --config configs/camelyon16_classification.yaml \
  --repeat 0 \
  --fold 0
```

Use `--repeat` in `{0, 1, 2}` and `--fold` in `{0, 1, 2}`.

### TCGA-BRCA IDC-versus-ILC classification

```bash
python train.py \
  --config configs/classification.yaml \
  --fold 0
```

### TCGA-LUSC overall-survival prediction

```bash
python train.py \
  --config configs/survival.yaml \
  --fold 0
```

For the included TCGA configurations, use `--fold` in `{0, 1, 2, 3, 4}`. Update the manifest and feature-root paths in the selected YAML before execution. Set `model.input_dim` to 1024 for ResNet-50 or UNI features and to 512 for PLIP or CONCH features.

Use `--dry-run` to validate the manifest, fixed partition, feature dimensions, and model construction without starting training:

```bash
python train.py --config CONFIG_PATH --fold 0 --dry-run
```

The primary experiments used seed 2026, Adam with an initial learning rate of `2e-4`, weight decay `1e-5`, cosine annealing, a maximum of 200 epochs, and batch size one. Early-stopping patience was 30 epochs for CAMELYON-16 and 20 epochs for TCGA. The retained checkpoint maximizes patient-level validation ROC AUC for classification or patient-level validation C-index for survival.

## Outputs

Each run writes to:

```text
runs/EXPERIMENT_NAME/repeat_R/fold_F/TIMESTAMP/
```

The directory contains:

- `metrics.csv`: epoch-level training loss and held-out patient-level metrics.
- `best_model.pt`: best checkpoint with model weights, model configuration, full YAML configuration, label mapping when applicable, and fold-specific survival boundaries when applicable.

For patients represented by multiple pathology images, classification probability vectors or WSI-level risk scores are averaged arithmetically before patient-level metric calculation.

## Baseline configurations

The manuscript compares DANTA-MIL with AB-MIL, CLAM, DSMIL, TransMIL, DTFD-MIL, IBMIL, MHIM-MIL, HIGT, RRTMIL, and MambaMIL.

Complete per-method configuration and provenance information is reported in **Supplementary Table S1** accompanying the manuscript (`supplementary_table_1.csv`, Additional file 13). It records the official repository, executed revision or software version when available, pretrained weights, model-specific hyperparameters, search space, tuning budget, seed, trainable parameter count, training epochs, early-stopping and checkpoint-selection rules, hardware and software environment, timing and memory information, and official-versus-independent implementation provenance.

Third-party baseline source code is not redistributed in this repository. Users should obtain each implementation from the official repository identified in Supplementary Table S1 and comply with its license. All predictive comparisons used the fixed split files in this repository, the same cached encoder-specific feature files, and the common evaluation protocol described in the manuscript.

## Reproducing manuscript tables and figures

The final manuscript tables, figure files, and supplementary files are distributed with the article submission. A dedicated reproducibility update containing the analysis and plotting entry points used to regenerate the manuscript tables and figures from retained predictions and summary files is planned for this repository. The future release will document the mapping from each manuscript item to its input, command, and output.

## Checkpoints

Representative public-cohort DANTA-MIL checkpoints are not included in the initial repository snapshot. They are planned for a subsequent tagged repository release. The checkpoint release will identify the cohort, task, feature encoder, repeat/fold, model configuration, and SHA-256 checksum for each file.

No checkpoint release will include private SMUZH-NSCLC patient data or information that permits re-identification.

## Citation

If you use this code, please cite the DANTA-MIL manuscript. The final bibliographic entry will be added after publication.

The WSI preprocessing pipeline should also be credited to CLAM:

```bibtex
@article{lu2021data,
  title={Data-efficient and weakly supervised computational pathology on whole-slide images},
  author={Lu, Ming Y. and Williamson, Drew F. K. and Chen, Tiffany Y. and Chen, Richard J. and Barbieri, Matteo and Mahmood, Faisal},
  journal={Nature Biomedical Engineering},
  volume={5},
  pages={555--570},
  year={2021},
  doi={10.1038/s41551-020-00682-w}
}
```