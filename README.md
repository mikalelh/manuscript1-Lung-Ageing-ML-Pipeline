# Ageing in the Lung — Machine Learning Pipeline

Single-cell preprocessing and **machine learning** for **Tabula Muris (TM) + Calico lung** data: exploration, standardization, QC, batch correction, harmonization, and LSTM-based age prediction.

## Overview

- **Data:** Tabula Muris lung (`Lung.h5ad`) and Calico lung (`lung_calico.h5ad`).
- **Goal:** One combined, batch-corrected dataset with aligned metadata (age, cell types) for downstream aging analyses.
- **Cell type alignment:** Calico is relabeled to **Tabula Muris ontology** (recommended: scANVI label transfer; fallback: dictionary mapping).

## Configuration

Edit **`config.py`** for paths and options:

| Option | Default | Description |
|--------|---------|-------------|
| `TABULA_MURIS_PATH` | (see config) | Path to TM lung `.h5ad` |
| `CALICO_PATH` | (see config) | Path to raw Calico lung `.h5ad` |
| `USE_LABEL_TRANSFER` | `True` | If `True`, step 02 uses **scVI + scANVI** to transfer TM labels to Calico. If `False` or scvi-tools is missing, uses dictionary mapping in `preprocessing/cell_type_mapping.py`. |
| `USE_SCCOBRA` | `True` | Batch correction: scCobra if available, else Harmony. |
| `CALICO_STANDARDIZED_PATH` | `data/lung_calico_standardized.h5ad` | Output of step 02. |
| `DISCOVERY_COMBINED_PATH` | `data/discovery_combined.h5ad` | Output of step 03. |

## Pipeline steps

Run from the **`msl_aging_pipeline/`** directory.

| Step | Script | What it does |
|------|--------|----------------|
| **00** | `preprocessing/00_compare_columns_tm_calico.py` | Compare obs/var columns between TM and Calico; write **cell type overlap** (matched / TM-only / Calico-only) to `results/exploration/`. |
| **01** | `preprocessing/01_explore_datasets.py` | Dataset exploration (shapes, columns, age/cell-type counts); writes `results/exploration/`, `figures/exploration/`. |
| **02** | `preprocessing/02_standardize_calico_for_tm.py` | **Standardize Calico:** age → 7m/22m; **cell types → TM ontology** (scANVI label transfer if `USE_LABEL_TRANSFER=True` and scvi-tools installed, else dictionary mapping). Writes `data/lung_calico_standardized.h5ad`. |
| **03** | `preprocessing/03_preprocess_discovery.py` | QC (per dataset), normalize, concatenate TM + standardized Calico, batch correct (scCobra or Harmony), UMAP, Leiden. Writes `data/discovery_combined.h5ad`, figures in `figures/qc/`, `figures/discovery/`. |
| **04** | `preprocessing/04_harmonization_metrics.py` | Harmonization metrics (ARI, NMI, silhouette, graph connectivity, k-NN mixing, etc.) on `discovery_combined.h5ad`. Writes `results/harmonization/`, `figures/harmonization/`. |
| **05** | `preprocessing/05_assess_batch_correction.py` | Batch correction assessment report and dashboard from 04 metrics. Writes `results/harmonization/`, `figures/harmonization/`. |

One-shot run:

```bash
./run_preprocessing_pipeline.sh
```

Or run scripts in order (00 → 01 → 02 → 03 → 04 → 05).

## Cell type alignment (step 02)

- **Label transfer (default, `USE_LABEL_TRANSFER=True`):**  
  Uses [scvi-tools (scVI + scANVI)](https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/tabula_muris.html). TM = labeled reference, Calico = query; scANVI predicts TM-style `cell_ontology_class` for each Calico cell. Install: `pip install scvi-tools`.

- **Dictionary mapping (fallback):**  
  If scvi-tools is not installed or `USE_LABEL_TRANSFER=False`, Calico labels are mapped via `preprocessing/cell_type_mapping.py` (e.g. "natural killer cell" → "NK cell", "lung endothelial cell" → "vein endothelial cell"). Original Calico labels are kept in `cell_type_original`.

Downstream steps (03–05) use `cell_ontology_class` / `cell_type_raw`; both modes produce a single, consistent cell type column for the combined dataset.

## Age prediction models (optional)

After 00–05, you can train **cell-type-specific models** to predict age from gene expression:

```bash
python models/lstm_aging_model.py
```

- **Input:** `data/discovery_combined.h5ad` (expects `cell_type_raw`, `age_months`).
- **Eligibility:** Cell types with ≥100 cells and ≥4 age timepoints.
- **Outputs:** Best checkpoint per cell type in `results/models/`, summary `results/lstm_training_summary.csv`, loss curves in `figures/`.
- **Dependencies:** torch, scanpy, sklearn.

### Model choice: LSTM vs MLP

| Context | Model used | Checkpoint name |
|--------|------------|------------------|
| **GPU** | LSTM (sequence + self-attention over genes) | `{cell_type}_lstm_best.pt` |
| **CPU, default** | MLP (flat gene vector → dense layers) | `{cell_type}_mlp_best.pt` |
| **CPU, `USE_MLP_ON_CPU = False`** | LSTM (slower; genes capped to `MAX_GENES_LSTM_CPU`) | `{cell_type}_lstm_best.pt` |

The **MLP is a different architecture** from the LSTM: it does not treat genes as a sequence and has no attention, so predictions and saved weights are not the same. Use the MLP on CPU for **faster, approximate** age prediction. If you need the **LSTM model** (e.g. for attention analysis or to match a method that uses recurrence), run on a GPU or set `USE_MLP_ON_CPU = False` in `models/lstm_aging_model.py` and accept slower CPU training.

## Output layout

- **`data/`** — Standardized Calico (02), discovery combined (03).
- **`results/`** — Text outputs, training summary; **`results/models/`** — Model checkpoints (LSTM or MLP per cell type). See **`results/README.md`**.
- **`figures/`** — Plots, loss curves. See **`figures/README.md`**.

## Dependencies

- Python 3.8+
- scanpy, anndata, pandas, numpy, matplotlib, seaborn, scipy, scikit-learn
- Optional: **scvi-tools** (for label transfer in 02), **scCobra** (for batch correction in 03)
