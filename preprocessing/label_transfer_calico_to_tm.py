#!/usr/bin/env python3
"""
Relabel Calico cells to Tabula Muris ontology using scVI + scANVI (label transfer).
Reference: https://docs.scvi-tools.org/en/1.0.0/tutorials/notebooks/tabula_muris.html

- TM = labeled reference (cell_ontology_class).
- Calico = query (treated as unlabeled); gets predicted TM-style labels.
- Output: standardized Calico with age 7m/22m and cell_ontology_class = scANVI predictions.
- Writes data/lung_calico_standardized.h5ad (Calico cells only, full genes).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config

# Age: Calico young/old -> 7m/22m (TM style)
AGE_MAP = {"young": "7m", "old": "22m"}
# scANVI unlabeled placeholder
UNLABELED_CATEGORY = "Unknown"
LABELS_KEY = "celltype_scanvi"
# Optional: subset to HVGs for faster training (set to None to use all genes)
N_TOP_GENES = 2000


def _ensure_counts(adata, layer_name="counts"):
    """Ensure we have integer counts for scVI (in X or layer)."""
    if layer_name in adata.layers:
        return
    x = adata.X
    if hasattr(x, "toarray"):
        x = x.toarray()
    if not np.issubdtype(x.dtype, np.integer) and np.all(x >= 0):
        x = np.rint(x).astype(np.float32)
    adata.layers[layer_name] = x
    return


def main():
    path_tm = Path(config.TABULA_MURIS_PATH)
    path_calico = Path(config.CALICO_PATH)
    out_path = config.CALICO_STANDARDIZED_PATH
    if not path_tm.exists():
        print(f"Tabula Muris not found: {path_tm}")
        return 1
    if not path_calico.exists():
        print(f"Calico not found: {path_calico}")
        return 1

    try:
        import scvi
    except ImportError:
        print("scvi-tools not installed. Install with: pip install scvi-tools")
        return 1

    print("Loading TM and Calico...")
    adata_tm = sc.read_h5ad(path_tm)
    adata_calico = sc.read_h5ad(path_calico)

    # Standardize Calico age and keep original labels for reference
    if "age" in adata_calico.obs.columns:
        age_vals = adata_calico.obs["age"].astype(str).str.strip().str.lower()
        adata_calico.obs["age"] = age_vals.replace(AGE_MAP).astype("category")
        print("Calico age standardized: young -> 7m, old -> 22m")
    if "cell_type" in adata_calico.obs.columns:
        adata_calico.obs["cell_type_original"] = adata_calico.obs["cell_type"].astype(str)

    n_calico = adata_calico.n_obs
    adata_tm.obs["dataset_source"] = "TM"
    adata_calico.obs["dataset_source"] = "Calico"
    # Labels for scANVI: TM = cell_ontology_class, Calico = Unknown
    adata_tm.obs[LABELS_KEY] = adata_tm.obs["cell_ontology_class"].astype(str)
    adata_calico.obs[LABELS_KEY] = UNLABELED_CATEGORY

    print("Concatenating (inner join on genes)...")
    adata = sc.concat([adata_tm, adata_calico], join="inner", label="batch_concat", keys=["TM", "Calico"])
    adata.obs["dataset_source"] = pd.Categorical(
        adata.obs["dataset_source"],
        categories=["TM", "Calico"],
    )
    # Ensure categorical labels with Unknown last (scANVI convention)
    all_tm_labels = list(adata_tm.obs["cell_ontology_class"].astype(str).unique())
    adata.obs[LABELS_KEY] = pd.Categorical(
        adata.obs[LABELS_KEY],
        categories=all_tm_labels + [UNLABELED_CATEGORY],
    )

    _ensure_counts(adata)
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()

    # HVG selection for faster/better integration (optional)
    if N_TOP_GENES and adata.n_vars > N_TOP_GENES:
        print(f"Selecting {N_TOP_GENES} HVGs...")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(
            adata,
            flavor="seurat_v3",
            n_top_genes=N_TOP_GENES,
            layer="counts",
            batch_key="dataset_source",
            subset=True,
        )
        # Restore counts in X for scVI (scVI uses layer="counts")
        adata.X = adata.layers["counts"].copy()

    print("Setting up scVI (batch = dataset_source)...")
    scvi.model.SCVI.setup_anndata(adata, layer="counts", batch_key="dataset_source")

    print("Training scVI...")
    vae = scvi.model.SCVI(adata, n_layers=2, n_latent=30)
    vae.train(max_epochs=400, early_stopping=True, early_stopping_patience=25)

    print("Setting up scANVI (label transfer from TM to Calico)...")
    lvae = scvi.model.SCANVI.from_scvi_model(
        vae,
        adata=adata,
        unlabeled_category=UNLABELED_CATEGORY,
        labels_key=LABELS_KEY,
    )
    lvae.train(max_epochs=20, n_samples_per_label=100)

    print("Predicting labels for Calico cells...")
    adata.obs["cell_ontology_class_transferred"] = lvae.predict(adata)

    # Calico cells are the last n_calico in concat order
    calico_mask = adata.obs["dataset_source"] == "Calico"
    pred_labels = adata.obs.loc[calico_mask, "cell_ontology_class_transferred"].values

    # Build standardized Calico: original Calico (full genes) with new labels
    calico_std = sc.read_h5ad(path_calico)
    if "age" in calico_std.obs.columns:
        age_vals = calico_std.obs["age"].astype(str).str.strip().str.lower()
        calico_std.obs["age"] = age_vals.replace(AGE_MAP).astype("category")
    calico_std.obs["cell_ontology_class"] = pd.Categorical(
        pred_labels,
        categories=sorted(pd.Series(pred_labels).unique()),
    )
    if "cell_type" in calico_std.obs.columns:
        calico_std.obs["cell_type_original"] = calico_std.obs["cell_type"].astype(str)
        calico_std.obs.drop(columns=["cell_type"], inplace=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    calico_std.write_h5ad(out_path)
    print(f"Saved standardized Calico: {out_path}")
    print("  Unique cell_ontology_class (TM-style):", list(calico_std.obs["cell_ontology_class"].cat.categories))
    return 0


if __name__ == "__main__":
    sys.exit(main())
