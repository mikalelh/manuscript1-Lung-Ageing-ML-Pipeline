#!/usr/bin/env python3
"""
Explore Tabula Muris, Lung Calico, and Validation h5ad datasets.
Do NOT modify any data. Write exploration summary and figures.
Run from msl_aging_pipeline or with Python path set to include msl_aging_pipeline.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

# Add pipeline root so we can import config
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config

OUTPUT_TXT = config.RESULTS_EXPLORATION / "dataset_exploration.txt"
OUTPUT_FIG = config.FIGURES_EXPLORATION / "dataset_overview.png"

DATASETS = [
    ("TM", config.TABULA_MURIS_PATH),
    ("Calico", config.CALICO_PATH),
    ("Validation", config.VALIDATION_PATH),
]


def _guess_age_column(obs):
    """Return name of column most likely to be age, or None."""
    candidates = [c for c in obs.columns if any(k in c.lower() for k in ("age", "time", "month", "development"))]
    if not candidates:
        return None
    # Prefer exact 'age' or 'age_group'
    for name in ("age", "age_group", "Age", "development_stage"):
        if name in obs.columns:
            return name
    return candidates[0]


def _guess_celltype_column(obs):
    """Return name of column most likely to be cell type, or None."""
    candidates = [c for c in obs.columns if any(k in c.lower() for k in ("cell_type", "celltype", "annotation", "ontology", "label", "cluster"))]
    if not candidates:
        return None
    for name in ("cell_ontology_class", "cell_type", "free_annotation", "annotation", "celltype"):
        if name in obs.columns:
            return name
    return candidates[0]


def _pct_cells_mt_detectable(adata):
    """Percentage of cells where at least one mt- gene has nonzero count."""
    var_mt = adata.var_names.str.startswith("mt-") | adata.var_names.str.startswith("MT-")
    if not var_mt.any():
        return float("nan")
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    mt_counts = np.asarray(X[:, var_mt].sum(axis=1)).ravel()
    return 100.0 * (mt_counts > 0).sum() / adata.n_obs


def explore_one(adata, name, f):
    """Write exploration summary for one dataset to file handle f."""
    f.write(f"\n{'='*60}\n{name}\n{'='*60}\n\n")
    f.write(f"shape: {adata.shape[0]} cells x {adata.shape[1]} genes\n\n")

    f.write("--- adata.obs columns (dtype, n_unique) ---\n")
    for col in adata.obs.columns:
        s = adata.obs[col]
        f.write(f"  {col}: dtype={s.dtype}, n_unique={s.nunique()}\n")
    f.write("\n")

    f.write("--- adata.var columns ---\n")
    for col in adata.var.columns:
        f.write(f"  {col}\n")
    f.write("\n")

    age_col = _guess_age_column(adata.obs)
    if age_col is not None:
        f.write(f"--- Age column: '{age_col}' ---\n")
        vc = adata.obs[age_col].value_counts().sort_index()
        for val, count in vc.items():
            f.write(f"  {val}: {count} cells\n")
        f.write("\n")
    else:
        f.write("--- Age column: not identified (check obs columns above) ---\n\n")

    ct_col = _guess_celltype_column(adata.obs)
    if ct_col is not None:
        f.write(f"--- Cell type column: '{ct_col}' ---\n")
        vc = adata.obs[ct_col].value_counts()
        for val, count in vc.items():
            f.write(f"  {val}: {count}\n")
        f.write("\n")
    else:
        f.write("--- Cell type column: not identified (check obs columns above) ---\n\n")

    X = adata.X
    sparse = hasattr(X, "toarray")
    dtype = getattr(X, "dtype", type(X).__name__)
    f.write(f"--- adata.X ---\n  matrix: {'sparse' if sparse else 'dense'}, dtype={dtype}\n\n")

    if "n_counts" in adata.obs.columns:
        mean_umi = float(adata.obs["n_counts"].mean())
        f.write(f"Mean UMI per cell: {mean_umi:.2f}\n")
    else:
        # Compute from X
        if hasattr(X, "toarray"):
            total = np.asarray(X.sum(axis=1)).ravel()
        else:
            total = np.asarray(X.sum(axis=1)).ravel()
        mean_umi = float(np.mean(total))
        f.write(f"Mean UMI per cell (from X): {mean_umi:.2f}\n")

    if "n_genes_by_counts" in adata.obs.columns:
        median_genes = float(adata.obs["n_genes_by_counts"].median())
    elif "n_genes" in adata.obs.columns:
        median_genes = float(adata.obs["n_genes"].median())
    else:
        n_genes_per_cell = np.asarray((X > 0).sum(axis=1)).ravel()
        median_genes = float(np.median(n_genes_per_cell))
    f.write(f"Median genes per cell: {median_genes:.1f}\n\n")

    pct_mt = _pct_cells_mt_detectable(adata)
    f.write(f"Percentage of cells with mt- gene detectable: {pct_mt:.2f}%\n\n")

    return age_col, ct_col


def get_n_counts_n_genes(adata):
    """Return (n_counts, n_genes) arrays for plotting without modifying adata."""
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    n_counts = np.asarray(X.sum(axis=1)).ravel()
    n_genes = np.asarray((X > 0).sum(axis=1)).ravel()
    if "n_counts" in adata.obs.columns:
        n_counts = adata.obs["n_counts"].values
    if "n_genes_by_counts" in adata.obs.columns:
        n_genes = adata.obs["n_genes_by_counts"].values
    elif "n_genes" in adata.obs.columns:
        n_genes = adata.obs["n_genes"].values
    return n_counts, n_genes


def main():
    print("Loading datasets (read-only)...")
    results = []
    adatas = {}

    with open(OUTPUT_TXT, "w") as f:
        f.write("Dataset exploration — Manuscript 1 pipeline\n")
        f.write("Do NOT modify data; exploration only.\n")

        for name, path in DATASETS:
            path = Path(path)
            if not path.exists():
                f.write(f"\n[{name}] SKIP: file not found: {path}\n")
                print(f"  {name}: file not found, skip")
                continue
            adata = sc.read_h5ad(path)
            adatas[name] = adata
            age_col, ct_col = explore_one(adata, name, f)
            results.append({"name": name, "path": str(path), "age_col": age_col, "celltype_col": ct_col, "adata": adata})

    # Build combined dataframe for violins (do not modify adata)
    dfs = []
    for name, adata in adatas.items():
        n_counts, n_genes = get_n_counts_n_genes(adata)
        dfs.append(pd.DataFrame({"n_counts": n_counts, "n_genes": n_genes, "dataset": name}))
    if not dfs:
        print("No datasets loaded. Exiting.")
        return 1
    combined = pd.concat(dfs, ignore_index=True)

    # Cell type column per dataset
    ct_cols = {}
    for name, adata in adatas.items():
        ct_cols[name] = _guess_celltype_column(adata.obs)

    n_ds = len(adatas)
    n_rows = 2 + n_ds  # violins + one bar chart per dataset
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 4 * n_rows))

    # Row 0: Violin n_counts
    sns.violinplot(data=combined, x="dataset", y="n_counts", ax=axes[0])
    axes[0].set_ylabel("n_counts")
    axes[0].set_title("UMI counts per dataset")

    # Row 1: Violin n_genes
    sns.violinplot(data=combined, x="dataset", y="n_genes", ax=axes[1])
    axes[1].set_ylabel("n_genes")
    axes[1].set_title("Genes per cell per dataset")

    # Rows 2+: Bar chart cell type distribution per dataset
    for i, (name, adata) in enumerate(adatas.items()):
        ax = axes[2 + i]
        ct_col = ct_cols.get(name)
        if ct_col is not None:
            vc = adata.obs[ct_col].value_counts().head(20)
            ax.barh(range(len(vc)), vc.values)
            ax.set_yticks(range(len(vc)))
            ax.set_yticklabels(vc.index, fontsize=8)
        ax.set_xlabel("Cell count")
        ax.set_title(f"Cell type distribution — {name}")

    plt.tight_layout()
    plt.savefig(OUTPUT_FIG, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved figure: {OUTPUT_FIG}")

    # Compatibility report
    print("\n" + "=" * 60)
    print("COMPATIBILITY REPORT")
    print("=" * 60)

    if "TM" in adatas and "Calico" in adatas:
        genes_tm = set(adatas["TM"].var_names)
        genes_calico = set(adatas["Calico"].var_names)
        inter = genes_tm & genes_calico
        print(f"Do TM and Calico share the same gene set? Intersection size: {len(inter)} (TM: {len(genes_tm)}, Calico: {len(genes_calico)})")
    else:
        print("TM and/or Calico not loaded; cannot compare gene sets.")

    # Gene names: symbols vs Ensembl
    for name, adata in adatas.items():
        sample = list(adata.var_names[:5])
        sample_str = ", ".join(sample)
        if any(s.startswith("ENSMUS") or s.startswith("ENSMUSG") for s in sample):
            kind = "likely Ensembl IDs"
        else:
            kind = "likely symbols (e.g. Cd68)"
        print(f"Gene names in {name}: {kind}. Sample: {sample_str}")

    # Normalized?
    for name, adata in adatas.items():
        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        max_val = np.max(X)
        already_norm = "yes" if max_val > 100 else "no (raw or log1p-like)"
        print(f"Dataset {name} already normalized (max > 100)? {already_norm} (max={max_val:.2f})")

    print("\nExact age column name per dataset:")
    for r in results:
        print(f"  {r['name']}: {r['age_col']}")

    print("\nExact cell type column name per dataset:")
    for r in results:
        print(f"  {r['name']}: {r['celltype_col']}")

    # Append compatibility report to text file
    with open(OUTPUT_TXT, "a") as f:
        f.write("\n" + "=" * 60 + "\nCOMPATIBILITY REPORT\n" + "=" * 60 + "\n")
        if "TM" in adatas and "Calico" in adatas:
            genes_tm = set(adatas["TM"].var_names)
            genes_calico = set(adatas["Calico"].var_names)
            inter = genes_tm & genes_calico
            f.write(f"TM vs Calico gene set intersection: {len(inter)} (TM: {len(genes_tm)}, Calico: {len(genes_calico)})\n")
        for name, adata in adatas.items():
            sample = list(adata.var_names[:5])
            kind = "likely Ensembl IDs" if any(s.startswith("ENSMUS") for s in sample) else "likely symbols"
            f.write(f"Gene names {name}: {kind}. Sample: {sample}\n")
        for name, adata in adatas.items():
            X = adata.X
            if hasattr(X, "toarray"):
                X = X.toarray()
            max_val = np.max(X)
            f.write(f"{name} normalized (max>100)? {max_val > 100} (max={max_val:.2f})\n")
        f.write("Age column per dataset: " + ", ".join(f"{r['name']}={r['age_col']}" for r in results) + "\n")
        f.write("Cell type column per dataset: " + ", ".join(f"{r['name']}={r['celltype_col']}" for r in results) + "\n")

    print(f"\nExploration summary written to: {OUTPUT_TXT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
