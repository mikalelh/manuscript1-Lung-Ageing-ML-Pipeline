#!/usr/bin/env python3
"""
Discovery preprocessing: TM + standardized Calico → QC, harmonize metadata, concatenate,
then either scCobra or Harmony batch correction, UMAP, Leiden.
Uses TABULA_MURIS_PATH and CALICO_STANDARDIZED_PATH (run 02_standardize_calico_for_tm.py first).
Set config.USE_SCCOBRA = True for scCobra (default), False for Harmony.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scanpy as sc

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config
from preprocessing.cell_type_mapping import apply_celltype_mapping

FIG_DIR_QC = config.FIGURES_QC
FIG_DIR_DISCOVERY = config.FIGURES_DISCOVERY
OUT_DIR = config.OUTPUT_DIR
DATA_DIR = config.DATA_DIR
USE_SCCOBRA = getattr(config, "USE_SCCOBRA", False)
EMBED_KEY = getattr(config, "INTEGRATION_EMBEDDING_KEY", "X_pca_harmony")

# Age string -> integer months (TM and standardized Calico use 1m, 3m, 7m, 18m, 21m, 22m, 30m)
AGE_TO_MONTHS = {"1m": 1, "3m": 3, "7m": 7, "18m": 18, "21m": 21, "22m": 22, "30m": 30}


def _mark_mt_genes(adata):
    """Mark mitochondrial genes. Tabula Muris may use 'Mt-' (capital M); Calico often 'mt-'. Case-insensitive."""
    names = adata.var_names.astype(str)
    adata.var["mt"] = names.str.lower().str.startswith("mt-")
    n_mt = adata.var["mt"].sum()
    if n_mt == 0:
        # Fallback: some datasets use 'MT-' or gene IDs; try contains 'mt' at start (e.g. MT-CO1)
        adata.var["mt"] = names.str.upper().str.startswith("MT")
        n_mt = adata.var["mt"].sum()
    return int(n_mt)


def _qc_one(adata, name, fig_path):
    """STEP 1: Mark mt, calculate QC, filter cells/genes; save QC violins."""
    n_mt = _mark_mt_genes(adata)
    print(f"  {name}: {n_mt} mitochondrial genes marked")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    # Ensure standard names for violins (scanpy may use n_genes_by_counts or n_genes)
    if "n_genes_by_counts" not in adata.obs and "n_genes" in adata.obs:
        adata.obs["n_genes_by_counts"] = adata.obs["n_genes"]
    if "total_counts" not in adata.obs and "n_counts" in adata.obs:
        adata.obs["total_counts"] = adata.obs["n_counts"]
    # Violins before filter
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ["n_genes_by_counts", "total_counts", "pct_counts_mt"]):
        key_use = key if key in adata.obs.columns else ({"total_counts": "n_counts"}.get(key))
        if key_use and key_use in adata.obs.columns:
            sns.violinplot(y=adata.obs[key_use], ax=ax)
        ax.set_ylabel(key)
    plt.suptitle(f"{name} QC (before filter)")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    # Filter cells: min 200 genes, max 5000 genes, max 20% mt
    sc.pp.filter_cells(adata, min_genes=200)
    n_genes_col = "n_genes" if "n_genes" in adata.obs else "n_genes_by_counts"
    adata = adata[adata.obs[n_genes_col] <= 5000].copy()
    adata = adata[adata.obs["pct_counts_mt"] <= 20].copy()
    sc.pp.filter_genes(adata, min_cells=10)
    # Violins after filter
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, key in zip(axes, ["n_genes_by_counts", "total_counts", "pct_counts_mt"]):
        key_use = key if key in adata.obs.columns else "n_counts"
        if key_use in adata.obs.columns:
            sns.violinplot(y=adata.obs[key_use], ax=ax)
        ax.set_ylabel(key)
    plt.suptitle(f"{name} QC (after filter)")
    plt.tight_layout()
    plt.savefig(fig_path.parent / (fig_path.stem + "_after.png"), dpi=150, bbox_inches="tight")
    plt.close()
    return adata


def _normalize_one(adata):
    """STEP 2: Store raw counts (for LIANA etc.), normalize_total 1e4, log1p."""
    adata.raw = adata.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata


def _harmonize_metadata(adata, dataset_source):
    """STEP 3: age_months, cell_type_raw, dataset_source."""
    # age_months from 'age' (values like 1m, 3m, 7m, ...)
    if "age" in adata.obs.columns:
        age_str = adata.obs["age"].astype(str).str.strip().str.lower()
        adata.obs["age_months"] = age_str.map(AGE_TO_MONTHS).astype("Int64")
    # cell_type_raw: canonical names so TM and Calico labels match
    if "cell_ontology_class" in adata.obs.columns:
        adata.obs["cell_type_raw"] = apply_celltype_mapping(adata.obs["cell_ontology_class"])
    adata.obs["dataset_source"] = dataset_source
    return adata


def main():
    # Load: TM and standardized Calico
    path_tm = Path(config.TABULA_MURIS_PATH)
    path_calico = Path(config.CALICO_STANDARDIZED_PATH)
    if not path_tm.exists():
        print(f"Tabula Muris not found: {path_tm}")
        return 1
    if not path_calico.exists():
        print(f"Standardized Calico not found: {path_calico}. Run 02_standardize_calico_for_tm.py first.")
        return 1
    adata_tm = sc.read_h5ad(path_tm)
    adata_calico = sc.read_h5ad(path_calico)

    # STEP 1: Per-dataset QC
    print("STEP 1: Per-dataset QC...")
    adata_tm = _qc_one(adata_tm, "Tabula Muris", FIG_DIR_QC / "qc_tabula_muris.png")
    adata_calico = _qc_one(adata_calico, "Calico", FIG_DIR_QC / "qc_calico.png")
    print(f"  TM: {adata_tm.shape}, Calico: {adata_calico.shape}")

    # STEP 2: Harmonize metadata (before concat / scCobra)
    print("STEP 2: Harmonize metadata...")
    adata_tm = _harmonize_metadata(adata_tm, "tabula_muris")
    adata_calico = _harmonize_metadata(adata_calico, "calico")

    # STEP 3: Common genes
    print("STEP 3: Common genes...")
    common_genes = list(set(adata_tm.var_names) & set(adata_calico.var_names))
    adata_tm = adata_tm[:, common_genes].copy()
    adata_calico = adata_calico[:, common_genes].copy()

    use_sccobra = USE_SCCOBRA
    run_scCobra = None
    if USE_SCCOBRA:
        try:
            import scCobra
            if hasattr(scCobra, "function") and hasattr(scCobra.function, "scCobra"):
                run_scCobra = scCobra.function.scCobra
            if (run_scCobra is None or not callable(run_scCobra)) and hasattr(scCobra, "scCobra"):
                run_scCobra = getattr(scCobra, "scCobra")
        except ImportError:
            pass
        if run_scCobra is None or not callable(run_scCobra):
            print("scCobra not available; falling back to Harmony. To use scCobra: install from https://github.com/mcgilldinglab/scCobra")
            use_sccobra = False

    if use_sccobra and run_scCobra is not None:
        # STEP 4–6 (scCobra): save raw counts, run scCobra, get latent
        sccobra_dir = DATA_DIR / "sccobra_output"
        sccobra_dir.mkdir(parents=True, exist_ok=True)
        path_tm_tmp = sccobra_dir / "tm_for_sccobra.h5ad"
        path_calico_tmp = sccobra_dir / "calico_for_sccobra.h5ad"
        adata_tm.write_h5ad(path_tm_tmp)
        adata_calico.write_h5ad(path_calico_tmp)
        print("STEP 4: scCobra batch correction (raw counts → latent)...")
        adata_combined = run_scCobra(
            [str(path_tm_tmp), str(path_calico_tmp)],
            batch_categories=["tabula_muris", "calico"],
            batch_name="batch",
            min_features=200,
            min_cells=10,
            n_top_features=2000,
            outdir=str(sccobra_dir / "run"),
            ignore_umap=True,
            show=False,
            gpu=0,
        )
        # Use same embedding key as Harmony path for downstream (04, etc.)
        adata_combined.obsm[EMBED_KEY] = adata_combined.obsm["latent"]
        if "dataset_source" not in adata_combined.obs.columns:
            adata_combined.obs["dataset_source"] = adata_combined.obs["batch"].astype(str)
        print(f"  scCobra latent shape: {adata_combined.obsm[EMBED_KEY].shape}")
    else:
        # STEP 4–6 (Harmony): normalize, concat, HVG, PCA, Harmony
        print("STEP 4: Normalize and concatenate...")
        adata_tm = _normalize_one(adata_tm)
        adata_calico = _normalize_one(adata_calico)
        adata_combined = adata_tm.concatenate(adata_calico, batch_key="batch")
        print(f"  Combined shape: {adata_combined.shape}")
        print("  Cells per age_months:", adata_combined.obs["age_months"].value_counts().sort_index().to_dict())
        print("  Cells per dataset_source:", adata_combined.obs["dataset_source"].value_counts().to_dict())
        print("STEP 5: Highly variable genes...")
        sc.pp.highly_variable_genes(
            adata_combined, min_mean=0.0125, max_mean=3, min_disp=0.5, batch_key="batch"
        )
        n_hvg = adata_combined.var["highly_variable"].sum()
        print(f"  HVGs: {n_hvg}")
        adata_combined = adata_combined[:, adata_combined.var["highly_variable"]].copy()
        print("STEP 6: Harmony batch correction...")
        import harmonypy as hm
        sc.pp.scale(adata_combined, max_value=10)
        sc.tl.pca(adata_combined, svd_solver="arpack", n_comps=50)
        harmony_out = hm.run_harmony(adata_combined.obsm["X_pca"], adata_combined.obs, "batch")
        adata_combined.obsm[EMBED_KEY] = harmony_out.Z_corr.T

    # STEP 7: Neighbors, UMAP, Leiden
    print("STEP 7: Neighbors, UMAP, Leiden...")
    sc.pp.neighbors(adata_combined, use_rep=EMBED_KEY, n_neighbors=30)
    sc.tl.umap(adata_combined)
    sc.tl.leiden(adata_combined, resolution=0.5)

    # UMAP plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    for ax, color in zip(axes.ravel(), ["age_months", "cell_type_raw", "dataset_source", "leiden"]):
        if color in adata_combined.obs.columns:
            sc.pl.umap(adata_combined, color=color, ax=ax, show=False, size=2.2, legend_loc="on data" if color == "leiden" else "right margin")
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(FIG_DIR_DISCOVERY / "umap_discovery_combined.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {FIG_DIR_DISCOVERY / 'umap_discovery_combined.png'}")

    # Save combined object
    out_h5ad = DATA_DIR / "discovery_combined.h5ad"
    adata_combined.write_h5ad(out_h5ad)
    print(f"  Saved: {out_h5ad}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
