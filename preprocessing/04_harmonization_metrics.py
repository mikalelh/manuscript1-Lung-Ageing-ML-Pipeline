#!/usr/bin/env python3
"""
Harmonization quality metrics for discovery_combined (TM + Calico after batch correction).
Reports: ARI, NMI, silhouette (batch vs cell type), graph connectivity, k-NN mixing,
per-cluster composition. Supports both Harmony (X_pca_harmony) and scCobra (latent).
Run after 03_preprocess_discovery.py. Writes results/harmonization_metrics.txt and figures.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config

# Optional: matplotlib for figures (skip if not available)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    HAS_MPL = False

RESULTS_DIR = config.RESULTS_HARMONIZATION
FIG_DIR = config.FIGURES_HARMONIZATION
DISCOVERY_PATH = config.DISCOVERY_COMBINED_PATH
OUT_TXT = RESULTS_DIR / "harmonization_metrics.txt"
OUT_FIG_MIXING = FIG_DIR / "harmonization_mixing_dist.png"
OUT_FIG_CLUSTERS = FIG_DIR / "harmonization_cluster_batch.png"
OUT_FIG_METRICS_BAR = FIG_DIR / "harmonization_metrics_summary.png"
OUT_FIG_UMAP_BATCH = FIG_DIR / "harmonization_umap_batch.png"
OUT_FIG_UMAP_CT = FIG_DIR / "harmonization_umap_celltype.png"
OUT_FIG_SILHOUETTE = FIG_DIR / "harmonization_silhouette_dist.png"


def _no_top_right_spines(ax):
    """Remove top and right axis spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _no_umap_spines(ax):
    """Remove all axis spines (black outline) from UMAP plot."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _embedding_key(adata):
    """Which obsm key to use (scCobra uses 'latent'; 03 stores as X_pca_harmony for compatibility)."""
    for key in (getattr(config, "INTEGRATION_EMBEDDING_KEY", None), "X_pca_harmony", "latent"):
        if key and key in adata.obsm:
            return key
    return None


def _get_neighbor_indices(adata, k=None):
    """From adata.obsp['connectivities'], return (n_cells, n_neighbors) array of neighbor indices."""
    conn = adata.obsp["connectivities"]
    n = adata.n_obs
    if k is None:
        k = conn.getnnz(axis=1).max()
    out = np.full((n, k), -1, dtype=np.int64)
    for i in range(n):
        row = conn[i]
        nz = row.indices[row.data.argsort()[::-1][:k]]
        out[i, : len(nz)] = nz
    return out


def _same_batch_fraction(adata, batch_col, neighbor_idx):
    """Per-cell fraction of k-NN that are from the same batch."""
    batch = adata.obs[batch_col].values
    n_cells, n_nbr = neighbor_idx.shape
    same = np.zeros(n_cells, dtype=np.float64)
    for i in range(n_cells):
        nbrs = neighbor_idx[i]
        nbrs = nbrs[nbrs >= 0]
        if len(nbrs) == 0:
            same[i] = np.nan
            continue
        same[i] = (batch[nbrs] == batch[i]).mean()
    return same


def _graph_connectivity(adata, batch_col, neighbor_idx):
    """Size of largest connected component when keeping only cross-batch edges (scIB-style). Higher = better."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    n = adata.n_obs
    batch = adata.obs[batch_col].values
    # Build adjacency with only cross-batch edges (symmetric)
    row, col = [], []
    for i in range(n):
        for j in neighbor_idx[i]:
            if j >= 0 and batch[i] != batch[j]:
                row.append(i)
                col.append(j)
    if not row:
        return 0.0
    adj = csr_matrix((np.ones(len(row)), (row, col)), shape=(n, n))
    adj = adj + adj.T  # symmetric
    n_comp, labels = connected_components(csgraph=adj, directed=False)
    sizes = np.bincount(labels)
    return float(sizes.max() / n) if n > 0 else 0.0


def _entropy(p):
    """Entropy of a 2-element distribution (p, 1-p)."""
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return -np.sum(p * np.log2(p))


def run_metrics(adata, batch_col="dataset_source", cell_type_col="cell_type_raw", cluster_col="leiden"):
    """Compute all harmonization metrics. Returns dict of metrics and DataFrames."""
    from sklearn.metrics import (
        silhouette_score,
        silhouette_samples,
        adjusted_rand_score,
        normalized_mutual_info_score,
    )

    emb_key = _embedding_key(adata)
    if emb_key is None:
        raise ValueError("No integration embedding found in adata.obsm (expect X_pca_harmony or latent)")
    X = adata.obsm[emb_key]
    batch = adata.obs[batch_col].astype(str)
    cell_type = adata.obs[cell_type_col].astype(str)
    n_batches = batch.nunique()
    n_celltypes = cell_type.nunique()

    batch_cat = pd.Categorical(batch)
    celltype_cat = pd.Categorical(cell_type)
    batch_labels = batch_cat.codes
    celltype_labels = celltype_cat.codes

    metrics = {}

    # 1) Silhouette: batch (low = good), cell type (high = good)
    if n_batches >= 2 and np.all(np.bincount(batch_labels) > 1):
        sil_batch = silhouette_score(X, batch_labels, metric="euclidean", sample_size=min(5000, adata.n_obs))
        metrics["silhouette_batch"] = float(sil_batch)
    else:
        metrics["silhouette_batch"] = None
    if n_celltypes >= 2 and np.all(np.bincount(celltype_labels) > 1):
        sil_celltype = silhouette_score(X, celltype_labels, metric="euclidean", sample_size=min(5000, adata.n_obs))
        metrics["silhouette_cell_type"] = float(sil_celltype)
    else:
        metrics["silhouette_cell_type"] = None

    # 2) ARI and NMI (Leiden vs cell_type_raw) – biology preservation
    if cluster_col in adata.obs.columns:
        leiden_labels = pd.Categorical(adata.obs[cluster_col]).codes
        metrics["ARI"] = float(adjusted_rand_score(celltype_labels, leiden_labels))
        metrics["NMI"] = float(normalized_mutual_info_score(celltype_labels, leiden_labels))
    else:
        metrics["ARI"] = None
        metrics["NMI"] = None

    # 3) Graph connectivity (cross-batch k-NN graph)
    neighbor_idx = _get_neighbor_indices(adata)
    metrics["graph_connectivity"] = float(_graph_connectivity(adata, batch_col, neighbor_idx))

    # 4) k-NN batch mixing
    same_frac = _same_batch_fraction(adata, batch_col, neighbor_idx)
    valid = ~np.isnan(same_frac)
    same_frac_valid = same_frac[valid]
    metrics["kNN_same_batch_fraction_mean"] = float(np.mean(same_frac_valid))
    metrics["kNN_same_batch_fraction_median"] = float(np.median(same_frac_valid))
    metrics["mixing_score"] = float(1.0 - np.mean(same_frac_valid))
    metrics["n_cells"] = int(adata.n_obs)
    metrics["n_batches"] = int(n_batches)
    metrics["n_cell_types"] = int(n_celltypes)

    # 5) Per-cluster batch composition
    if cluster_col in adata.obs.columns:
        comp = adata.obs.groupby(cluster_col)[batch_col].value_counts(normalize=True).unstack(fill_value=0)
        counts = adata.obs.groupby(cluster_col).size()
        entropies = []
        for cl in comp.index:
            p = comp.loc[cl].values
            p = p[p > 0]
            entropies.append(_entropy(p))
        comp["n_cells"] = counts
        comp["entropy"] = entropies
        metrics["mean_cluster_batch_entropy"] = float(np.mean(entropies))
        cluster_df = comp
    else:
        cluster_df = None
        metrics["mean_cluster_batch_entropy"] = None

    # 6) Per-cell silhouette for plots (subsample if huge)
    sample_size = min(3000, adata.n_obs)
    if sample_size < adata.n_obs:
        rng = np.random.default_rng(42)
        idx = rng.choice(adata.n_obs, sample_size, replace=False)
        X_s = X[idx]
        batch_s = batch_labels[idx]
        celltype_s = celltype_labels[idx]
    else:
        idx = np.arange(adata.n_obs)
        X_s, batch_s, celltype_s = X, batch_labels, celltype_labels
    sil_batch_per_cell = silhouette_samples(X_s, batch_s, metric="euclidean") if n_batches >= 2 else None
    sil_celltype_per_cell = silhouette_samples(X_s, celltype_s, metric="euclidean") if n_celltypes >= 2 else None

    return metrics, cluster_df, same_frac, (sil_batch_per_cell, sil_celltype_per_cell)


def main():
    if not DISCOVERY_PATH.exists():
        print(f"Discovery object not found: {DISCOVERY_PATH}. Run 03_preprocess_discovery.py first.")
        return 1
    import anndata
    adata = anndata.read_h5ad(DISCOVERY_PATH)
    batch_col = "dataset_source" if "dataset_source" in adata.obs.columns else "batch"
    cell_type_col = "cell_type_raw" if "cell_type_raw" in adata.obs.columns else "cell_ontology_class"
    metrics, cluster_df, same_frac, sil_per_cell = run_metrics(adata, batch_col=batch_col, cell_type_col=cell_type_col)

    # Report text
    lines = [
        "Harmonization quality metrics (discovery_combined)",
        "=" * 50,
        "",
        "Batch mixing (lower = better mixing):",
        f"  silhouette_batch       = {metrics['silhouette_batch']}",
        f"  graph_connectivity    = {metrics['graph_connectivity']:.4f}  (LCC of cross-batch k-NN graph)",
        "",
        "Biology preservation (higher = better):",
        f"  silhouette_cell_type  = {metrics['silhouette_cell_type']}",
        f"  ARI (Leiden vs cell_type) = {metrics.get('ARI')}",
        f"  NMI (Leiden vs cell_type) = {metrics.get('NMI')}",
        "",
        "k-NN local mixing (higher mixing_score = more mixed):",
        f"  kNN_same_batch_fraction_mean  = {metrics['kNN_same_batch_fraction_mean']:.4f}",
        f"  kNN_same_batch_fraction_median = {metrics['kNN_same_batch_fraction_median']:.4f}",
        f"  mixing_score (1 - mean same_batch) = {metrics['mixing_score']:.4f}",
        "",
        f"  n_cells = {metrics['n_cells']}, n_batches = {metrics['n_batches']}, n_cell_types = {metrics['n_cell_types']}",
        "",
    ]
    if metrics.get("mean_cluster_batch_entropy") is not None:
        lines.append(f"Per-cluster batch diversity: mean entropy = {metrics['mean_cluster_batch_entropy']:.4f}")
        lines.append("")
    if cluster_df is not None:
        lines.append("Per-cluster batch composition (fractions and n_cells):")
        lines.append(cluster_df.to_string())
        lines.append("")

    text = "\n".join(lines)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_TXT.write_text(text, encoding="utf-8")
    print(text)
    print(f"Written: {OUT_TXT}")

    # Figures
    if HAS_MPL:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        adata.obs["_same_batch_frac"] = same_frac

        # 1) Distribution of same-batch fraction by dataset
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        for batch_name in adata.obs[batch_col].unique():
            mask = adata.obs[batch_col] == batch_name
            vals = adata.obs.loc[mask, "_same_batch_frac"].dropna()
            sns.kdeplot(vals, ax=ax, label=batch_name)
        ax.set_xlabel("Fraction of k-NN from same batch")
        ax.set_ylabel("Density")
        ax.set_title("Local batch mixing (lower = better mixed)")
        ax.legend()
        _no_top_right_spines(ax)
        plt.tight_layout()
        plt.savefig(OUT_FIG_MIXING, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_FIG_MIXING}")

        # 2) Per-cluster batch composition bar chart
        if cluster_df is not None:
            plot_df = cluster_df.drop(columns=["n_cells", "entropy"], errors="ignore")
            if not plot_df.empty:
                fig, ax = plt.subplots(1, 1, figsize=(max(8, len(plot_df) * 0.4), 5))
                plot_df.plot(kind="bar", stacked=True, ax=ax)
                ax.set_xlabel("Leiden cluster")
                ax.set_ylabel("Fraction of cells")
                ax.set_title("Batch composition per cluster")
                ax.legend(title=batch_col)
                ax.tick_params(axis="x", rotation=45)
                plt.setp(ax.get_xticklabels(), ha="right")
                _no_top_right_spines(ax)
                plt.tight_layout()
                plt.savefig(OUT_FIG_CLUSTERS, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved: {OUT_FIG_CLUSTERS}")

        # 3) Metrics summary bar chart (batch vs biology)
        plot_metrics = [
            ("Batch ASW\n(lower=better)", metrics["silhouette_batch"], "C0"),
            ("Cell type ASW\n(higher=better)", metrics["silhouette_cell_type"], "C1"),
            ("Graph connectivity\n(higher=better)", metrics["graph_connectivity"], "C2"),
            ("Mixing score\n(higher=better)", metrics["mixing_score"], "C3"),
        ]
        if metrics.get("ARI") is not None:
            plot_metrics.extend([
                ("ARI\n(higher=better)", metrics["ARI"], "C4"),
                ("NMI\n(higher=better)", metrics["NMI"], "C5"),
            ])
        names = [m[0] for m in plot_metrics]
        values = [m[1] if m[1] is not None else 0 for m in plot_metrics]
        colors = [m[2] for m in plot_metrics]
        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        bars = ax.bar(range(len(names)), values, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Harmonization metrics summary")
        ax.axhline(0, color="gray", linewidth=0.5)
        _no_top_right_spines(ax)
        plt.tight_layout()
        plt.savefig(OUT_FIG_METRICS_BAR, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {OUT_FIG_METRICS_BAR}")

        # 4) Silhouette distribution (batch vs cell type)
        sil_batch_pc, sil_ct_pc = sil_per_cell
        if sil_batch_pc is not None or sil_ct_pc is not None:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            if sil_batch_pc is not None:
                axes[0].hist(sil_batch_pc, bins=50, color="C0", edgecolor="white")
                axes[0].set_xlabel("Silhouette (batch)")
                axes[0].set_ylabel("Cells")
                axes[0].set_title("Batch (lower = better mixed)")
            if sil_ct_pc is not None:
                axes[1].hist(sil_ct_pc, bins=50, color="C1", edgecolor="white")
                axes[1].set_xlabel("Silhouette (cell type)")
                axes[1].set_ylabel("Cells")
                axes[1].set_title("Cell type (higher = better separated)")
            for ax in axes:
                _no_top_right_spines(ax)
            plt.tight_layout()
            plt.savefig(OUT_FIG_SILHOUETTE, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"Saved: {OUT_FIG_SILHOUETTE}")

        # 5) UMAP by batch and by cell type (if X_umap exists)
        if "X_umap" in adata.obsm:
            for color_key, fpath in [
                (batch_col, OUT_FIG_UMAP_BATCH),
                (cell_type_col, OUT_FIG_UMAP_CT),
            ]:
                if color_key not in adata.obs.columns:
                    continue
                fig, ax = plt.subplots(1, 1, figsize=(7, 5))
                ax.scatter(
                    adata.obsm["X_umap"][:, 0],
                    adata.obsm["X_umap"][:, 1],
                    c=pd.Categorical(adata.obs[color_key]).codes,
                    cmap="tab20" if adata.obs[color_key].nunique() > 10 else "tab10",
                    s=2.2,
                    alpha=0.7,
                )
                ax.set_xlabel("UMAP 1")
                ax.set_ylabel("UMAP 2")
                ax.set_title(f"UMAP colored by {color_key}")
                _no_umap_spines(ax)
                plt.tight_layout()
                plt.savefig(fpath, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved: {fpath}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
