#!/usr/bin/env python3
"""
Batch correction performance assessment: metrics, interpretive report, and dashboard figure.
Loads discovery_combined.h5ad, reuses metrics from 04_harmonization_metrics, and produces:
  - results/batch_correction_assessment.txt   (narrative + metrics)
  - figures/batch_correction_assessment_dashboard.png  (multi-panel visual)
Run after 03_preprocess_discovery.py (and optionally 04_harmonization_metrics.py).
"""

import sys
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MPL = True
except Exception:
    HAS_MPL = False

RESULTS_DIR = config.RESULTS_HARMONIZATION
FIG_DIR = config.FIGURES_HARMONIZATION
DISCOVERY_PATH = config.DISCOVERY_COMBINED_PATH
OUT_REPORT = RESULTS_DIR / "batch_correction_assessment.txt"
OUT_DASHBOARD = FIG_DIR / "batch_correction_assessment_dashboard.png"


def _load_metrics_module():
    """Load 04_harmonization_metrics and return run_metrics."""
    path_04 = Path(__file__).parent / "04_harmonization_metrics.py"
    spec = importlib.util.spec_from_file_location("harmonization_metrics", path_04)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["harmonization_metrics"] = mod
    spec.loader.exec_module(mod)
    return mod.run_metrics


def _write_assessment_report(metrics, batch_col, out_path):
    """Write narrative assessment and metrics to out_path."""
    m = metrics
    sil_b = m["silhouette_batch"] if m["silhouette_batch"] is not None else float("nan")
    sil_c = m["silhouette_cell_type"] if m["silhouette_cell_type"] is not None else float("nan")
    ari = m["ARI"] if m.get("ARI") is not None else float("nan")
    nmi = m["NMI"] if m.get("NMI") is not None else float("nan")

    lines = [
        "Batch correction performance assessment",
        "=" * 50,
        "",
        "1. BATCH MIXING (technical variation removed)",
        "-" * 40,
        "  • Silhouette (batch):     {:.4f}  → lower is better (batch should not separate cells)".format(sil_b),
        "  • Graph connectivity:     {:.4f}  → fraction of cells in largest cross-batch component (higher = better mix)".format(
            m["graph_connectivity"]
        ),
        "  • Mixing score:           {:.4f}  → 1 - mean(same-batch k-NN fraction); higher = more mixed neighborhoods".format(
            m["mixing_score"]
        ),
        "  • k-NN same-batch mean:   {:.4f}  → lower = more cross-batch neighbors".format(
            m["kNN_same_batch_fraction_mean"]
        ),
        "",
        "2. BIOLOGY PRESERVATION (cell types still separable)",
        "-" * 40,
        "  • Silhouette (cell type): {:.4f}  → higher is better (cell types should still cluster)".format(sil_c),
        "  • ARI (Leiden vs labels): {:.4f}  → agreement of clustering with cell types (higher = better)".format(ari),
        "  • NMI (Leiden vs labels): {:.4f}  → mutual information (higher = better)".format(nmi),
        "  • Mean cluster entropy:   {:.4f}  → batch diversity within clusters (higher = more mixed)".format(
            m["mean_cluster_batch_entropy"] if m.get("mean_cluster_batch_entropy") is not None else float("nan")
        ),
        "",
        "3. SAMPLE",
        "-" * 40,
        "  n_cells = {}, n_batches = {}, n_cell_types = {}".format(
            m["n_cells"], m["n_batches"], m["n_cell_types"]
        ),
        "",
        "4. ASSESSMENT OF ALL STATS AND DEDUCTIONS",
        "=" * 50,
        "",
    ]
    # Per-metric deduction
    lines.append("  Batch silhouette ({:.4f}):".format(sil_b))
    if sil_b < 0.08:
        lines.append("    → Deduction: Batch identity is poorly predictive of position; technical batch effect has been successfully reduced.")
    elif sil_b < 0.2:
        lines.append("    → Deduction: Moderate batch structure remains; integration is acceptable but some batch-driven separation persists.")
    else:
        lines.append("    → Deduction: Strong batch structure; correction is insufficient or batches differ biologically.")
    lines.append("")

    lines.append("  Graph connectivity ({:.4f}):".format(m["graph_connectivity"]))
    if m["graph_connectivity"] > 0.8:
        lines.append("    → Deduction: Most cells are connected via cross-batch edges; batches are well mixed in local neighborhoods.")
    elif m["graph_connectivity"] > 0.5:
        lines.append("    → Deduction: Substantial cross-batch connectivity; integration is adequate.")
    else:
        lines.append("    → Deduction: Limited cross-batch connectivity; many cells have only same-batch neighbors.")
    lines.append("")

    lines.append("  Mixing score ({:.4f}):".format(m["mixing_score"]))
    if m["mixing_score"] > 0.4:
        lines.append("    → Deduction: Neighborhoods are well mixed; on average a good fraction of k-NN are from the other batch.")
    elif m["mixing_score"] > 0.2:
        lines.append("    → Deduction: Moderate mixing; roughly 20–40% of neighbors are from the other batch.")
    else:
        lines.append("    → Deduction: Low mixing; most neighbors are from the same batch.")
    lines.append("")

    lines.append("  Cell-type silhouette ({:.4f}):".format(sil_c))
    if sil_c > 0.1:
        lines.append("    → Deduction: Cell types remain well separated; biological signal has been preserved.")
    elif sil_c > 0.04:
        lines.append("    → Deduction: Cell-type structure is partially preserved; monitor for over-correction.")
    else:
        lines.append("    → Deduction: Weak cell-type separation; risk of over-correction or loss of biological structure.")
    lines.append("")

    lines.append("  ARI ({:.4f}) and NMI ({:.4f}):".format(ari, nmi))
    if ari > 0.5 and nmi > 0.7:
        lines.append("    → Deduction: Leiden clusters align well with annotated cell types; biology is retained in the embedding.")
    elif ari > 0.3 or nmi > 0.5:
        lines.append("    → Deduction: Clusters partially reflect cell types; some biological structure is captured.")
    else:
        lines.append("    → Deduction: Clusters do not strongly match cell-type labels; may reflect over-correction or annotation noise.")
    lines.append("")

    # Overall verdict
    lines.append("  OVERALL VERDICT:")
    sil_b_val = m["silhouette_batch"] or 0
    sil_c_val = m["silhouette_cell_type"] or 0
    if sil_b_val < 0.1 and m["graph_connectivity"] > 0.5:
        lines.append("    Batch mixing: GOOD – batches are well integrated.")
    elif sil_b_val < 0.3:
        lines.append("    Batch mixing: MODERATE – some batch structure remains.")
    else:
        lines.append("    Batch mixing: WEAK – batch still drives structure.")
    if sil_c_val > 0.08:
        lines.append("    Biology: PRESERVED – cell-type structure retained.")
    elif sil_c_val > 0.03:
        lines.append("    Biology: MODERATE – some cell-type separation preserved.")
    else:
        lines.append("    Biology: AT RISK – possible over-correction.")
    lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _no_top_right_spines(ax):
    """Remove top and right axis spines (border)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _no_umap_spines(ax):
    """Remove all axis spines (black outline) from UMAP-style plot."""
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(False)


def _build_dashboard(adata, metrics, cluster_df, same_frac, sil_per_cell, batch_col, cell_type_col):
    """Create one multi-panel dashboard figure."""
    sil_batch_pc, sil_ct_pc = sil_per_cell
    n_cells = adata.n_obs
    point_size = 2.2  # bigger cells in UMAP

    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.28)

    # 1) UMAP by batch
    ax1 = fig.add_subplot(gs[0, 0])
    if "X_umap" in adata.obsm and batch_col in adata.obs.columns:
        c = pd.Categorical(adata.obs[batch_col]).codes
        ax1.scatter(
            adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1],
            c=c, cmap="tab10", s=point_size, alpha=0.7,
        )
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    ax1.set_title("UMAP by batch (good: mixed colors)")
    ax1.set_xticks([])
    ax1.set_yticks([])
    _no_umap_spines(ax1)

    # 2) UMAP by cell type
    ax2 = fig.add_subplot(gs[0, 1])
    if "X_umap" in adata.obsm and cell_type_col in adata.obs.columns:
        c = pd.Categorical(adata.obs[cell_type_col]).codes
        ax2.scatter(
            adata.obsm["X_umap"][:, 0], adata.obsm["X_umap"][:, 1],
            c=c, cmap="tab20", s=point_size, alpha=0.7,
        )
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    ax2.set_title("UMAP by cell type (good: distinct clusters)")
    ax2.set_xticks([])
    ax2.set_yticks([])
    _no_umap_spines(ax2)

    # 3) Metrics bar chart
    ax3 = fig.add_subplot(gs[1, 0])
    names = ["Batch ASW\n(low=good)", "CellType ASW\n(high=good)", "Graph conn.\n(high=good)", "Mixing score\n(high=good)"]
    vals = [
        metrics["silhouette_batch"] or 0,
        metrics["silhouette_cell_type"] or 0,
        metrics["graph_connectivity"],
        metrics["mixing_score"],
    ]
    colors = ["C0", "C1", "C2", "C3"]
    if metrics.get("ARI") is not None:
        names.extend(["ARI", "NMI"])
        vals.extend([metrics["ARI"], metrics["NMI"]])
        colors.extend(["C4", "C5"])
    ax3.barh(range(len(names)), vals, color=colors)
    ax3.set_yticks(range(len(names)))
    ax3.set_yticklabels(names, fontsize=9)
    ax3.set_xlabel("Score")
    ax3.set_title("Batch correction metrics")
    ax3.axvline(0, color="gray", linewidth=0.5)
    _no_top_right_spines(ax3)

    # 4) Same-batch fraction distribution (k-NN mixing)
    ax4 = fig.add_subplot(gs[1, 1])
    adata.obs["_same_frac"] = same_frac
    for batch_name in adata.obs[batch_col].unique():
        mask = adata.obs[batch_col] == batch_name
        v = adata.obs.loc[mask, "_same_frac"].dropna()
        if len(v) > 0:
            sns.kdeplot(v, ax=ax4, label=batch_name)
    ax4.set_xlabel("Fraction of k-NN from same batch")
    ax4.set_ylabel("Density")
    ax4.set_title("Local mixing (left-shifted = better)")
    ax4.legend(fontsize=8)
    _no_top_right_spines(ax4)

    # 5) Silhouette distributions
    ax5 = fig.add_subplot(gs[2, 0])
    if sil_batch_pc is not None:
        ax5.hist(sil_batch_pc, bins=40, alpha=0.7, color="C0", label="Batch", density=True)
    if sil_ct_pc is not None:
        ax5.hist(sil_ct_pc, bins=40, alpha=0.7, color="C1", label="Cell type", density=True)
    ax5.set_xlabel("Silhouette coefficient")
    ax5.set_ylabel("Density")
    ax5.set_title("Batch (low) vs cell type (high)")
    ax5.legend(fontsize=8)
    _no_top_right_spines(ax5)

    # 6) Cluster batch composition (stacked bar)
    ax6 = fig.add_subplot(gs[2, 1])
    if cluster_df is not None:
        plot_df = cluster_df.drop(columns=["n_cells", "entropy"], errors="ignore")
        if not plot_df.empty and len(plot_df) <= 25:
            plot_df.plot(kind="bar", stacked=True, ax=ax6, legend=True, width=0.8)
        elif len(plot_df) > 25:
            # Show first 25 clusters
            plot_df.head(25).plot(kind="bar", stacked=True, ax=ax6, legend=True, width=0.8)
    ax6.set_xlabel("Leiden cluster")
    ax6.set_ylabel("Fraction")
    ax6.set_title("Batch composition per cluster")
    ax6.tick_params(axis="x", rotation=45)
    plt.setp(ax6.get_xticklabels(), ha="right", fontsize=7)
    _no_top_right_spines(ax6)

    fig.suptitle("Batch correction performance assessment (n = {:,} cells)".format(n_cells), fontsize=12, y=1.02)
    return fig


def main():
    if not DISCOVERY_PATH.exists():
        print(f"Discovery object not found: {DISCOVERY_PATH}. Run 03_preprocess_discovery.py first.")
        return 1

    import anndata
    adata = anndata.read_h5ad(DISCOVERY_PATH)
    batch_col = "dataset_source" if "dataset_source" in adata.obs.columns else "batch"
    cell_type_col = "cell_type_raw" if "cell_type_raw" in adata.obs.columns else "cell_ontology_class"

    run_metrics = _load_metrics_module()
    metrics, cluster_df, same_frac, sil_per_cell = run_metrics(
        adata, batch_col=batch_col, cell_type_col=cell_type_col
    )

    # Report
    _write_assessment_report(metrics, batch_col, OUT_REPORT)
    print(OUT_REPORT.read_text(encoding="utf-8"))
    print(f"Written: {OUT_REPORT}")

    # Dashboard figure
    if HAS_MPL:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig = _build_dashboard(
            adata, metrics, cluster_df, same_frac, sil_per_cell,
            batch_col=batch_col, cell_type_col=cell_type_col,
        )
        fig.savefig(OUT_DASHBOARD, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {OUT_DASHBOARD}")
    else:
        print("Matplotlib not available; skipping dashboard figure.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
