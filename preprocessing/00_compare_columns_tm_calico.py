#!/usr/bin/env python3
"""
Compare obs and var column names between Tabula Muris and (standardized) Calico.
Report which cell type names match (same canonical name in both) and which are unique.
Writes results/exploration/column_comparison_tm_calico.txt and cell_type_overlap.txt.
"""

import sys
from pathlib import Path
import anndata

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config
from preprocessing.cell_type_mapping import apply_celltype_mapping

OUT_PATH = config.RESULTS_EXPLORATION / "column_comparison_tm_calico.txt"
OVERLAP_PATH = config.RESULTS_EXPLORATION / "cell_type_overlap.txt"


def main():
    path_tm = Path(config.TABULA_MURIS_PATH)
    path_calico = Path(config.CALICO_STANDARDIZED_PATH)
    if not path_calico.exists():
        path_calico = Path(config.CALICO_PATH)
        label_calico = "Calico (raw)"
    else:
        label_calico = "Calico (standardized)"
    if not path_tm.exists():
        print(f"TM not found: {path_tm}")
        return 1
    if not path_calico.exists():
        print(f"Calico not found: {path_calico}")
        return 1
    adata_tm = anndata.read_h5ad(path_tm)
    adata_calico = anndata.read_h5ad(path_calico)

    obs_tm = set(adata_tm.obs.columns)
    obs_calico = set(adata_calico.obs.columns)
    var_tm = set(adata_tm.var.columns)
    var_calico = set(adata_calico.var.columns)

    in_both_obs = obs_tm & obs_calico
    only_tm_obs = obs_tm - obs_calico
    only_calico_obs = obs_calico - obs_tm
    in_both_var = var_tm & var_calico
    only_tm_var = var_tm - var_calico
    only_calico_var = var_calico - var_tm

    lines = [
        "Column comparison: Tabula Muris vs " + label_calico,
        "=" * 60,
        "",
        "--- adata.obs columns ---",
        f"In BOTH ({len(in_both_obs)}): " + ", ".join(sorted(in_both_obs)) if in_both_obs else "In BOTH: (none)",
        "",
        f"Only in TM ({len(only_tm_obs)}): " + ", ".join(sorted(only_tm_obs)) if only_tm_obs else "Only in TM: (none)",
        "",
        f"Only in Calico ({len(only_calico_obs)}): " + ", ".join(sorted(only_calico_obs)) if only_calico_obs else "Only in Calico: (none)",
        "",
        "--- adata.var columns ---",
        f"In BOTH ({len(in_both_var)}): " + ", ".join(sorted(in_both_var)) if in_both_var else "In BOTH: (none)",
        "",
        f"Only in TM ({len(only_tm_var)}): " + ", ".join(sorted(only_tm_var)) if only_tm_var else "Only in TM: (none)",
        "",
        f"Only in Calico ({len(only_calico_var)}): " + ", ".join(sorted(only_calico_var)) if only_calico_var else "Only in Calico: (none)",
        "",
        "--- Harmonization columns (used in 03_preprocess_discovery) ---",
        "  age          -> age_months (numeric)",
        "  cell_ontology_class -> cell_type_raw",
        "  dataset_source (added)",
        "",
    ]
    for col, desc in [("age", "age_months"), ("cell_ontology_class", "cell_type_raw")]:
        in_tm = "yes" if col in adata_tm.obs.columns else "NO"
        in_cal = "yes" if col in adata_calico.obs.columns else "NO"
        lines.append(f"  {col}: TM={in_tm}, Calico={in_cal}  ({desc})")
    lines.append("")

    text = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(text, encoding="utf-8")
    print(text)
    print(f"Written to: {OUT_PATH}")

    # Cell type overlap: canonical names in TM vs Calico (after mapping)
    col_tm = "cell_ontology_class"
    col_cal = "cell_ontology_class" if "cell_ontology_class" in adata_calico.obs.columns else "cell_type"
    if col_tm not in adata_tm.obs.columns or col_cal not in adata_calico.obs.columns:
        print("Skipping cell type overlap (missing cell type column in one dataset).")
        return 0
    canonical_tm = set(apply_celltype_mapping(adata_tm.obs[col_tm]).dropna().unique())
    canonical_cal = set(apply_celltype_mapping(adata_calico.obs[col_cal]).dropna().unique())
    in_both = sorted(canonical_tm & canonical_cal)
    only_tm = sorted(canonical_tm - canonical_cal)
    only_cal = sorted(canonical_cal - canonical_tm)
    overlap_lines = [
        "Cell type names (canonical): Tabula Muris vs Calico (standardized)",
        "Same name in both = match; only in one = unmatched in that dataset.",
        "=" * 60,
        "",
        f"Matched (in both datasets) ({len(in_both)}):",
        "\n".join(f"  {t}" for t in in_both) if in_both else "  (none)",
        "",
        f"Only in TM ({len(only_tm)}):",
        "\n".join(f"  {t}" for t in only_tm) if only_tm else "  (none)",
        "",
        f"Only in Calico ({len(only_cal)}):",
        "\n".join(f"  {t}" for t in only_cal) if only_cal else "  (none)",
        "",
    ]
    overlap_text = "\n".join(overlap_lines)
    OVERLAP_PATH.write_text(overlap_text, encoding="utf-8")
    print(overlap_text)
    print(f"Written to: {OVERLAP_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
