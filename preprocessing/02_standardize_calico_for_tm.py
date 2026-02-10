#!/usr/bin/env python3
"""
Standardize Calico so age and cell type match Tabula Muris.

- Age: young -> 7m, old -> 22m (TM style).
- Cell type: either (1) scANVI label transfer from TM (config.USE_LABEL_TRANSFER = True, recommended),
  or (2) dictionary mapping (cell_type_mapping.py) if scvi-tools unavailable or USE_LABEL_TRANSFER = False.
Writes data/lung_calico_standardized.h5ad.
"""

import sys
from pathlib import Path
import scanpy as sc

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config

# Age: Calico uses young/old; TM uses 1m, 3m, 18m, 21m, 30m
AGE_MAP = {"young": "7m", "old": "22m"}


def main():
    use_transfer = getattr(config, "USE_LABEL_TRANSFER", True)
    if use_transfer:
        try:
            import scvi  # noqa: F401
        except ImportError:
            print("scvi-tools not installed; using dictionary mapping for Calico cell types.")
            use_transfer = False
    if use_transfer:
        from preprocessing.label_transfer_calico_to_tm import main as label_transfer_main
        return label_transfer_main()

    # Fallback: dictionary-based alignment
    from preprocessing.cell_type_mapping import apply_celltype_mapping
    path = Path(config.CALICO_PATH)
    if not path.exists():
        print(f"Calico file not found: {path}")
        return 1
    adata = sc.read_h5ad(path)

    if "age" in adata.obs.columns:
        age_vals = adata.obs["age"].astype(str).str.strip().str.lower()
        adata.obs["age"] = age_vals.replace(AGE_MAP).astype("category")
        print("Age standardized: young -> 7m, old -> 22m")
        print("  Unique ages:", list(adata.obs["age"].cat.categories))
    else:
        print("Warning: no 'age' column found in Calico")

    if "cell_type" in adata.obs.columns:
        adata.obs["cell_ontology_class"] = apply_celltype_mapping(adata.obs["cell_type"]).astype("category")
        adata.obs["cell_type_original"] = adata.obs["cell_type"]
        adata.obs.drop(columns=["cell_type"], inplace=True)
        print("Cell type: dictionary mapping to TM-style names (cell_type_original kept)")
        print("  Unique cell_ontology_class:", list(adata.obs["cell_ontology_class"].cat.categories))
    else:
        print("Warning: no 'cell_type' column found in Calico")

    out_path = config.CALICO_STANDARDIZED_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
