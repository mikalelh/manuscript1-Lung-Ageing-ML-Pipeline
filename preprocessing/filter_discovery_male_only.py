#!/usr/bin/env python3
"""
Filter discovery_combined.h5ad to male mice only.
Keeps only cells where sex is 'male' (case-insensitive). Cells with missing sex (e.g. Calico) are dropped.
Overwrites data/discovery_combined.h5ad; run from pipeline root.
"""

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config

DISCOVERY_PATH = getattr(config, "DISCOVERY_COMBINED_PATH", config.DATA_DIR / "discovery_combined.h5ad")


def main():
    import anndata

    path = Path(DISCOVERY_PATH)
    if not path.exists():
        print(f"Not found: {path}")
        return 1

    adata = anndata.read_h5ad(path)
    n_before = adata.n_obs

    # Find sex column (TM uses 'sex')
    sex_col = None
    for col in ("sex", "Sex", "organism_sex", "gender"):
        if col in adata.obs.columns:
            sex_col = col
            break
    if sex_col is None:
        print("No sex column found in adata.obs. Columns:", list(adata.obs.columns))
        return 1

    # Normalize: strip, lowercase; keep only male
    sex_vals = adata.obs[sex_col].astype(str).str.strip().str.lower()
    male_mask = (sex_vals == "male") | (sex_vals == "m")
    n_male = male_mask.sum()
    n_dropped = n_before - n_male
    if n_male == 0:
        print("No male cells found. Unique values:", adata.obs[sex_col].astype(str).unique().tolist())
        return 1

    adata = adata[male_mask].copy()
    print(f"Filtered to male only: {n_before} -> {adata.n_obs} cells (dropped {n_dropped})")
    adata.write_h5ad(path)
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
