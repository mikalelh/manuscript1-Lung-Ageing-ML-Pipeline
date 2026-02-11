#!/usr/bin/env python3
"""
Save Tabula Muris lung, male cells only, to data/tabula_muris_lung_male_only.h5ad.
Uses config.TABULA_MURIS_PATH. Run from pipeline root.
"""

import sys
from pathlib import Path

PIPELINE_ROOT = Path(__file__).resolve().parents[1]
if str(PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(PIPELINE_ROOT))
import config


def main():
    import anndata

    path_in = Path(config.TABULA_MURIS_PATH)
    if not path_in.exists():
        print(f"Tabula Muris not found: {path_in}")
        return 1

    adata = anndata.read_h5ad(path_in)
    n_before = adata.n_obs

    sex_col = None
    for col in ("sex", "Sex", "organism_sex"):
        if col in adata.obs.columns:
            sex_col = col
            break
    if sex_col is None:
        print("No sex column in TM. obs columns:", list(adata.obs.columns))
        return 1

    sex_vals = adata.obs[sex_col].astype(str).str.strip().str.lower()
    male_mask = (sex_vals == "male") | (sex_vals == "m")
    n_male = male_mask.sum()
    adata_male = adata[male_mask].copy()

    out_path = config.DATA_DIR / "tabula_muris_lung_male_only.h5ad"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    adata_male.write_h5ad(out_path)
    print(f"Tabula Muris lung: {n_before} -> {n_male} male cells. Saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
