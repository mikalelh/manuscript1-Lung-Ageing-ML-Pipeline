"""
Canonical cell type names so TM and Calico use the same labels.

How matching works:
- One shared dict CANONICAL_CELLTYPE_MAP: (any label → canonical name).
- Step 02: Calico cell_type is mapped through this to set cell_ontology_class
  (e.g. "natural killer cell" → "NK cell", "lung endothelial cell" → "vein endothelial cell").
- Step 03: Both TM and Calico cell_ontology_class are mapped when creating cell_type_raw,
  so the combined object has a single set of names; TM labels not in the map are kept as-is.

Result: wherever the same biology had different names, we use one canonical name.
Types only in one dataset (e.g. TM "CD8-positive, alpha-beta T cell", Calico "stromal cell")
stay as-is, so they appear as distinct labels (no forced lumping).
"""

import pandas as pd

# Map (dataset label → canonical name). Use TM names as canonical where possible.
# Only labels that need changing are listed; others pass through as-is.
CANONICAL_CELLTYPE_MAP = {
    # Calico-only renames so they match TM:
    "natural killer cell": "NK cell",
    "lung endothelial cell": "vein endothelial cell",
    # Identity (same in both; listed so we have one place to edit if needed):
    "stromal cell": "stromal cell",
    "myeloid cell": "myeloid cell",
    "leukocyte": "leukocyte",
    "mast cell": "mast cell",
    "B cell": "B cell",
    "T cell": "T cell",
    "classical monocyte": "classical monocyte",
    "non-classical monocyte": "non-classical monocyte",
    "alveolar macrophage": "alveolar macrophage",
    "type II pneumocyte": "type II pneumocyte",
    "ciliated columnar cell of tracheobronchial tree": "ciliated columnar cell of tracheobronchial tree",
}


def apply_celltype_mapping(series: pd.Series) -> pd.Series:
    """Map a pandas Series of cell type labels to canonical names."""
    if series is None or len(series) == 0:
        return series
    s = series.astype(str).str.strip()
    return s.map(lambda x: CANONICAL_CELLTYPE_MAP.get(x, x))
