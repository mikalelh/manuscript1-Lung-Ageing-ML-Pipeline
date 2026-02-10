"""
Configuration for Manuscript 1 aging pipeline.
All paths are configurable here.
"""

from pathlib import Path

# -----------------------------------------------------------------------------
# Data paths (edit these to match your system)
# -----------------------------------------------------------------------------
TABULA_MURIS_PATH = "/mnt/data/melhajjar/tabula_muris/all_tissues/Lung.h5ad"
CALICO_PATH = "/mnt/data/melhajjar/tabula_muris/all_tissues/droplet_h5ad/lung_calico.h5ad"
VALIDATION_PATH = "/mnt/data/melhajjar/tabula_muris/all_tissues/droplet_h5ad/GSE124872_seurat_cells_annotated.h5ad"

# -----------------------------------------------------------------------------
# Output directories (relative to this config file)
# -----------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = _CONFIG_DIR / "results"
FIGURES_DIR = _CONFIG_DIR / "figures"
DATA_DIR = _CONFIG_DIR / "data"
CELLCHAT_DIR = _CONFIG_DIR / "cellchat"

# Organized subfolders (scripts write here)
RESULTS_EXPLORATION = OUTPUT_DIR / "exploration"      # 00, 01: column comparison, dataset exploration
RESULTS_HARMONIZATION = OUTPUT_DIR / "harmonization"  # 04, 05: metrics, batch correction assessment
FIGURES_EXPLORATION = FIGURES_DIR / "exploration"     # 01: dataset overview
FIGURES_QC = FIGURES_DIR / "qc"                       # 03: per-dataset QC violins
FIGURES_DISCOVERY = FIGURES_DIR / "discovery"         # 03: combined UMAP
FIGURES_HARMONIZATION = FIGURES_DIR / "harmonization" # 04, 05: mixing, clusters, dashboard
RESULTS_MODELS = OUTPUT_DIR / "models"  # LSTM checkpoints (models/lstm_aging_model.py)

# Standardized Calico (age 7m/22m, cell_ontology_class); created by preprocessing/02_standardize_calico_for_tm.py
CALICO_STANDARDIZED_PATH = DATA_DIR / "lung_calico_standardized.h5ad"

# Discovery combined (TM + Calico, QC + batch correction + UMAP); created by preprocessing/03_preprocess_discovery.py
DISCOVERY_COMBINED_PATH = DATA_DIR / "discovery_combined.h5ad"

# Batch correction: set True to use scCobra (https://github.com/mcgilldinglab/scCobra), False for Harmony
USE_SCCOBRA = True
# Embedding key in discovery_combined.obsm (03 writes the chosen method here as X_pca_harmony for compatibility)
INTEGRATION_EMBEDDING_KEY = "X_pca_harmony"

# Calico cell types: True = relabel Calico via scANVI label transfer from TM (recommended); False = dictionary mapping
USE_LABEL_TRANSFER = True

# Ensure dirs exist when config is loaded (optional)
for _d in (
    OUTPUT_DIR, FIGURES_DIR, DATA_DIR, CELLCHAT_DIR,
    RESULTS_EXPLORATION, RESULTS_HARMONIZATION, RESULTS_MODELS,
    FIGURES_EXPLORATION, FIGURES_QC, FIGURES_DISCOVERY, FIGURES_HARMONIZATION,
):
    _d.mkdir(parents=True, exist_ok=True)
