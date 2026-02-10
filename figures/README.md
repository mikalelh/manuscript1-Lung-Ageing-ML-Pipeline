# Figures

Pipeline plots, organized by stage. See the [main README](../README.md) for pipeline overview and step descriptions.

| Folder | Contents | Scripts |
|--------|----------|---------|
| **exploration/** | Dataset overview (violins, cell-type bars) | 01_explore_datasets.py |
| **qc/** | Per-dataset QC violins (before/after filter) for TM and Calico | 03_preprocess_discovery.py |
| **discovery/** | Combined UMAP (age, cell type, batch, Leiden) | 03_preprocess_discovery.py |
| **harmonization/** | Mixing distributions, cluster composition, metrics bar, UMAPs by batch/cell type, assessment dashboard | 04_harmonization_metrics.py, 05_assess_batch_correction.py |
