# Results

Pipeline text outputs, organized by stage. See the [main README](../README.md) for pipeline overview, config (e.g. `USE_LABEL_TRANSFER`), and cell type alignment.

| Folder | Contents | Scripts |
|--------|----------|---------|
| **exploration/** | Column comparison (TM vs Calico), cell type overlap (matched / TM-only / Calico-only), dataset exploration summary | 00_compare_columns_tm_calico.py, 01_explore_datasets.py |
| **harmonization/** | Harmonization metrics, batch correction assessment report | 04_harmonization_metrics.py, 05_assess_batch_correction.py |
