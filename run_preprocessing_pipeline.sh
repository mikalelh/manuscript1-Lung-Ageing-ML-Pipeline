#!/bin/bash
# Re-run full preprocessing + batch correction assessment pipeline.
# Run from: msl_aging_pipeline/
set -e
cd "$(dirname "$0")"
echo "=== 00 Column comparison (TM vs Calico) ==="
python preprocessing/00_compare_columns_tm_calico.py
echo ""
echo "=== 01 Explore datasets ==="
python preprocessing/01_explore_datasets.py
echo ""
echo "=== 02 Standardize Calico for TM ==="
python preprocessing/02_standardize_calico_for_tm.py
echo ""
echo "=== 03 Discovery preprocessing (QC + batch correction + UMAP) ==="
python preprocessing/03_preprocess_discovery.py
echo ""
echo "=== 04 Harmonization metrics ==="
python preprocessing/04_harmonization_metrics.py
echo ""
echo "=== 05 Batch correction assessment ==="
python preprocessing/05_assess_batch_correction.py
echo ""
echo "Done. Results in results/ and figures/."
