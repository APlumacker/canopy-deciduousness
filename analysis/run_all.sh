#!/usr/bin/env bash
# Reproduce all analyses from the data release (DATA_DIR, default ./data) into RESULTS_DIR (default ./results).
set -euo pipefail
cd "$(dirname "$0")/.."
for s in 01_crown_metrics 02_diversity_synchrony 03_fig3_crown_example 04_fig5_metrics 05_fig6_climate 06_noise_sensitivity; do
    echo "== $s"
    python analysis/$s.py
done
