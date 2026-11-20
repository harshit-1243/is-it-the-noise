#!/usr/bin/env bash
# Reproduces every IA-2 result (about 25-30 min on a 2-core CPU). Run from the project root.
set -euo pipefail
python -m pytest -q tests                                    | tee results/tests_log.txt
python src/exp_main.py        --seeds 10 --out results/exp_main
python src/exp_lira.py        --models 64 --out results/exp_lira
python src/exp_ldp.py         --seeds 10 --out results/exp_ldp
python src/exp_kanon.py       --seeds 10 --out results/exp_kanon
python src/exp_compressive.py --seeds 10 --out results/exp_compressive
python src/exp_sensitivity.py --seeds 10 --out results/exp_sensitivity
python src/exp_scaling.py     --seeds 3  --out results/exp_scaling
python src/make_figures_ia2.py --results results --out figures
