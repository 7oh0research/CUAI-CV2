#!/usr/bin/env bash
set -e
conda run --no-capture-output -n h1_llava python -m h1bench.aggregate
echo "완료: results/summary.csv, results/ratios.csv, results/figs/"
