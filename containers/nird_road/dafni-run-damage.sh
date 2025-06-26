#!/usr/bin/env bash
# set -ex

export DISABLE_PANDERA_IMPORT_WARNING=True
# /opt/conda/lib/python3.12/site-packages/pandera/_pandas_deprecated.py:157:
# FutureWarning: Importing pandas-specific classes and functions from the
# top-level pandera module will be **removed in a future version of pandera**.
# Set this environment variable to disable this warning

echo "Start dafni-run-damage.sh"
echo "Found input files:"
ls -lah /data/inputs/*

# Run process step
python nird_3_damage_analysis.py |& tee /data/outputs/log.txt
echo "model run end"

echo "Found output files:"
ls -lah /data/outputs/*

# Copy example notebook
mkdir -p /data/notebooks
cp visualise_damage_analysis.ipynb /data/notebooks/

echo "End dafni-run-damage.sh"
