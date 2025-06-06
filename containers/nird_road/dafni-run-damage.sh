#!/usr/bin/env bash
# set -ex

echo "Start dafni-run-damage.sh"

echo "Found input files:"
ls -lah /data/inputs/*

# Run process step
python nird_3_damage_analysis.py |& tee /data/outputs/log.txt
echo "model run end"

echo "Found output files:"
ls -lah /data/outputs/

echo "End dafni-run-damage.sh"
