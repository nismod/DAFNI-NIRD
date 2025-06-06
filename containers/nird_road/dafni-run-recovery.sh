#!/usr/bin/env bash
# set -ex

echo "Start dafni-run-recovery.sh"

echo "Running with:"
echo "- NUMBER_CPUS: $NUMBER_CPUS"
# echo "- DEPTH_THRESHOLD: $DEPTH_THRESHOLD"

echo "Found input files:"
ls -lah /data/inputs/*

# Run process step
python nird_4_rerouting_and_recovery.py $NUMBER_CPUS
# python nird_3_damage_analysis.py
# date > /data/outputs/test.txt

echo "model run end"

echo "Found output files:"
ls -lah /data/outputs/*

echo "End dafni-run-recovery.sh"
