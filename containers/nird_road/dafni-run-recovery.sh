#!/usr/bin/env bash
# set -ex

echo "Start dafni-run-recovery.sh"
echo "Running with:"
echo "- NUMBER_CPUS: $NUMBER_CPUS"

echo "Found input files:"
ls -lah /data/inputs/*

# Run process step
python nird_4_rerouting_and_recovery.py $NUMBER_CPUS |& tee /data/outputs/log.txt
echo "model run end"

echo "Found output files:"
ls -lah /data/outputs/*

echo "End dafni-run-recovery.sh"
