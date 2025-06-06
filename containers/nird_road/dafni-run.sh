#!/usr/bin/env bash
# set -ex

echo "Start dafni-run.sh"

echo "Running with:"
echo "- ncpus: $NUMBER_CPUS"
echo "- depth threshold: $DEPTH_THRESHOLD"

echo "Found input files:"
ls -lah /data/inputs/*

# Run process step
python nird_4_rerouting_and_recovery.py
# python nird_3_damage_analysis.py
# date > /data/outputs/test.txt
# python network_flow_model.py /data/inputs/nird_config.json
echo "model run end"

echo "Found output files:"
ls -lah /data/outputs/*

echo "End dafni-run.sh"
