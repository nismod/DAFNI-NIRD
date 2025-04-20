#!/bin/bash
#SBATCH --nodes=1
#SBATCH --job-name=distributed_example
#SBATCH --partition=Medium
#SBATCH --time=48:00:00
#SBATCH --mem=192512
#SBATCH --cpus-per-task=20
#SBATCH --array=0-14

echo "Running" $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
python network_flow_model_distributed.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_TASK_COUNT
