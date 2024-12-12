#!/bin/bash --login
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=15G
#SBATCH --time=01:30:00
#SBATCH --partition=Short
#SBATCH --array=0-481

# >>> mamba initialize >>>
# !! Contents within this block are managed by 'mamba init' !!
export MAMBA_EXE='/lustre/soge1/users/mert2014/.local/bin/micromamba';
export MAMBA_ROOT_PREFIX='/lustre/soge1/users/mert2014/micromamba';
__mamba_setup="$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__mamba_setup"
else
    alias micromamba="$MAMBA_EXE"  # Fallback on help from mamba activate
fi
unset __mamba_setup
# <<< mamba initialize <<<

# VALUES=({0..1000})
#VALUES=({1000..2000})
VALUES=({2000..3000})
LINE_NUMBER=${VALUES[$SLURM_ARRAY_TASK_ID]}

# Read index_i,index_j from a text file
LINE=(awk "NR==$LINE_NUMBER" 2.2_road_link_indexes.csv)
INDEX_I=${LINE%,*}
INDEX_J=${LINE#*,}

# Run exposure (and disruption?) for the subset of roads
micromamba run --name skmob \
    python 2.2_disruption_analysis.py $INDEX_I $INDEX_J ../config.json
