#!/bin/bash
#SBATCH -c 8                               # Request one core
#SBATCH -N 1                               # Request one node (if you request more than one core with -c, also using
                                           # -N 1 means all cores will be on the same node)
#SBATCH -t 0-02:30                         # Runtime in D-HH:MM format
#SBATCH -p short                           # Partition to run in
#SBATCH --mem=1500M                          # Memory total in MB (for all cores)
#SBATCH -o hostname_%j.out                 # File to which STDOUT will be written, including job ID
#SBATCH -e hostname_%j.err                 # File to which STDERR will be written, including job ID
#SBATCH --mail-type=FAIL                    # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=dl263@hms.harvard.edu   # Email to which notifications will be sent


EXP=$1
EXTRA_ARGS=$2

ROOT_DIR=/home/dl263/varderiv/
source activate varderiv
cd $ROOT_DIR

export XLA_FLAGS="--xla_llvm_disable_expensive_passes"
python -m "varderiv.experiments.${EXP}" -p --sql=sqlite:///$ROOT_DIR/results.db $EXTRA_ARGS