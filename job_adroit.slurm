#!/bin/bash
#SBATCH --job-name=amr-gpu   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=200G       # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=00:20:00          # total run time limit (HH:MM:SS)

export DATE=`date +%F_%Hh%M`
./rungpu > run_$DATE.log 2>&1

