#!/bin/bash
#SBATCH --qos=stellar-debug      # select job queue
#SBATCH --job-name=amr-gpu       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G       # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)

export DATE=`date +%F_%Hh%Mm%S`
./rungpu > run_{$DATE}_hashtable.log 2>&1
