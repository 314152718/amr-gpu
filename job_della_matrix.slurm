#!/bin/bash
#SBATCH --qos=della-debug      # select job queue
#SBATCH --job-name=amr-gpu-matrix # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G         # memory per cpu-core (4G is default)
#SBATCH --gres=gpu:1             # number of gpus per node
#SBATCH --time=02:00:00          # total run time limit (HH:MM:SS)
#SBATCH --partition=grace

export DATE=`date +%F_%Hh%Mm%S`
./rungpu_matrix > ./log/run_{$DATE}_matrix_GH.log 2>&1

