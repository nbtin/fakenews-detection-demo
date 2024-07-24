#!/bin/bash
#SBATCH --job-name=demo       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=4G         # memory per cpu-core (4G is default)
#SBATCH --time=5-00:00:00          # total run time limit (HH:MM:SS)
#Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1
#SBATCH --nodelist=selab3
#SBATCH -oslurm1.out
#SBATCH -eslurm1.err

nvidia-smi

cd /home/nbtin/repositories/fakenews-detection-demo

module use /sw/software/rootless-docker
module load rootless-docker
start_rootless_docker.sh

docker rm thesis-demo

docker run --name=thesis-demo -p 20333:8501 --gpus all -v $(realpath ./):/thesis-demo/ thesis-demo streamlit run demo_docker.py

while true; do
    sleep 1
done
# stop_rootless_docker.sh