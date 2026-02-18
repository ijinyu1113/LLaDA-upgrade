#!/bin/bash
#SBATCH --job-name=1-gpu
#SBATCH --output=logs/finetune-gpu-%j.out
#SBATCH --error=logs/finetune-gpu-%j.err
#SBATCH --partition=gpuA100x4          # use the GPU partition
#SBATCH --gres=gpu:1                  # request 1 GPU
#SBATCH --account=benv-delta-gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:0


module purge
# 1. Load the tools that actually exist
module load miniforge3-python
module load cuda/12.8
# 5. Verify the environment in your logs (very helpful for debugging!)
echo "Using python from: $(which python)"
python --version
eval "$(conda shell.bash hook)"
conda activate nim-env

#pip -m install matplotlib
#timeout 48h python /u/iyu1/nim_game_project/access_files/test_model_maxrem.py
timeout 48h python /u/iyu1/nim_game_project/access_files/llada/test_router.py
#echo "=== Starting NVIDIA-SMI monitoring in background ==="
#nvidia-smi --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free --format=csv -l 60 > logs/gpu_usage_$SLURM_JOB_ID.log &
#NVIDIA_MON_
#timeout 48h python single_discriminator.py
#timeout 48h python finetune_nim.py
#timeout 48h python dann.py
#timeout 5h python test_model.py
#timeout 30h python.py
#timeout 24h python test.py
#timeout 48h python dann.py

#echo "=== Stopping NVIDIA-SMI monitoring ==="
#kill $NVIDIA_MON_PID
