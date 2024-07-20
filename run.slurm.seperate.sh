#!/bin/bash
#SBATCH -A buildlam
#SBATCH -p large-project
#SBATCH -J Seperation          # job名
#SBATCH -N 1                    # 节点数量
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8         # 每个节点的gpu数量
#SBATCH --cpus-per-task=200       # 28*GPU_PER_NODE
#SBATCH --mem=1024               # 8*CPUS_PER_TASK
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err
#SBATCH -t 3-00:00:00
#SBATCH --exclusive



nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
echo SLURM_PROCID: $SLURM_PROCID
echo SLURM_NODEID: $SLURM_NODEID
export LOGLEVEL=INFO


PROJECT_ROOT=/scratch/buildlam/codeclm

start_from=${1:-0}

# remember to change TASK_ID
srun -l --container-image $PROJECT_ROOT/containers/pytorch-23.10-py3-code-cli-torchaudio.sqsh --container-writable --container-remap-root --no-container-mount-home -p large-project \
    --container-mounts $PROJECT_ROOT/ultimatevocalremovergui:/workspace/uvr,$PROJECT_ROOT/dataset:/workspace/dataset,$PROJECT_ROOT/checkpoints:/workspace/checkpoints \
    --container-workdir /workspace/uvr bash -c "\
     bash run.multigpu.sh txwy_rename false 0 8 8 2"
