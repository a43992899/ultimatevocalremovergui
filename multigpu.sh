# Description: extract codecs from raw audio
#!/bin/bash
# source /scratch/buildlam/codeclm/miniconda3/bin/activate encodec

# user input test_flg, default false
DATASET_NAME=${1:-txwy_rename}
TEST_FLG=${2:-false}
START_FROM=${3:-0}
NNODES=${4:-1}
GPU_PER_NODE=${5:-8}
SHARD_PER_GPU=${6:-1}


if [ $DATASET_NAME = txwy_rename ]; then
    DATA_PATH=/scratch/buildlam/rawdata/codeclm/music/txwy100w_rename
    SAVE_PATH=/scratch/buildlam/rawdata/codeclm/music/txwy100w_rename_sep
    EXP_NAME=txwy100w_rename_sep
else
    echo "invalid dataset name: $DATASET_NAME"
    exit
fi


mkdir -p $SAVE_PATH
export OMP_NUM_THREADS=2

TOTAL_SHARD=$((GPU_PER_NODE*SHARD_PER_GPU*NNODES))
echo "total shard: $TOTAL_SHARD, nnodes: $NNODES, gpu_per_node: $GPU_PER_NODE, shard_per_gpu: $SHARD_PER_GPU"
echo "start_from shard: $START_FROM"
echo "data_path: $DATA_PATH"
echo "save_path: $SAVE_PATH"
echo "exp_name: $EXP_NAME"

if [ $TEST_FLG = true ]; then
    echo "testing..."
    echo "overwritting total_shard = 1 and start from shard = 0"
    python main.py \
            --input $DATA_PATH \
            --output $SAVE_PATH \
            --total_shard 1 \
            --cur_shard 0
    exit
fi

echo "extracting discrete features from raw audio..."

CUR_SHARD=$START_FROM
echo "total shard: $TOTAL_SHARD, start from shard: $START_FROM"
for CUR_GPU in $(seq 0 $((GPU_PER_NODE-1))); do
    echo "gpu $CUR_GPU"
    for SHARD in $(seq 0 $((SHARD_PER_GPU-1))); do
        echo "shard $CUR_SHARD"
        LOG_NAME=$SAVE_PATH/${EXP_NAME}_${CUR_SHARD}_of_${TOTAL_SHARD}.log
        echo "log_file: $LOG_NAME"
        nohup python main.py \
            --input $DATA_PATH \
            --output $SAVE_PATH \
            --total_shard $TOTAL_SHARD \
            --cur_shard $CUR_SHARD \
            --resume \
            --mode 0 \
            --shuffle \
            --shuffle_seed 0 \
            --cuda_idx $CUR_GPU >${LOG_NAME} 2>&1 &
        
        CUR_SHARD=$((CUR_SHARD+1))
    done
done
wait
echo "All conversions completed."