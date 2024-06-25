#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7091
export PYTHONPATH=$PYTHONPATH:../../flops/

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=test
export INPUT_FOLDER=/mnt/d/data/COSMOS/icmr2024/
data=/mnt/d/data/COSMOS/icmr2024/test.json
path=/mnt/d/data/COSMOS/OFA/run_scripts/snli_ve/checkpoints/checkpoint.gendata_task2_4000.best_snli_score_0.7870.pt
result_path=../../results/snli_ve
selected_cols=0,2,3,4,5

CUDA_VISIBLE_DEVICES=0 python ../../inference_batch.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=2 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"