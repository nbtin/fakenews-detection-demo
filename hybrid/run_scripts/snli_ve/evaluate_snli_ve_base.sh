#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7091

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=test

data=/mnt/d/data/cosmos_ofa_style/snli_ve_test_val.tsv
path=/mnt/d/data/COSMOS/OFA/run_scripts/snli_ve/checkpoints/20_5e-5_gentrain50publishtest/checkpoint.best_snli_score_0.8670.pt
result_path=../../results/snli_ve
selected_cols=0,2,3,4,5

CUDA_VISIBLE_DEVICES=0 python ../../evaluate.py \
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