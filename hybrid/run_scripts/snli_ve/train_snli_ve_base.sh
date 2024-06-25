#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7061

log_dir=./logs
save_dir=./checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

data_dir=/mnt/d/data/cosmos_ofa_style/
data=/mnt/d/data/cosmos_ofa_style/gendata_task1_task2_mixed_2000.tsv,/mnt/d/data/cosmos_ofa_style/snli_ve_publish_test.tsv
restore_file=/mnt/d/data/COSMOS/OFA/ofa_tiny.pt
selected_cols=0,2,3,4,5

task=snli_ve
arch=ofa_tiny
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.0
lr=1e-3
max_epoch=5
warmup_ratio=0.06
batch_size=2
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=480
prompt_type="prev_output"
echo "max_epoch "
for max_epoch in 20; do
  echo "max_epoch "${max_epoch}
  for lr in 5e-5; do
    echo "lr "${lr}

    log_file=${log_dir}/${max_epoch}"_"${lr}".log"
    save_path=${save_dir}/${max_epoch}"_"${lr}
    mkdir -p $save_path

    python ../../train.py \
        $data \
        --selected-cols=${selected_cols} \
        --bpe-dir=${bpe_dir} \
        --user-dir=${user_dir} \
        --restore-file=${restore_file} \
        --reset-optimizer --reset-dataloader --reset-meters \
        --save-dir=${save_path} \
        --task=${task} \
        --arch=${arch} \
        --criterion=${criterion} \
        --label-smoothing=${label_smoothing} \
        --batch-size=${batch_size} \
        --update-freq=${update_freq} \
        --encoder-normalize-before \
        --decoder-normalize-before \
        --share-decoder-input-output-embed \
        --share-all-embeddings \
        --layernorm-embedding \
        --patch-layernorm-embedding \
        --code-layernorm-embedding \
        --resnet-drop-path-rate=${resnet_drop_path_rate} \
        --encoder-drop-path-rate=${encoder_drop_path_rate} \
        --decoder-drop-path-rate=${decoder_drop_path_rate} \
        --dropout=${dropout} \
        --attention-dropout=${attention_dropout} \
        --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
        --lr-scheduler=polynomial_decay --lr=${lr} \
        --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
        --log-format=simple --log-interval=10 \
        --fixed-validation-seed=7 \
        --keep-best-checkpoints=1 \
        --save-interval=1 --validate-interval=1 \
        --save-interval-updates=500 --validate-interval-updates=500 \
        --best-checkpoint-metric=snli_score --maximize-best-checkpoint-metric \
        --max-src-length=${max_src_length} \
        --max-tgt-length=${max_tgt_length} \
        --find-unused-parameters \
        --add-type-embedding \
        --scale-attn \
        --scale-fc \
        --scale-heads \
        --disable-entangle \
        --num-bins=${num_bins} \
        --patch-image-size=${patch_image_size} \
        --prompt-type=${prompt_type} \
        --add-caption \
        --fp16 \
        --fp16-scale-window=512 \
        --num-workers=0
  done
done