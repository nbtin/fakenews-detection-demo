#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import logging
import os
import sys

import numpy as np
import torch
import sklearn
from fairseq import distributed_utils, options, tasks, utils
from ptflops import get_model_complexity_info
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.utils import reset_logging
from omegaconf import DictConfig

from utils import checkpoint_utils
from utils.eval_utils import eval_step
from PIL import Image
from io import BytesIO
from tqdm import tqdm
from time import time
import base64
import json
import re

CONTEXT = {
    'yes': 0,
    'no': 1,
}

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")

IMG_PREFIX = os.environ.get("INPUT_FOLDER", None)
assert IMG_PREFIX is not None, "Please set INPUT_FOLDER environment variable"

def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

def data_preprocess(dataset, img_path, caption1, caption2, use_cuda, cfg):
    img = Image.open(img_path) # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str

    uniq_id, image, hypothesis, caption, label = 0, base64_str, caption1, caption2, "OOC"
    if label == 'OOC':
        label = 'no'
    elif label == 'NOOC':
        label = 'yes'
    elif label == 'neutral':
        label = 'maybe'
    else:
        raise NotImplementedError
    
    image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
    patch_image = dataset.patch_resize_transform(image)
    patch_mask = torch.tensor([True])

    hypothesis = dataset.pre_caption(hypothesis, dataset.max_src_length)
    src_item = dataset.encode_text(' does the image describe " {} "?'.format(hypothesis))
    tgt_item = dataset.encode_text(" {}".format(label))
    ref_dict = {label: 1.0}

    if dataset.add_caption:
        caption = dataset.pre_caption(caption, dataset.max_src_length)
        src_item = dataset.encode_text(' can image and text1 " {} " imply text2 " {} "?'.format(caption, hypothesis))

    src_item = torch.cat([dataset.bos_item, src_item, dataset.eos_item])
    if dataset.prompt_type == 'none':
        prev_output_item = torch.cat([dataset.bos_item, tgt_item])
        target_item = torch.cat([prev_output_item[1:], dataset.eos_item])
        decoder_prompt = dataset.bos_item
    elif dataset.prompt_type == 'src':
        prev_output_item = torch.cat([src_item, tgt_item])
        target_item = torch.cat([prev_output_item[1:], dataset.eos_item])
        decoder_prompt = src_item
    elif dataset.prompt_type == 'prev_output':
        prev_output_item = torch.cat([src_item[:-1], tgt_item])
        target_item = torch.cat([prev_output_item[1:], dataset.eos_item])
        decoder_prompt = src_item[:-1]
    else:
        raise NotImplementedError
    target_item[:-len(tgt_item)-1] = dataset.tgt_dict.pad()

    example = {
        "id": uniq_id,
        "source": src_item,
        "patch_image": patch_image,
        "patch_mask": patch_mask,
        "target": target_item,
        "prev_output_tokens": prev_output_item,
        "decoder_prompt": decoder_prompt,
        "ref_dict": ref_dict,
    }
    if dataset.constraint_trie is not None:
        constraint_mask = torch.zeros((len(target_item), len(dataset.tgt_dict))).bool()
        start_idx = len(target_item) - len(tgt_item) - 1
        for i in range(len(target_item)-len(tgt_item)-1, len(target_item)):
            constraint_prefix_token = [dataset.tgt_dict.bos()] + target_item[start_idx:i].tolist()
            constraint_nodes = dataset.constraint_trie.get_next_layer(constraint_prefix_token)
            constraint_mask[i][constraint_nodes] = True
        example["constraint_mask"] = constraint_mask
    sample = dataset.collater([example])
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(
        apply_half, sample) if cfg.common.fp16 else sample
    return sample

def main(cfg: DictConfig, **kwargs):
    utils.import_user_module(cfg.common)

    reset_logging()
    logger.info(cfg)

    assert (
        cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)
    # Deal with beam-search / all-candidate VQA eval
    data_fl = open(cfg.task.data, 'r')
    temp_datafile = 'temp.tsv'
    fl = open(temp_datafile, 'w')
    fl.write("id\timg\tsentence1\tsentence2\tlabel\n")
    fl.close()
    cfg.task.data = temp_datafile
    overrides['data'] = temp_datafile
    if cfg.task._name == "vqa_gen":
        overrides['val_inference_type'] = "beamsearch" if kwargs['beam_search_vqa_eval'] else "allcand"

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if kwargs["zero_shot"]:
        task = tasks.setup_task(cfg.task)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    # loading the dataset should happen after the checkpoint has been loaded
    # so we can give it the saved task config
    saved_cfg.task.data = temp_datafile
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    if cfg.generation.lm_path is not None:
        overrides["data"] = cfg.task.data

        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [cfg.generation.lm_path], arg_overrides=overrides, task=None
            )
        except:
            logger.warning(
                f"Failed to load language model! Please make sure that the language model dict is the same "
                f"as target dict and is located in the data dir ({cfg.task.data})"
            )
            raise

        assert len(lms) == 1
    else:
        lms = [None]

    # Move models to GPU
    for model, ckpt_path in zip(
            models, utils.split_paths(
            cfg.common_eval.path)):
        if kwargs['ema_eval']:
            logger.info("loading EMA weights from {}".format(ckpt_path))
            model.load_state_dict(
                checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    dataset=task.dataset(cfg.dataset.gen_subset)
    datalines = data_fl.readlines()

    predict_context_task = []
    gt_context_task = []
    GFlops = 0
    inference_time_cap1_cap2 = 0
    inference_time_cap2_cap1 = 0
    for i,dataline in tqdm(enumerate(datalines)):
        data_point = json.loads(dataline)
        img_path = os.path.join(IMG_PREFIX, data_point['img_local_path'])
        caption1 = data_point['caption1']
        caption2 = data_point['caption2'] if 'caption2' in data_point else ""

        sample=data_preprocess(dataset, img_path, caption1, "", use_cuda, cfg)
        start = time()
        result1, scores1, valid_result1 = eval_step(
            task, None, models, sample, **kwargs)
        inference_time_cap1_cap2 += time() - start
        sample=data_preprocess(dataset, img_path, "", caption1, use_cuda, cfg)
        start = time()
        result2, scores2, valid_result2 = eval_step(
            task, None, models, sample, **kwargs)
        inference_time_cap2_cap1 += time() - start
        
        valid_result = valid_result1 + valid_result2
        answer = valid_result.argmax(1)
        if answer == 1:
            answer = 'yes'
        else:
            answer = 'no'
        
        predict_context_task.append(CONTEXT[answer])
        gt_context_task.append(data_point['context_label'])
        
        macs, params = get_model_complexity_info(models[0], sample, task, as_strings=True,
            print_per_layer_stat=False, verbose=False)
        # Extract the numerical value
        flops = eval(re.findall(r'([\d.]+)', macs)[0])*2

        # Extract the unit
        flops_unit = re.findall(r'([A-Za-z]+)', macs)[0][0]
        print('Computational complexity: {:<8}'.format(macs))
        print('Computational complexity: {} {}Flops'.format(flops, flops_unit))
        print('Number of parameters: {:<8}'.format(params))
        GFlops += flops
    
    print("accuracy task 2: ", sklearn.metrics.accuracy_score(gt_context_task, predict_context_task))
    print("f1 - score task 2: ", sklearn.metrics.f1_score(gt_context_task, predict_context_task))
    print("Average precision task 2: ", sklearn.metrics.average_precision_score(gt_context_task, predict_context_task))
    print(f"Average GFlops per {len(datalines)} samples for task 2: {GFlops/len(datalines)} {flops_unit}")
    print('Number of parameters: {:<8}'.format(params))
    print(f"Average inference time (cap1, cap2 direction) per {len(datalines)} samples for task 2: {inference_time_cap1_cap2/len(datalines)} seconds")
    print(f"Average inference time (cap1, cap2 direction) per {len(datalines)} samples for task 2: {inference_time_cap2_cap1/len(datalines)} seconds")

def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument(
        "--ema-eval",
        action='store_true',
        help="Use EMA weights to make evaluation.")
    parser.add_argument(
        "--beam-search-vqa-eval",
        action='store_true',
        help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
    parser.add_argument("--zero-shot", action='store_true')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(
        cfg,
        main,
        ema_eval=args.ema_eval,
        beam_search_vqa_eval=args.beam_search_vqa_eval,
        zero_shot=args.zero_shot,
    )


if __name__ == "__main__":
    cli_main()
