
import torch
from fairseq import utils
from PIL import Image
from io import BytesIO
import base64
from time import time

def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t

CONTEXT = {
    'yes': 0,
    'no': 1,
}
def data_preprocess(dataset, img_path, caption1, caption2, use_cuda, cfg):
    img = Image.open(img_path) # path to file
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data) # bytes
    base64_str = base64_str.decode("utf-8") # str
    samples = [[caption1, caption2], [caption2, caption1]]
    batches = []
    for i, sml in enumerate(samples):
        caption1, caption2 = sml
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
        batches.append(example)
    sample = dataset.collater(batches)
    sample = utils.move_to_cuda(sample) if use_cuda else sample
    sample = utils.apply_to_sample(
        apply_half, sample) if cfg.common.fp16 else sample
    return sample