from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from .models import TransformerLanguageModelWrapper
from os.path import join
from tqdm import tqdm

import _pickle
import torch
import os
import sys


def get_task_args():
    parser = options.get_eval_lm_parser()
    return options.parse_args_and_arch(parser)


def load_megatron_lm(args):
    """
    Load Megatron_lm in fp16. A Tesla V100 is enough for inference.
    I haven't implement the parallel method for fine-tuning.
    You can refer to fairseq_eval_lm.py implementation for parallel function.
    :return: TransformerLanguageModelWrapper
    """
    megatron_path = join(args.checkpoint_dir, 'Megatron_11b', 'megatron_11b')
    # init args for task initialization
    if os.path.exists(join(megatron_path, 'task.pkl')):
        task = _pickle.load(open(join(megatron_path, 'task.pkl'), 'rb'))
    else:
        sys.argv.append(megatron_path)
        task_args = get_task_args()
        distributed_utils.infer_init_method(task_args)
        task_args.distributed_rank = None
        task = tasks.setup_task(task_args)
        _pickle.dump(task, open(join(megatron_path, 'task.pkl'), 'wb'))

    # load model partitions
    if os.path.exists(join(megatron_path, 'model.pt')):
        merge_partition = torch.load(join(megatron_path, 'model.pt'))
    else:
        merge_partition = dict()
        for i in range(8):
            # load checkpoints
            ckpt = torch.load(join(megatron_path, 'model-model_part-{}.pt'.format(i)),
                              map_location='cpu')
            if i == 0:
                merge_partition = ckpt
            else:
                print("Load from partition {}".format(i))
                for param_name, param in tqdm(ckpt['model'].items()):
                    if 'version' in param_name:
                        continue
                    src_param, tgt_param = merge_partition['model'][param_name], param
                    if param_name.endswith('out_proj.weight') or param_name.endswith('fc2.weight'):
                        res = torch.cat((src_param, tgt_param), dim=1)
                    elif param_name.endswith('k_proj.weight') or param_name.endswith('k_proj.bias') or \
                            param_name.endswith('v_proj.weight') or param_name.endswith('v_proj.bias') or \
                            param_name.endswith('q_proj.weight') or param_name.endswith('q_proj.bias') or \
                            param_name.endswith('fc1.weight') or param_name.endswith('fc1.bias') or \
                            param_name.endswith('output_projection.weight') or param_name.endswith('embed_tokens.weight'):
                        res = torch.cat((src_param, tgt_param), dim=0)
                    else:
                        res = src_param
                    merge_partition['model'][param_name] = res

    # build model
    args = merge_partition['args']
    args.model_parallel_size = 0
    # torch.save(merge_partition, join(CKPT_DIR, 'Megatron_11b/megatron_11b/model.pt'))
    model = TransformerLanguageModelWrapper.build_model(args, task)
    model.load_state_dict(merge_partition['model'])
    return model.half().cuda()
