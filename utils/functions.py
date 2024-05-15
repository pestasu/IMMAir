import os
import json
import glob
import random
import pynvml
import logging
import torch
import numpy as np
from pathlib import Path
import torch.nn as nn
from torch.nn.parallel import scatter

my_logger = 'my_logger'
logger = logging.getLogger(my_logger)


def move_data_to_device(data, device):
    if len(device) > 1:
        data = scatter(data, device, dim=0)[0]
    else:
        set_device = torch.device(f'cuda:{device[0]}' if len(device) > 0 else 'cpu')
        data = data.to(set_device)

    return data

def is_json_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False

def serializable_parts_of_dict(d):
    serializable_dict = {k: v for k, v in d.items() if is_json_serializable(v)}
    return serializable_dict

def gen_version(args):
    dirs = glob.glob(os.path.join('saved/'+ args.train_mode+'/'+args.model_name, '*'))
    if not dirs:
        return 0
    if args.version < 0:
        version = max([int(x.split(os.sep)[-1].split('_')[-1])
                    for x in dirs]) + 1
    else:
        version = args.version
    return version

def set_logger(log_dir, model_name, dataset_name, verbose_level=1):
    # base logger
    log_file_path = Path(log_dir) / f"{model_name}-{dataset_name}.log"
    logger = logging.getLogger(my_logger)
    logger.setLevel(logging.DEBUG)

    # file handler
    fh = logging.FileHandler(log_file_path)
    fh_formatter = logging.Formatter('%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fh_formatter)
    logger.addHandler(fh)

    # stream handler
    stream_level = {0: logging.ERROR, 1: logging.INFO, 2: logging.DEBUG}
    ch = logging.StreamHandler()
    ch.setLevel(stream_level[verbose_level])
    ch_formatter = logging.Formatter('%(name)s - %(message)s')
    ch.setFormatter(ch_formatter)
    logger.addHandler(ch)

    return logger

def dict_to_str(src_dict):
    dst_str = ""
    for key in src_dict.keys():
        dst_str += " %s: %.4f " %(key, src_dict[key]) 
    return dst_str

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def assign_gpu(gpu_ids, memory_limit=1e16):
    if len(gpu_ids) == 0 and torch.cuda.is_available():
        # find most free gpu
        pynvml.nvmlInit()
        n_gpus = pynvml.nvmlDeviceGetCount()
        dst_gpu_id, min_mem_used = 0, memory_limit
        for g_id in range(n_gpus):
            handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            mem_used = meminfo.used
            if mem_used < min_mem_used:
                min_mem_used = mem_used
                dst_gpu_id = g_id
        logger.info(f'Found gpu {dst_gpu_id}, used memory {min_mem_used}.')
        gpu_ids.append(dst_gpu_id)
    # device
    using_cuda = len(gpu_ids) > 0 and torch.cuda.is_available()
    # logger.info("Let's use %d GPUs!" % len(gpu_ids))
    
    device = [torch.device('cuda:%d' % int(gpu_id)) for gpu_id in gpu_ids] if using_cuda else [torch.device('cpu')]
    return device

def count_parameters(model):
    res = 0
    for p in model.parameters():
        if p.requires_grad:
            res += p.numel()
            # print(p)
    return res

def save_checkpoint(ckpt_dir, state):
  saved_state = {
    'optimizer': state['optimizer'].state_dict(),
    'model': state['model'].state_dict(),
    'ema': state['ema'].state_dict(),
    'step': state['step']
  }
  torch.save(saved_state, ckpt_dir)