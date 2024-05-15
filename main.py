import gc
import logging
import json
import os
import time
import datetime
import torch
import argparse
import models
from pathlib import Path
import numpy as np
import pandas as pd
from configs import get_config_regression
from utils.data_loader import MMDataLoader
from trainers import Trainer
from utils import serializable_parts_of_dict, assign_gpu, setup_seed, set_logger, gen_version


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"


def _run(args, num_workers=1, from_sena=False, seed_epoch=0):
    """
    Training script for iMMAir
    """
    logger.info(f"****{args.dataset_name}: ({args.featPath})****")
    dataloader, scalers = MMDataLoader(args, num_workers)
    logger.info(f"-----{args.model_name} IMPLEMENTION ({args.version}-{seed_epoch})-----")
    
    trainer = Trainer(args, scalers, seed_epoch)

    epoch_result = trainer.do_train(dataloader, return_epoch_results=from_sena)
    trainer.model.load_state_dict(torch.load(trainer.model_save_path))
    result = trainer.do_test(dataloader['test'], mode="TEST")
    logger.info(f"-----{args.model_name} DONE ({args.version}-{seed_epoch})-----")
    del trainer.model
    torch.cuda.empty_cache()
    gc.collect()
    time.sleep(1)

    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="config.json")
    parser.add_argument('--model_name', type=str, default="iMMAir")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--mr', type=float, default=0.2, help='missing rate ranging from 0.0 to 0.6')
    parser.add_argument('--dataset_name', type=str, default="mmair")
    parser.add_argument('--train_mode', type=str, default="regression")
    parser.add_argument('--recovery_type', type=str, default="diffusion")
    parser.add_argument('--pretrained_type', type=str, default='custom', help='resnet50, dinov2, custom')
    # parser.add_argument('--local_rank', default=0, type=int, help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--fusion_type', type=str, default='cross', help='cat, hierch, cross')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()
    
    args.devices = args.devices.replace(' ', '')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]
    args.device = assign_gpu(args.device_ids)[0]


    seeds = [1111, 3333, 5555, 7777, 9999]
    model_name = args.model_name.lower()
    dataset_name = args.dataset_name.lower()
    recovery_type = args.recovery_type.lower()
    num_workers = args.num_workers

    # ----1111. Loading and Printing-----
    if args.config_file != "":
            config_file = Path(__file__).parent / "configs" / args.config_file
    else:  
        config_file = Path(__file__).parent / "configs" / "config.json"
    if not config_file.is_file():
        raise ValueError(f"Config file {str(config_file)} not found.")

    config = get_config_regression(model_name, dataset_name, recovery_type, config_file)
    config.update(vars(args))
    
    config['version'] = gen_version(config)
    pre_version = f"fix-{config.available}" if config.miss_type == 'fix' else f'mr-{config.mr}'
    save_floder = os.path.join('saved/'+ config.train_mode+'/'+config.model_name, f"{pre_version}_{config.version}")
    model_save_dir = Path(save_floder) / "pt"
    log_dir = Path(save_floder) / "logs"

    config.model_save_dir = model_save_dir
    Path(model_save_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = set_logger(log_dir, model_name, dataset_name, verbose_level=1)
    serializable_dict = serializable_parts_of_dict(config)
    logger.info(json.dumps(serializable_dict, indent=4))

    # ----2222. Training and Testing-----
    config.KeyEval = 'rmse' if config.train_mode=='regression' else 'f1_score'
    min_or_max = 'min' if config.KeyEval in ['rmse', 'mae'] else 'max'
    best_valid = 1e8 if min_or_max == 'min' else 0
    model_results, best_result = [], []

    for i, seed in enumerate(seeds):
        setup_seed(seed)
        config['cur_seed'] = seed
        result = _run(config, num_workers, seed_epoch=seed)
        model_results.append(result)
        isBetter = result[config.KeyEval] <= (best_valid - 1e-6) if min_or_max == 'min' else result[config.KeyEval] >= (best_valid + 1e-6)
        if isBetter:
            best_result, best_valid = result, result[config.KeyEval]

    # ----3333. Saving results-----
    criterions = list(model_results[0].keys())
    csv_file = Path(save_floder) / f"{pre_version}.csv"
    if csv_file.is_file():
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=["Model"] + criterions)

    res = [model_name]
    for c in criterions:
        values = [r[c] for r in model_results]
        mean = round(np.mean(values), 4)
        std = round(np.std(values), 2)
        best = best_result[c]
        res.append((best, mean, std))
    df.loc[len(df)] = res
    df.to_csv(csv_file, index=None)

    logger.info(f'Results saved to {csv_file} at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.')