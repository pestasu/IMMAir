import os
import pickle
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import MetricsTop, dict_to_str, move_data_to_device, restore_checkpoint, save_checkpoint

import models
import gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
my_logger = 'my_logger'
logger = logging.getLogger(my_logger)


class StandardScaler():
    """
    Standard scaler for input normalization
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

class Trainer():
    def __init__(self, args, scalers, seed_epoch):
        self.args = args
        self.need_val = args.need_val
        self.train_mode = args.train_mode
        self.metrics = MetricsTop().getMetics(args.train_mode)
        self.scalers = {
            'pm25': StandardScaler(*scalers['pm25']), 
            'aqi': StandardScaler(*scalers['aqi']),
        }
        self.model_save_path = args.model_save_dir / f"{args.model_name}-{args.train_mode}-pretrained-{seed_epoch}.pth"
        self.writer = SummaryWriter(args.model_save_dir)
        self.model = getattr(models, args.model_name)(args).to(args.device)

        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.optimizer = self._select_optimizer() 
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, verbose=True, patience=self.args.patience)      
        
        torch.autograd.set_detect_anomaly(True)

    def _select_optimizer(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.learning_rate) 

        return optimizer

    def _load_pretrained(self):
        pretrained_file = f'{self.args.model_name}-{self.args.train_mode}-pretrained.pth'
        origin_model = torch.load('pretrained/' + pretrained_file) # mr=0.0
        net_dict = self.model.state_dict()
        new_state_dict = {}
        for k, v in origin_model.items():
            k = k.replace('Model.', '')
            should_update = not any(k.startswith(prefix) for prefix in self.args.fixed_prefixes)
            if should_update: new_state_dict[k] = v
        net_dict.update(new_state_dict)
        self.model.load_state_dict(net_dict, strict=False)
        logger.info(f'Loading pretrained model from {pretrained_file}.')

    def _balance_ogm(self, outs, y_true, cur_epoch):
        relu = nn.ReLU(inplace=True)
        tanh = nn.Tanh()
        if self.train_mode == 'regression':
            score_a = torch.tensor(self.metrics(self.scalers['pm25'].inverse_transform(outs[0].reshape(y_true.shape[0], -1)), y_true)['corr'])
            score_m = torch.tensor(self.metrics(self.scalers['pm25'].inverse_transform(outs[1].reshape(y_true.shape[0], -1)), y_true)['corr'])
            score_o = torch.tensor(self.metrics(self.scalers['pm25'].inverse_transform(outs[2].reshape(y_true.shape[0], -1)), y_true)['corr'])
        elif self.train_mode == 'classification':
            score_a = torch.tensor(self.metrics(outs[0].reshape(y_true.shape[0], -1), y_true)['acc'])
            score_m = torch.tensor(self.metrics(outs[1].reshape(y_true.shape[0], -1), y_true)['acc'])
            score_o = torch.tensor(self.metrics(outs[2].reshape(y_true.shape[0], -1), y_true)['acc'])
       
        ratio_a = score_a / (max(score_m, score_o))
        ratio_m = score_m / (max(score_a, score_o))
        ratio_o = score_o / (max(score_a, score_m))
        if ratio_a > 1:
            coeff_a = 1 - tanh(self.alpha * relu(ratio_a))
        else:
            coeff_a = 1

        if ratio_m > 1:
            coeff_m = 1 - tanh(self.alpha * relu(ratio_m))
        else:
            coeff_m = 1

        if ratio_o > 1:
            coeff_o = 1 - tanh(self.alpha * relu(ratio_o))
        else:
            coeff_o = 1
        
        # print(f"score!!!!!!!!!!!!{score_a},{score_m},{score_o}")
        # print(f"ratio!!!!!!!!!!!!{ratio_a},{ratio_m},{ratio_o}")
        # print(f"coeff!!!!!!!!!!!!{coeff_a},{coeff_m},{coeff_o}")
        
        if self.modulation_epoch[0] <= cur_epoch <= self.modulation_epoch[1]: # bug fixed
            for name, parms in self.model.named_parameters():
                layer = str(name).split('.')[1]
                if str(name).split('.')[0] != 'cdr_layer':
                    if ('_a' in layer) and (parms.grad is not None) and (len(parms.grad.size()) >1):
                        if self.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_a + \
                                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif self.modulation == 'OGM':
                            parms.grad *= coeff_a

                    if ('_m' in layer) and (parms.grad is not None) and (len(parms.grad.size()) >1):
                        if self.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_m + \
                                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif self.modulation == 'OGM':
                            parms.grad *= coeff_m 

                    if ('_o' in layer) and (parms.grad is not None) and (len(parms.grad.size()) >1):
                        if self.modulation == 'OGM_GE':  # bug fixed
                            parms.grad = parms.grad * coeff_o + \
                                            torch.zeros_like(parms.grad).normal_(0, parms.grad.std().item() + 1e-8)
                        elif self.modulation == 'OGM':
                            parms.grad *= coeff_o 

    def do_train(self, dataloader, return_epoch_results=False):     
        # initilize results
        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }

        min_or_max = 'min' if self.args.KeyEval in ['rmse', 'mae'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        torch.cuda.empty_cache()
        gc.collect()

        # load pretrained
        self._load_pretrained()

        while True:
            epochs += 1
            # ----------------train--------------------------------------------------
            y_pred, y_true = [], []

            self.model.train()
            train_loss = 0.0
            miss_one, miss_two = 0, 0  # num of missing one modal and missing two modal
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    self.optimizer.zero_grad()
                    mark = move_data_to_device(batch_data['mark'], self.args.device_ids) 
                    aqi = move_data_to_device(batch_data['aqi'], self.args.device_ids) 
                    meo = move_data_to_device(batch_data['meo'], self.args.device_ids) 
                    photo = move_data_to_device(batch_data['photo'], self.args.device_ids) 
                    if self.args.train_mode == 'classification':
                        labels = move_data_to_device(batch_data['label'], self.args.device_ids) 
                        labels = labels.view(-1).long()
                    else:
                        labels = move_data_to_device(batch_data['hats']['pm25'], self.args.device_ids) 
                        labels = labels.view(-1, self.args.pred_len)

                    # forward
                    if self.args.miss_type == 'fix':
                        outputs = self.model(aqi, meo, photo, labels, mark)
                    else:
                        miss_2 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 0.0]
                        miss_1 = [0.1, 0.2, 0.3, 0.4, 0.3, 0.0, 0.0]

                        if miss_two / (np.round(len(dataloader['train']) / 10) * 10) < miss_2[int(self.args.mr*10-1)]:  # missing two modal
                            outputs = self.model(aqi, meo, photo, labels, mark, num_modal=1, is_train=True)
                            miss_two += 1
                        elif miss_one / (np.round(len(dataloader['train']) / 10) * 10) < miss_1[int(self.args.mr*10-1)]:  # missing one modal
                            outputs = self.model(aqi, meo, photo, labels, mark, num_modal=2, is_train=True)
                            miss_one += 1
                        else:  # no missing
                            outputs = self.model(aqi, meo, photo, labels, mark, num_modal=3, is_train=True)
                    # compute loss
                    if self.args.train_mode == 'classification':
                        raw_pred = outputs['y_hat'].reshape(labels.shape[0], -1)
                        raw_true = labels
                    else:
                        raw_pred = self.scalers['pm25'].inverse_transform(outputs['y_hat'].reshape(labels.shape[0], -1))
                        raw_true = self.scalers['pm25'].inverse_transform(labels)
                    task_loss = self.criterion(raw_pred, raw_true)
                    loss_trans = outputs['loss_trans']
                    loss_rec = outputs['loss_rec']
                    # print(f'task_loss:{task_loss}, loss_trans:{loss_trans}')
                    combine_loss = task_loss + self.lambda_l * (loss_trans + loss_rec)
                    # backward
                    combine_loss.backward()

                    if self.args.grad_clip != -1.0:
                        nn.utils.clip_grad_value_([param for param in self.model.parameters() if param.requires_grad],
                                                  self.args.grad_clip)
                    if self.is_balance:
                        self._balance_ogm(outputs['uni_hat'], raw_true, cur_epoch=epochs)
                    
                    # store results
                    train_loss += combine_loss.item()
                    y_pred.append(raw_pred.cpu())
                    y_true.append(raw_true.cpu())
                    self.optimizer.step()

            train_loss = train_loss / len(dataloader['train'])

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            train_results = self.metrics(pred, true)
            logger.info(
                f"TRAIN-({self.args.model_name}) [{epochs - best_epoch}/{epochs}/{self.args.cur_seed}] "
                f">> loss: {round(train_loss, 4)} "
                f"{dict_to_str(train_results)}"
            )
            self.writer.add_scalar(
            'train/train_loss', train_loss, global_step=epochs)

            # ----------------validation----------------
            val_results = self.do_test(self.model, dataloader['valid'], mode="VAL")
            for key, val in val_results.items():
                self.writer.add_scalar(f'validate/{key}', val, global_step=epochs)

            # ----------------test----------------
            test_results = self.do_test(dataloader['test'], mode="TEST")

            # save test results
            for key, val in test_results.items():
                self.writer.add_scalar(f'test/{key}', val, global_step=epochs)
            cur_valid = val_results[self.args.KeyEval] if self.need_val else test_results[self.args.KeyEval]
            self.scheduler.step(test_results['Loss'])                                                                                                               
            # save each epoch model
            # model_save_path = self.args.model_save_dir + str(epochs) + '.pth'
            # torch.save(model.state_dict(), model_save_path)
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            # save best model
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(self.model.cpu().state_dict(), self.model_save_path)
                print(f'------------{self.model_save_path}:{cur_valid}------------')
                self.model.to(self.args.device)

                # save parameters
                args_dict = {attr: getattr(self.args, attr) for attr in dir(self.args) if not callable(getattr(self.args, attr)) and not attr.startswith("__")}
                args_dict_str = {key: str(val) for key, val in args_dict.items()}
                metric_dict_float = {f'best_{key}': float(val) for key, val in test_results.items()}
                self.writer.add_hparams(args_dict_str, metric_dict_float)
            # epoch results
            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                if self.need_val: epoch_results['valid'].append(val_results)
                test_results = self.do_test(dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)
            # early stop
            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, dataloader, mode="VAL", return_sample_results=False):
        self.model.eval()
        y_pred, y_true = [], []
        miss_one, miss_two = 0, 0
        eval_loss = 0.0
        if return_sample_results:
            sample_results = {}
            features = {
                "ava_modal_idx":[],
                "feat_air": [],
                "feat_meo": [],
                "feat_pho": [],
                "raw_air": [],
                "raw_meo": [],
                "raw_pho": []
                }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    mark = move_data_to_device(batch_data['mark'], self.args.device_ids) 
                    aqi = move_data_to_device(batch_data['aqi'], self.args.device_ids) 
                    meo = move_data_to_device(batch_data['meo'], self.args.device_ids) 
                    photo = move_data_to_device(batch_data['photo'], self.args.device_ids) 
                    if self.args.train_mode == 'classification':
                        labels = move_data_to_device(batch_data['label'], self.args.device_ids) 
                        labels = labels.view(-1).long()
                    else:
                        labels = move_data_to_device(batch_data['hats']['pm25'], self.args.device_ids) 
                        labels = labels.view(-1, self.args.pred_len)
                    
                    # pretraining the pretrained model
                    if self.args.miss_type == 'fix':
                        outputs = self.model(aqi, meo, photo, labels, mark, is_train=False)

                    else:
                        miss_2 = [0.1, 0.2, 0.3, 0.4, 0.6, 0.9, 0.0]
                        miss_1 = [0.1, 0.2, 0.3, 0.4, 0.3, 0.0, 0.0]
                        if miss_two / (np.round(len(dataloader) / 10) * 10) < miss_2[int(self.args.mr*10-1)]:  # missing two modal
                            outputs = self.model(aqi, meo, photo, labels, mark, num_modal=1, is_train=False)
                            miss_two += 1

                        elif miss_one / (np.round(len(dataloader) / 10) * 10) < miss_1[int(self.args.mr*10-1)]:  # missing one modal
                            outputs = self.model(aqi, meo, photo, labels, mark, num_modal=2, is_train=False)
                            miss_one += 1
                        else:  # no missing
                            outputs = self.model(aqi, meo, photo, labels, mark, num_modal=3, is_train=False)

                    if return_sample_results:
                        if len(outputs['ava_modal_idx']) < 3:
                            combined_ava_modal_idx = [''.join(outputs['ava_modal_idx'])]
                            features['ava_modal_idx'].append(combined_ava_modal_idx*outputs['feat_z1'][0].shape[0])
                            features['feat_air'].append(outputs['feat_z1'][0].cpu().detach().numpy())
                            features['feat_meo'].append(outputs['feat_z1'][1].cpu().detach().numpy())
                            features['feat_pho'].append(outputs['feat_z1'][2].cpu().detach().numpy())
                            features['raw_air'].append(outputs['raw_z1'][0].cpu().detach().numpy())
                            features['raw_meo'].append(outputs['raw_z1'][1].cpu().detach().numpy())
                            features['raw_pho'].append(outputs['raw_z1'][2].cpu().detach().numpy())

                    if self.args.train_mode == 'classification':
                        raw_pred = outputs['y_hat'].reshape(labels.shape[0], -1)
                        raw_true = labels
                    else:
                        raw_pred = self.scalers['pm25'].inverse_transform(outputs['y_hat'].reshape(labels.shape[0], -1))
                        raw_true = self.scalers['pm25'].inverse_transform(labels)

                    loss = self.criterion(raw_pred, raw_true)
                    eval_loss += loss.item()
                    y_pred.append(raw_pred.cpu())
                    y_true.append(raw_true.cpu())

        eval_loss = eval_loss / len(dataloader)
        pred, true = torch.cat(y_pred), torch.cat(y_true)
        eval_results = self.metrics(pred, true)

        eval_results["Loss"] = round(eval_loss, 4)
        logger.info(f"{mode}-({self.args.model_name}) >> {dict_to_str(eval_results)}")

        if return_sample_results:
            for k in features.keys():
                sample_results[k] = np.concatenate(features[k], axis=0)
            output_pkl_file = self.args.model_save_dir / 'MISS_AIR.pkl'
            with open(output_pkl_file, 'wb') as pkl_file:
                pickle.dump(sample_results, pkl_file)

        return eval_results
