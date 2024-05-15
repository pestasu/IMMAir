import logging
import pickle
import numpy as np
import ipdb
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from PIL import Image

__all__ = ['MMDataLoader']
my_logger = 'my_logger'
logger = logging.getLogger(my_logger)


class MMDataset(Dataset):
    def __init__(self, args, mode='train'):
        self.mode = mode
        self.args = args
        # self.transform = transforms.Compose([
        #     transforms.ToPILImage(), 
        #     transforms.GaussianBlur(7, sigma=(0.1, 2.0)),
        #     transforms.Resize((16*14, 16*14)),
        #     transforms.ToTensor()
        # ])
        DATASET_MAP = {
            'MMAir': self.__init_mmdata
        }
        DATASET_MAP[args['model_name']]()
    
    def __init_mmdata(self):
        featfile = self.args['featPath'] + str(self.mode) + '.pkl'
        with open(featfile, 'rb') as f:
            data = pickle.load(f)

        self.mark = data['mark'] # [N, L, f1]
        self.aqi = data['aqi'] # [N, L, f2]
        self.meo = data['meo'] # [N, L, f3]
        self.photo = data['photo']
        self.label = data['label']
        self.scalers = {
            'pm25': [data['scaler'][0][0], data['scaler'][1][0]],
            'aqi': data['scaler'],
        }
        self.hats = {
            'pm25': data['hat'][...,0], 
            'aqi': data['hat']
        }
        logger.info(f"{self.mode} samples: {self.hats['pm25'].shape[0]}")

    def __len__(self):
        return len(self.hats['pm25'])

    def __getitem__(self, index):
        sample = {
            'index': index,
            'mark': torch.Tensor(self.mark[index]),
            'aqi': torch.Tensor(self.aqi[index]),
            'meo': torch.Tensor(self.meo[index]),
            'photo': torch.Tensor(self.photo[index]),
            'label': torch.Tensor(self.label[index]),
            'hats': {k: torch.Tensor(v[index]) for k, v in self.hats.items()},
        } 
        # # preprocess outphoto
        return sample

def MMDataLoader(args, num_workers):

    datasets = {
        'train': MMDataset(args, mode='train'),
        'valid': MMDataset(args, mode='valid'),
        'test': MMDataset(args, mode='test')
    }

    scalers = datasets['train'].scalers

    dataLoader = {
        ds: DataLoader(datasets[ds],
                    batch_size=args['batch_size'],
                    num_workers=num_workers,
                    shuffle=True)
        for ds in datasets.keys()
    }
    return dataLoader, scalers
