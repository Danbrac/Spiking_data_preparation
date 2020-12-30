from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
import torch, h5py
import numpy as np


class H5Dataset(torch.utils.data.Dataset):
    
    def __init__(self, f_path, length=None):
        super().__init__()
        if torch.cuda.is_available():
            self.device     = torch.device("cuda")     
        else:
            self.device     = torch.device("cpu")

        self.f_path = f_path
        self.length = length
        self.data, self.target = self._load_data(self.f_path)
        self.dataset_shape = (self.data[0].shape, self.target[0].shape)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, target = self.data[idx], self.target[idx]
        return data, target

    def _load_data(self, file_path):
        f = h5py.File(file_path, 'r')
        if self.length:
            data = f['x'][:self.length]
            targets = f['y'][:self.length].astype('uint8')
        else:
            data = f['x'][:]
            targets = f['y'][:].astype('uint8')
        data, targets = shuffle(data, targets, random_state=0)        
        print('\nLabel balance {}'.format(Counter(targets)))
        print('Data shape : {}\n'.format(data.shape))
        return data, targets