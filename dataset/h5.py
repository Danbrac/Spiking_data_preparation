from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
import torch, h5py
import numpy as np


class HDF5Store(object):
    def __init__(self, datapath, dataset, shape, dtype=np.float32, compression="gzip", chunk_len=1):
        self.datapath = datapath
        self.dataset = dataset
        self.shape = shape
        self.d_counter = [0 for i in range(len(dataset))]
        
        for i in range(len(self.dataset)):
            with h5py.File(self.datapath, mode='a') as h5f:
                self.dset = h5f.require_dataset(
                    self.dataset[i],
                    shape=(0, ) + self.shape[i],
                    maxshape=(None, ) + self.shape[i],
                    dtype=dtype,
                    compression=compression,
                    chunks=(chunk_len, ) + self.shape[i])
                
    def append(self, data, dataset, shape):
        with h5py.File(self.datapath, mode='a') as h5f:
            dataset_id = self.dataset.index(dataset)
            dset = h5f[dataset]
            dset.resize((self.d_counter[dataset_id] + 1, ) + shape)
            dset[self.d_counter[dataset_id]] = [data]
            self.d_counter[dataset_id] += 1

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