import h5py
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