import torch, h5py, cv2
import numpy as np
from .pcnn_models import mPCNN, FL_Signature
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils import shuffle
from utils.signal import smooth

class ImageSignatureEncoder(torch.utils.data.Dataset):
    
    def __init__(self, f_path, smooth=False, length=None):
        super().__init__()
        if torch.cuda.is_available():
            self.device     = torch.device("cuda")     
        else:
            self.device     = torch.device("cpu")

        self.f_path = f_path
        self.length = length
        self.smooth = smooth
        self.data, self.target = self._load_data(self.f_path)
        self.data_shapes = (self.data[0].shape, self.target[0].shape)
        self.pcnn_models = [mPCNN, FL_Signature]
        self.pcnn_parameters = [{
                "a_T": 0.015, "v_T": 2, 
                "beta": [0.8, 0.4, 0.8, 0.4], 
                "lvl_factor": 1.3
            },
            {
                "v_t": 20,
                "f": 0.001,
                "beta": 0.1
            }
        ]
        self.sign_shape = self.image_to_signature(self.data[0]).shape

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data, target = self.data[idx], self.target[idx]
        return data, target
    
    def image_to_signature(self, x):
        '''Iterates through the patch image dataset and generates signature
        using PCNN models for fusion and spike counting.
        
        Parameters
        ----------
        x : np.array of shape (p_size, p_size, dim)
            Patch image
        
        Returns
        ------
        np.array
            Image signature, smoothed and normalized in [0, 1].
        '''
        d = np.swapaxes(x, 0, 2)
        d1, d2, d3, d4 = d
        
        fuse_model = self.pcnn_models[0]([d1, d2, d3, d4], self.pcnn_parameters[0])
        fuse_model.do_iteration()
        
        encoding_model = self.pcnn_models[1](fuse_model.U, self.pcnn_parameters[1])
        encoding_model.do_iteration()
        
        x = np.array(encoding_model.signature)
        if self.smooth:
            x = smooth(x, 27)
            x = np.squeeze(cv2.normalize(x, None, 0.0, 1.0, cv2.NORM_MINMAX))
            
        return  x
    
    def _load_data(self, file_path):
        '''Load an h5 file and returns the datasets it contains.
        A random under sampler is used to prevent imbalance between labels.
        
        Parameters
        ----------
        file_path : str
            Path to h5 file to load
        
        Returns
        ------
        data : np.array
            Dataset containing the X dataset
        targets : np.array
            Dataset containing the Y dataset
        '''
        f = h5py.File(file_path, 'r')
        if self.length:
            data = f['x'][:self.length]
            targets = f['y'][:self.length].astype('uint8')
        else:
            data = f['x'][:]
            targets = f['y'][:].astype('uint8')
        print('\n## Original dataset shape {}'.format(Counter(targets)))
        ros = RandomUnderSampler()
        d_shape = data.shape
        data = np.reshape(data, (d_shape[0], (d_shape[1]**2)*4))

        data, targets = ros.fit_sample(data, targets)
        data = np.reshape(data, (data.shape[0], d_shape[1], d_shape[1], 4))
        data, targets = shuffle(data, targets, random_state=0)        
        print('## Resampled dataset shape {}'.format(Counter(targets)))
        print('## Dataset shape : {}\n'.format(data.shape))
        return data, targets