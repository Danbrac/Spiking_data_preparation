from .h5 import H5Dataset
from PCNN_Models.models import FL_Signature, mPCNN
from processing import smooth, BSAEncoder
from tqdm import tqdm
from scipy import signal
import cv2
import numpy as np

class SpikingDataset(H5Dataset):
    
    def __init__(self, f_path, length=None):
        super().__init__(f_path, length)
        self.pcnn_parameters = [{"a_T": 0.015, "v_T": 2, "beta": [0.8, 0.4, 0.8, 0.4], "lvl_factor": 1.3},
                            {"v_t": 20,"f": 0.001,"beta": 0.1}]
        self.pcnn_models = [mPCNN, FL_Signature]

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
            Image signature.
        '''
        d = np.swapaxes(x, 0, 2)
        d1, d2, d3, d4 = d
        
        fuse_model = self.pcnn_models[0]([d1, d2, d3, d4], self.pcnn_parameters[0])
        fuse_model.do_iteration()
        
        encoding_model = self.pcnn_models[1](fuse_model.U, self.pcnn_parameters[1])
        encoding_model.do_iteration()
        
        return np.array(encoding_model.signature)
    
    def spike_train_gen(self):
        '''Takes an image signature and convert it to spike train using
        the Ben's spiker algorithm (BSA). The signature is first smoothed
        using a moving average window method. Parameters were found using
        differential evolution.
        
        Yields
        ------
        np.array
            Spike train of given image signature
        '''
        win_size, mean, std, amp, threshold, step = [27, 21, 5,  0.1,  1.1 , 1]
        
        with tqdm(total=len(self.data)) as pbar:
            for x, y in zip(self.data, self.target):
                x_sign = self.image_to_signature(x)
                x = smooth(x_sign, win_size)
                x = np.squeeze(cv2.normalize(x, None, 0.0, 
                                                    1.0, cv2.NORM_MINMAX))[:len(x_sign)]
                bsa = BSAEncoder(filter_response=signal.gaussian(M=mean, std=std), 
                            step=step, filter_amp=amp, threshold=threshold)
                yield bsa.encode(x), y
                pbar.update()
                