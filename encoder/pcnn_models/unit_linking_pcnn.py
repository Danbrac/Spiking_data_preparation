import math, cv2
import numpy as np
from .base import AbstractPCNN
from .utils import gaussian_kernel
from scipy.ndimage import convolve

class UnitLinkingPCNN(AbstractPCNN):
    
    NAME = "UnitLinkingPCNN"
    
    def __init__(self, S, parameters):
        super().__init__(S)
        self.S = cv2.normalize(self.S.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX) 
        self.F = self.S
        self.kernel = gaussian_kernel(parameters['k_size'])
        self.a_t = parameters['a_t']
        self.v_t = parameters['v_t']
        self.beta = parameters['beta']
    
    def update_linking(self) -> None:
        """
        Update the linking input
        """
        Y_sum = convolve(self.Y.astype('float'), self.kernel)
        self.L = np.where(Y_sum > 0, 1, 0)
        return
    
    def update_threshold(self) -> None:
        """
        Update the threshold value
        """
        self.T = self.T + (-self.a_t + self.v_t * self.Y)
        return
    
    def compute_internal_activation(self) -> None:
        """
        Compute the internal activation 
        """
        self.U = self.F * (1 + self.beta * self.L)
        return
        
    def do_iteration(self) -> None:
        """
        Run PCNN model
        """
        self.update_threshold()
        self.update_linking()
        self.compute_internal_activation()
        self.Y = np.where(self.U > self.T, 1, 0)
        self.YY = self.YY + self.Y
        self.update_threshold()
        return self.YY
    
class ULPCNN_Signature(UnitLinkingPCNN):
    
    NAME = "ULPCNN_Signature"
    
    def __init__(self, S, parameters):
        super().__init__(S, parameters)
        self.signature = []
        
    def do_iteration(self) -> None:
        total_count = 0
        n = 1
        while total_count != self.S.shape[0] * 2:
            #self.update_threshold()
            self.update_linking()
            self.compute_internal_activation()
            self.Y = np.where(self.U > self.T, 1, 0)
            self.YY = self.YY + self.Y
            total_count += self.Y.sum()
            self.signature.append(self.Y.sum())
            self.update_threshold()
            n += 1