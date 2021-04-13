import math, cv2
import numpy as np
from scipy.signal import convolve2d
from .base import AbstractPCNN
from .utils import gaussian_kernel
from scipy.ndimage import convolve

class StandardPCNN(AbstractPCNN):
    
    NAME = 'StandardPCNN'
    
    def __init__(self, S, parameters):
        super().__init__(S)
        self.W = gaussian_kernel(7, 1)
        self.M = self.W
        self.S = (self.S - np.min(self.S)) / (np.max(self.S) - np.min(self.S))
        self.a_f = parameters['a_f']
        self.a_t = parameters['a_t']
        self.a_l = parameters['a_l']
        self.v_f = parameters['v_f']
        self.v_t = parameters['v_t']
        self.v_l = parameters['v_l']
        self.beta = parameters['beta']
        
    
    def __repr__(self):
        return 'Standard PCNN Model - Input shape : {}'.format(self.input_dim)
    
    def update_feeding(self) -> None:
        """
        Update the feeding input
        """
        self.F = math.exp(-self.a_f) * self.F + self.v_f * convolve(self.Y, self.M) + self.S
        return
        
    def update_linking(self) -> None:
        """
        Update the linking input
        """
        self.L = math.exp(-self.a_l) * self.L + self.v_l * convolve(self.Y, self.M)
        return
    
    def update_threshold(self) -> None:
        """
        Update the threshold value
        """
        self.T = math.exp(-self.a_t) * self.T + self.v_t * self.Y
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
        self.update_feeding()
        self.update_linking()
        self.compute_internal_activation()
        self.Y = np.where(self.U > self.T, 1, 0)
        self.YY = self.YY + self.Y
        self.update_threshold()
        return self.YY