from abc import ABC, abstractmethod
import numpy as np
import cv2


class AbstractPCNN(ABC):
    
    def __init__(self, S, **kwargs):
        super().__init__()
        self.S = cv2.normalize(S, None, alpha = 0, beta = 255, 
                            norm_type = cv2.NORM_MINMAX, 
                            dtype = cv2.CV_32F).astype(np.uint8)
        self.input_dim = self.S.shape
        self.F = np.zeros(self.input_dim)
        self.L, self.Y, self.U = self.F, self.F, self.F
        self.T = np.ones(self.input_dim)
        self.YY = self.Y
        
    @abstractmethod
    def update_linking(self) -> None:
        """
        Update the linking input
        """
        raise NotImplementedError
        
    @abstractmethod
    def update_threshold(self) -> None:
        """
        Update the threshold value
        """
        raise NotImplementedError
        
    @abstractmethod
    def compute_internal_activation(self) -> None:
        """
        Compute the internal activation 
        """
        raise NotImplementedError
        
    @abstractmethod
    def do_iteration(self) -> None:
        raise NotImplementedError