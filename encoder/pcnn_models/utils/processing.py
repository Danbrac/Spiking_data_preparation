import scipy.ndimage.filters as fi
import numpy as np

def gaussian_kernel(size=5, sigma=1.):
	if size % 2 == 0:
		size += 1
	k = np.zeros((size,size))
	k[int(size/2), int(size/2)] = 1
	k = fi.gaussian_filter(k, sigma)
	return k