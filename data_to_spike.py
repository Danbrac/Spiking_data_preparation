import h5py
from utils import HDF5Store
from encoder import ImageSignatureEncoder


if __name__ == "__main___":

    data_path = ''
    save_path = ''
    
    # 1. Load patch dataset
    patch_data = h5py.File(data_path, 'r')
    
    # 2. Prepare H5 store to save the generated data
    

    