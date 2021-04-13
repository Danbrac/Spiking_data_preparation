import h5py
from encoder import ImageSignatureEncoder
from utils import HDF5Store
from tqdm import tqdm


if __name__ == "__main__":

    data_path = './data/train_patch_all_5_1617970687.153493.h5'
    save_path = './data/signature_data.h5'
    
    # 1. Load Tensor dataset
    sign_dataset = ImageSignatureEncoder(data_path, smooth=False, length=None)
    
    # 2. Create H5 store to save signature
    store = HDF5Store('./data/signature_data.h5',
                    ['x', 'y'], 
                    [sign_dataset.sign_shape, 
                    sign_dataset.target[0].shape])
    
    # 3. Iterate through torch dataset and append to h5 file
    with tqdm(total=len(sign_dataset.data)) as pbar:
        for i, j in sign_dataset:
            sign = sign_dataset.image_to_signature(i)
            store.append(sign, 'x', sign.shape)
            store.append(j, 'y', j.shape)
            pbar.update()