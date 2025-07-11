## data.py ##

import h5py
import numpy as np
import os
import random
import torch
from torch.utils.data import TensorDataset, DataLoader


# remove the annoying user warning - internet says the code still runs correctly lul
# C:\Users\Christian\AppData\Local\Programs\Python\Python312\Lib\site-packages\e2cnn\nn\modules\r2_conv\basisexpansion_singleblock.py:80: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at C:\actions-runner\_work\pytorch\pytorch\builder\windows\pytorch\aten\src\ATen/native/IndexingUtils.h:28.)
import warnings
warnings.filterwarnings("ignore", category=UserWarning)



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def sample_indices(dataset_path, size):
    with h5py.File(dataset_path, 'r') as f:
        dataset = f[list(f.keys())[0]]
        total = len(dataset)
        indices = np.random.choice(total, size=size, replace=False)
    return indices

def load_samples(dataset_path, indices):
    with h5py.File(dataset_path, 'r') as f:
        dataset = f[list(f.keys())[0]]
        indices_sorted = np.sort(indices)
        data_sorted = dataset[indices_sorted]
    # Annoying trick we have to do, in order to reorder to match original random order
    reorder = np.argsort(np.argsort(indices))
    return data_sorted[reorder]

def load_pcam_subset(train_size, test_size, data_dir=r'./data/raw', seed=42, batch_size=32):
    # Reproducibility which is VERY important
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Paths for the data we have on pc
    train_x_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    train_y_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')
    test_x_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_test_x.h5')
    test_y_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_test_y.h5')

    # Sample indices without loading full data which is a trick we use to only load the data we need
    # and NOT load the entire dataset (which requires large amounts of memory)
    train_indices = sample_indices(train_x_path, train_size)
    test_indices = sample_indices(test_x_path, test_size)

    # Now we load only the selected samples
    X_train = load_samples(train_x_path, train_indices)
    y_train = load_samples(train_y_path, train_indices)
    X_test = load_samples(test_x_path, test_indices)
    y_test = load_samples(test_y_path, test_indices)

    # Convert to torch tensors, since torch expect difference shape [N, H, W, C] -> [N, C, H, W]
    # Also we normalize
    X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).float() / 255.0 
    X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).float() / 255.0

    # Add normalization (channel-wise)
    mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
    std = X_train.std(dim=(0, 2, 3), keepdim=True)

    # Apply to both train and test
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Squeeze the labels so its [500] instead of [500, 1], also convert it to an int
    y_train = torch.from_numpy(y_train).squeeze().long()
    y_test = torch.from_numpy(y_test).squeeze().long()

    print("Train label distribution:", torch.bincount(y_train))
    print("Test label distribution:", torch.bincount(y_test))

    train_set = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, generator=g)
    test_set = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    

    return train_set, test_set


def load_pcam_subset_for_final_testing(train_size, test_size, data_dir=r'./data/raw', seed=67, batch_size=32, normalize=True, 
                                       test_x_path='camelyonpatch_level_2_split_train_x.h5', 
                                       test_y_path='camelyonpatch_level_2_split_train_y.h5'):
    # Reproducibility which is VERY important
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Paths for the data we have on pc
    test_x_path = os.path.join(data_dir, test_x_path)
    test_y_path = os.path.join(data_dir, test_y_path)

    # Sample indices without loading full data which is a trick we use to only load the data we need
    # and NOT load the entire dataset (which requires large amounts of memory)
    test_indices = sample_indices(test_x_path, test_size)

    # Now we load only the selected samples
    
    X_test = load_samples(test_x_path, test_indices)
    y_test = load_samples(test_y_path, test_indices)

    # Convert to torch tensors, since torch expect difference shape [N, H, W, C] -> [N, C, H, W]
    # Also we normalize
    X_test = torch.from_numpy(X_test).permute(0, 3, 1, 2).float()

    if normalize:
        X_test =  X_test / 255.0
        # moved everything down here, so if no normalization we dont have to import the entire big dataset.. SMAAAAAAAAAAAAAAART
        train_x_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
        train_y_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')

        train_indices = sample_indices(train_x_path, train_size)

        X_train = load_samples(train_x_path, train_indices)
        y_train = load_samples(train_y_path, train_indices)

        X_train = torch.from_numpy(X_train).permute(0, 3, 1, 2).float() / 255.0 

        # Add normalization (channel-wise)
        mean = X_train.mean(dim=(0, 2, 3), keepdim=True)
        std = X_train.std(dim=(0, 2, 3), keepdim=True)

        # Apply to both train and test
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

    # Squeeze the labels so its [500] instead of [500, 1], also convert it to an int
    y_test = torch.from_numpy(y_test).squeeze().long()

    print("Test label distribution:", torch.bincount(y_test))

    test_set = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

    return test_set


import os
import h5py
import torch
from data import load_pcam_subset_for_final_testing

def save_final_testset_split(x_save_path, y_save_path, n_train=20000, n_test=1000, data_dir="./data/raw", seed=67, batch_size=32):
    # Load test set
    test_loader = load_pcam_subset_for_final_testing(n_train, n_test, data_dir=data_dir, seed=seed, batch_size=batch_size)

    all_images = []
    all_labels = []

    for x, y in test_loader:
        all_images.append(x)
        all_labels.append(y)

    # Stack into tensors
    all_images = torch.cat(all_images, dim=0).cpu().numpy()  # shape: (N, 3, H, W)
    all_images = np.transpose(all_images, (0, 2, 3, 1))  # fix for HDF5 saving
    all_labels = torch.cat(all_labels, dim=0).cpu().numpy()  # shape: (N,)

    # Save images to x.h5
    with h5py.File(x_save_path, "w") as f:
        f.create_dataset("x", data=all_images, compression="gzip")
    print(f"Saved images to {x_save_path}")

    # Save labels to y.h5
    with h5py.File(y_save_path, "w") as f:
        f.create_dataset("y", data=all_labels, compression="gzip")
    print(f"Saved labels to {y_save_path}")


if __name__ == "__main__":
    if 1:
        x_path = "./data/raw/final_test_x.h5"
        y_path = "./data/raw/final_test_y.h5"
        save_final_testset_split(x_path, y_path)
    train_set, test_set = load_pcam_subset(train_size=100, test_size=100, data_dir=r"/zhome/d1/3/206707/Desktop/G-CNN-P5/data/raw/")

    # find device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup model to use device (important)
    # model.to(device)
    for X_batch, y_batch in train_set:
        print(X_batch.shape, y_batch.shape)
        # when training and testing, make sure data is sent to device
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        break

    for X_batch, y_batch in test_set:
        print(X_batch.shape, y_batch.shape)
        break

