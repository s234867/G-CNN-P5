import h5py
import numpy as np
import os
import random
import torch

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

def load_pcam_subset(train_size, test_size):
    data_dir='./data/raw/'

    # Reproducibility which is VERY important
    random.seed(42)
    np.random.seed(42)


    # Paths for the data we have on pc
    train_x_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_train_x.h5')
    train_y_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_train_y.h5')
    test_x_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_x.h5')
    test_y_path = os.path.join(data_dir, 'camelyonpatch_level_2_split_valid_y.h5')

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

    # Squeeze the labels so its [500] instead of [500, 1], also convert it to an int
    y_train = torch.from_numpy(y_train).squeeze().long()
    y_test = torch.from_numpy(y_test).squeeze().long()

    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_pcam_subset(train_size=500, test_size=500)
    print("Train:", X_train.shape, y_train.shape)
    print("Test: ", X_test.shape, y_test.shape)
