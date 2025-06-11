import h5py
import numpy as np
import os
import random
import torch
from torch.utils.data import TensorDataset, DataLoader


def set_seed(seed=42):
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

def load_pcam_subset(train_size, test_size, data_dir=r'./data/raw', seed=42):
    # Reproducibility which is VERY important
    set_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)

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

    print("Train label distribution:", torch.bincount(y_train))
    print("Test label distribution:", torch.bincount(y_test))

    train_set = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True, generator=g)
    test_set = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    

    return train_set, test_set

if __name__ == "__main__":
    train_set, test_set = load_pcam_subset(train_size=100, test_size=100, data_dir=r"D:\GCNN\G-CNN-P5\data\raw")

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

