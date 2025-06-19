import os
import random
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TF

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def translate_data(dataloader_set, translate=(5, 5)):
    device = next(iter(dataloader_set))[0].device
    imgs, labels = [], []
    for X_batch, y_batch in dataloader_set:
        for img, label in zip(X_batch, y_batch):
            translated = TF.affine(img, angle=0, translate=translate, scale=1.0, shear=(0.0, 0.0))
            imgs.append(translated)
            labels.append(label)
    imgs = torch.stack(imgs).to(device)
    labels = torch.stack(labels).to(device)
    return DataLoader(TensorDataset(imgs, labels), batch_size=dataloader_set.batch_size, shuffle=True)

def rotate_data(dataloader_set, angle=30):
    device = next(iter(dataloader_set))[0].device
    imgs, labels = [], []
    for X_batch, y_batch in dataloader_set:
        for img, label in zip(X_batch, y_batch):
            rotated = TF.rotate(img, angle)
            imgs.append(rotated)
            labels.append(label)
    imgs = torch.stack(imgs).to(device)
    labels = torch.stack(labels).to(device)
    return DataLoader(TensorDataset(imgs, labels), batch_size=dataloader_set.batch_size, shuffle=True)

def symmetri_data(dataloader_set, horizontal=True, vertical=False):
    device = next(iter(dataloader_set))[0].device
    imgs, labels = [], []
    for X_batch, y_batch in dataloader_set:
        for img, label in zip(X_batch, y_batch):
            out = img
            if horizontal:
                out = TF.hflip(out)
            if vertical:
                out = TF.vflip(out)
            imgs.append(out)
            labels.append(label)
    imgs = torch.stack(imgs).to(device)
    labels = torch.stack(labels).to(device)
    return DataLoader(TensorDataset(imgs, labels), batch_size=dataloader_set.batch_size, shuffle=True)

def combine_multiple_augmentations(loader, transform_fns, extend=True):
    device = next(iter(loader))[0].device
    batch_size = loader.batch_size
    original_imgs, original_labels = [], []

    for X_batch, y_batch in loader:
        for img, label in zip(X_batch, y_batch):
            original_imgs.append(img)
            original_labels.append(label)

    # the good agumentations
    augmented_imgs, augmented_labels = [], []
    for img, label in zip(original_imgs, original_labels):
        for fn in transform_fns:
            augmented_imgs.append(fn(img))
            augmented_labels.append(label)

    # combine the data
    all_imgs = original_imgs + augmented_imgs
    all_labels = original_labels + augmented_labels

    if not extend:
        # we randomly sample to keep same size of dataset
        idx = torch.randperm(len(all_imgs))[:len(original_imgs)]
        all_imgs = [all_imgs[i] for i in idx]
        all_labels = [all_labels[i] for i in idx]

    # do conversions back and return the augmented dataset
    all_imgs = torch.stack(all_imgs).to(device)
    all_labels = torch.tensor(all_labels).to(device)

    return DataLoader(TensorDataset(all_imgs, all_labels), batch_size=batch_size, shuffle=True)




if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data', 'raw'))

    from data import load_pcam_subset

    try:
        train_loader, _ = load_pcam_subset(
            train_size=32,
            test_size=10,
            data_dir=data_dir,
            seed=42
        )
    except FileNotFoundError as e:
        print(f"Error loading PCAM data: {e}\nPlease check that '{data_dir}' contains the .h5 files.")
        exit(1)

    transforms = [
        lambda img: TF.affine(img, angle=0, translate=(10, 10), scale=1.0, shear=(0.0, 0.0)),
        lambda img: TF.rotate(img, angle=45),
        lambda img: TF.hflip(TF.vflip(img))
    ]

    extended_loader = combine_multiple_augmentations(train_loader, transforms)

    X_ext, y_ext = next(iter(extended_loader))
    print(f"Extended batch shapes: images {X_ext.shape}, labels {y_ext.shape}")
