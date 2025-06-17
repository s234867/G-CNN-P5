import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms.functional as TF


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


if __name__ == "__main__":
    # Determine project root and data directory relative to this script
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.abspath(os.path.join(base_dir, '..', 'data', 'raw'))

    # Import PCAM loader with resolved path
    from data import load_pcam_subset

    # Load a small subset for quick testing
    try:
        train_loader, _ = load_pcam_subset(
            train_size=16,
            test_size=0,
            data_dir=data_dir,
            seed=42
        )
    except FileNotFoundError as e:
        print(f"Error loading PCAM data: {e}\nPlease check that '{data_dir}' contains the .h5 files.")
        exit(1)

    # Test translate_data
    translated_loader = translate_data(train_loader, translate=(10, 10))
    X_t, y_t = next(iter(translated_loader))
    print(f"Translated batch shapes: images {X_t.shape}, labels {y_t.shape}")

    # Test rotate_data
    rotated_loader = rotate_data(train_loader, angle=45)
    X_r, y_r = next(iter(rotated_loader))
    print(f"Rotated batch shapes: images {X_r.shape}, labels {y_r.shape}")

    # Test symmetri_data (horizontal & vertical flips)
    flipped_loader = symmetri_data(train_loader, horizontal=True, vertical=True)
    X_s, y_s = next(iter(flipped_loader))
    print(f"Flipped batch shapes: images {X_s.shape}, labels {y_s.shape}")
