# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Subset

# Torchvision
import torchvision
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl



SEED = 42

# Set the random seed for reproducibility.
pl.seed_everything(SEED)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False



def check_files(data_dir="data/pcam") -> None:
    required_files = [
        "camelyonpatch_level_2_split_train_meta.csv",
        "camelyonpatch_level_2_split_train_x.h5",
        "camelyonpatch_level_2_split_train_y.h5",
        "camelyonpatch_level_2_split_test_meta.csv",
        "camelyonpatch_level_2_split_test_x.h5",
        "camelyonpatch_level_2_split_test_y.h5",
        "camelyonpatch_level_2_split_valid_meta.csv",
        "camelyonpatch_level_2_split_valid_x.h5",
        "camelyonpatch_level_2_split_valid_y.h5",
    ]

    missing_files = [f for f in required_files if not os.path.isfile(os.path.join(data_dir, f))]

    assert not missing_files, f"Missing files in {FULL_PATH}: {missing_files}"



# Normalize training data
train_transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # RGB
])

# Roto-reflected test transforms
test_transform = transforms.Compose([
    torchvision.transforms.ToTensor(),
    transforms.RandomRotation(
        degrees=[0, 360],
        interpolation=transforms.InterpolationMode.BILINEAR, # Ensure idenitcal interpolation method
        fill=0
    ),
    transforms.RandomHorizontalFlip(),
    transforms.Normalize((0.5,), (0.5,))
])


# Dataset
train_dataset = torchvision.datasets.PCAM(root=data_dir, split="train", transform=train_transform, download=False)
test_dataset = torchvision.datasets.PCAM(root=data_dir, split="test", transform=train_transform, download=False)
validation_dataset = torchvision.datasets.PCAM(root=data_dir, split="val", transform=train_transform, download=False)

### Sampler
train_samples = 1000
test_samples = 200
val_samples = 200

rs_train = torch.utils.data.RandomSampler(train_dataset, num_samples=train_samples)
subset_test_dataset = Subset(test_dataset, list(range(test_samples)))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, sampler=rs_train)
test_loader = torch.utils.data.DataLoader(subset_test_dataset, batch_size=16, shuffle=False)
