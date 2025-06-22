## final_tests.py ##

import torch
import numpy as np
from data import load_pcam_subset_for_final_testing
from MODELS import GECNN, CNN
#from steerable import SteerableGCNN
SteerableGCNN = None
from subfiles.groups import CyclicGroup, DihedralGroup
import os
import csv
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, log_loss

from sklearn.model_selection import KFold
from torch.utils.data import Subset, DataLoader

def create_kfold_loaders_from_loader(test_loader, k=5, batch_size=32, seed=42):
    dataset = test_loader.dataset  # this is a TensorDataset
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (_, test_idx) in enumerate(kf.split(dataset)):
        test_subset = Subset(dataset, test_idx)
        test_fold_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
        folds.append((fold_idx, test_fold_loader))
    return folds

# remove old file to avoid duplicates
if os.path.exists("confusion_results.csv"):
    os.remove("confusion_results.csv")

SEED = 67
N_TRAIN = 20000
N_TEST = 10000
# Data Loader
test_loader = load_pcam_subset_for_final_testing(N_TRAIN, N_TEST, r"./data/raw/", batch_size=32, seed=SEED, normalize=False, test_x_path='final_test_x.h5', test_y_path='final_test_y.h5')

# create the folds for cross validation
folds = create_kfold_loaders_from_loader(test_loader, k=10, batch_size=32, seed=SEED)

# get the models (instead of just hardcoding the names)
model_dir = "models"
model_files = [f for f in os.listdir(model_dir) if f.endswith(".pt")]

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GENERAL
IN_CHANNELS = 3
OUT_CHANNELS = 2
KERNEL_SIZE = 5

# CNN
CNN_NUM_HIDDEN = 4
CNN_HIDDEN_CHANNELS = 16

# STEERABLE
STEERABLE_NUM_HIDDEN = 4
BASE_GROUP_ORDER = 4
cyclic_group = CyclicGroup(n=BASE_GROUP_ORDER).to(device)
num_elements = cyclic_group.elements().numel()
HIDDEN_CHANNELS = round(CNN_HIDDEN_CHANNELS/np.log2(num_elements))

# GCNN
GCNN_NUM_HIDDEN = 4

for model in model_files:
    model_name = model.replace("-pretrained.pt", "")
    model_path = os.path.join(model_dir, model)

    print(f"Evaluating model: {model_name}")

    # Instantiate model based on name
    if "CNN" in model_name and "GCNN" not in model_name:
        model_instance = CNN(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            num_hidden=CNN_NUM_HIDDEN,
            hidden_channels=CNN_HIDDEN_CHANNELS
        ).to(device)

    elif "GCNN_CYCLIC" in model_name:
        n = int(model_name.split("_")[2].replace(".pt", ""))
        group = CyclicGroup(n=n).to(device)
        hidden_channels = round(CNN_HIDDEN_CHANNELS / np.log2(group.elements().numel()))
        model_instance = GECNN(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            num_hidden=GCNN_NUM_HIDDEN,
            hidden_channels=hidden_channels,
            group=group
        ).to(device)

    elif "GCNN_DIHEDRAL" in model_name:
        n = int(model_name.split("_")[2].replace(".pt", ""))
        group = DihedralGroup(n=n).to(device)
        state_dict = torch.load(model_path, map_location=device)
        hidden_channels = state_dict["lifting_conv.bias"].shape[0]
        model_instance = GECNN(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            num_hidden=GCNN_NUM_HIDDEN,
            hidden_channels=hidden_channels,
            group=group
        ).to(device)

    elif "SteerableGCNN" in model_name:
        state_dict = torch.load(model_path, map_location=device)
        hidden_channels_steerable = state_dict["classifier.weight"].shape[1]
        model_instance = SteerableGCNN(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            num_hidden=STEERABLE_NUM_HIDDEN,
            hidden_channels=hidden_channels_steerable
        ).to(device)


    else:
        print(f"Unknown model type for: {model_name}")
        continue

    # Load pretrained weights
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    # Run over all folds
    for fold_idx, fold_loader in folds:
        print(f"Fold {fold_idx}...")

        all_preds, all_targets, all_probs = [], [], []

        with torch.no_grad():
            for x, y in fold_loader:
                x, y = x.to(device), y.to(device)
                logits = model_instance(x)
                preds = logits.argmax(dim=1)
                probs = torch.softmax(logits, dim=1)[:, 1]  # Prob for class 1

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                all_probs.append(probs.cpu())

        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        all_probs = torch.cat(all_probs).numpy()

        # Metrics
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds)
        try:
            auroc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auroc = float('nan')
        try:
            loss = log_loss(all_targets, all_probs)
        except ValueError:
            loss = float('nan')

        # Save results
        file_exists = os.path.exists("confusion_results.csv")
        with open("confusion_results.csv", mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["Model", "Seed", "Fold", "Accuracy", "F1", "AUROC", "LogLoss", "Preds", "Targets"])
            writer.writerow([
                model_name,
                SEED,
                fold_idx,
                round(acc, 4),
                round(f1, 4),
                round(auroc, 4),
                round(loss, 4),
                " ".join(map(str, all_preds)),
                " ".join(map(str, all_targets))
            ])
