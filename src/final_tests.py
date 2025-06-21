## final_tests.py ##

import torch
import numpy as np
from data import load_pcam_subset_for_final_testing
from MODELS import GECNN, CNN
from steerable import SteerableGCNN
#SteerableGCNN = None
from subfiles.groups import CyclicGroup, DihedralGroup
import os
import csv



# remove old file to avoid duplicates
if os.path.exists("confusion_results.csv"):
    os.remove("confusion_results.csv")


SEED = 67
N_TRAIN = 20000
N_TEST = 1000
# Data Loader
test_loader = load_pcam_subset_for_final_testing(N_TRAIN, N_TEST, r"./data/raw/", batch_size=32, seed=SEED, normalize=False, test_x_path='final_test_x.h5', test_y_path='final_test_y.h5')

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

    print(f"Evaluating: {model_name}")
    
    # simple if's to get the correct model
    if "CNN" in model_name and "GCNN" not in model_name:
        model_class = CNN
        model_instance = model_class(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            num_hidden=CNN_NUM_HIDDEN,
            hidden_channels=CNN_HIDDEN_CHANNELS
        ).to(device)

    elif "GCNN_CYCLIC" in model_name:
        n = int(model_name.split("_")[2].replace(".pt", ""))
        group = CyclicGroup(n=n).to(device)
        num_elements = group.elements().numel()
        hidden_channels = round(CNN_HIDDEN_CHANNELS / np.log2(num_elements))  # MATCHES TRAINING

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
        hidden_channels_steerable = state_dict["classifier.weight"].shape[1]  # for fixing
        model_class = SteerableGCNN
        model_instance = model_class(
            in_channels=IN_CHANNELS,
            out_channels=OUT_CHANNELS,
            kernel_size=KERNEL_SIZE,
            num_hidden=STEERABLE_NUM_HIDDEN,
            hidden_channels=hidden_channels_steerable
        ).to(device)

    else:
        print(f"Unknown model type for: {model_name}")
        continue

    # Load model weights
    model_instance.load_state_dict(torch.load(model_path, map_location=device))
    model_instance.eval()

    # Run inference
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            out = model_instance(x)
            preds = out.argmax(dim=1)
            all_preds.append(preds.cpu())
            all_targets.append(y.cpu())

    # Flatten - apparently important for dimensions
    all_preds = torch.cat(all_preds).tolist()
    all_targets = torch.cat(all_targets).tolist()

    # Save to CSV
    confusion_csv_path = "confusion_results.csv"
    write_header = not os.path.exists(confusion_csv_path)

    with open(confusion_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Model", "Seed", "Epoch", "Preds", "Targets"])
            write_header = False  # prevent repeating header

        writer.writerow([
            model_name,
            SEED,
            "final",
            " ".join(map(str, all_preds)),
            " ".join(map(str, all_targets))
        ])