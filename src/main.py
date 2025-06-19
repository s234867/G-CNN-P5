## main.py ##

import torch
import torch.nn as nn
import numpy as np
import random
from data import load_pcam_subset
from data_augmentation import combine_multiple_augmentations
from MODELS import GECNN, CNN
from steerable import SteerableGCNN
from subfiles.groups import CyclicGroup, DihedralGroup
import statistics
import time
import os
import torchvision.transforms.functional as TF

from save_data import save_results

from sklearn.metrics import f1_score, roc_auc_score

import logging

# Setup logging for when using HPC
log_file = f"training_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Add console handler to also print to terminal
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter("%(message)s")  # Or keep full format if you want
console.setFormatter(formatter)
logging.getLogger().addHandler(console)

from collections import defaultdict

# Collect per-model results
results = defaultdict(list) 




def train_model(model_name, name, model_hparams, optimizer_name, optimizer_hparams, save_name, test_loader, train_loader, n_epochs):
    if model_name == "GCNN":
        model_class = GECNN 
    else:
        model_class = SteerableGCNN  if model_name == "SteerableGCNN" else CNN
    model = model_class(**model_hparams).to(device)

    optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), **optimizer_hparams)
    criterion = nn.CrossEntropyLoss()

    history = {
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'f1': [],
        'auroc': [],
        'epoch_time': []
    }

    for epoch in range(1, n_epochs+1):
        epoch_start = time.time()
        model.train()
        correct, total, loss_sum = 0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)
        train_loss = loss_sum / total
        train_acc = correct / total

        model.eval()
        correct, total, loss_sum = 0, 0, 0
        all_preds, all_targets, all_probs = [], [], []
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(1)
                probs = torch.softmax(out, dim=1)[:, 1]  # for AUROC

                all_preds.append(preds.cpu())
                all_targets.append(y.cpu())
                all_probs.append(probs.cpu())

                loss = criterion(out, y)
                loss_sum += loss.item() * x.size(0)
                correct += (out.argmax(1) == y).sum().item()
                total += x.size(0)
        test_loss = loss_sum / total
        test_acc = correct / total

        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        all_probs = torch.cat(all_probs)

        pred_counts = torch.bincount(all_preds, minlength=2)

        f1 = f1_score(all_targets, all_preds)
        try:
            auroc = roc_auc_score(all_targets, all_probs)
        except ValueError:
            auroc = float('nan')  # Handle if only one class is present

        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['test_loss'].append(test_loss)
        history['test_acc'].append(test_acc)
        history['f1'].append(f1)
        history['auroc'].append(auroc)
        history['epoch_time'].append(round(time.time()-epoch_start, 2))

        logging.info(
            f"Epoch {epoch}: {name} | "
            f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | "
            f"F1: {f1:.4f} | AUROC: {auroc:.4f}"
        )


        logging.info(f"Predicted counts - Class 0: {pred_counts[0].item()}, Class 1: {pred_counts[1].item()}")


        # Debug weights of final linear layer
        if hasattr(model, 'final_linear') and 0: # CHANGE TO 1 IF WANT LINEAR WEIGHTS FOR DEBUG :)
            print("Final linear weights:", model.final_linear.weight.data)


    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), os.path.join("models", save_name + ".pt"))
    return model, history

# Define training augmentations
transforms = [
    lambda img: TF.affine(img, angle=0, translate=(10, 10), scale=1.0, shear=(0,0)),
    lambda img: TF.rotate(img, angle=45),
    lambda img: TF.hflip(TF.vflip(img))
]


# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GENERAL
IN_CHANNELS = 3
OUT_CHANNELS = 2
KERNEL_SIZE = 5

# CNN
CNN_NUM_HIDDEN = 4
CNN_HIDDEN_CHANNELS = 16
CNN_OPTIMIZER = "Adam"
CNN_LR = 1e-2
CNN_WEIGHT_DECAY = 0.0


# STEERABLE
STEERABLE_NUM_HIDDEN = 4
STEERABLE_OPTIMIZER = "Adam"
STEERABLE_LR = 1e-2
STEERABLE_WEIGHT_DECAY = 0.0

# GCNN
GCNN_NUM_HIDDEN = 4
GCNN_GROUP_ORDERS = [4, 8, 16]
BASE_GROUP_ORDER = 4
cyclic_group = CyclicGroup(n=BASE_GROUP_ORDER).to(device)
num_elements = cyclic_group.elements().numel()
HIDDEN_CHANNELS = round(CNN_HIDDEN_CHANNELS/np.log2(num_elements))

GCNN_CYCLIC_OPTIMIZER = "AdamW"
GCNN_CYCLIC_LR = 1e-3
GCNN_CYCLIC_WEIGHT_DECAY = 1e-4

GCNN_DIHEDRAL_OPTIMIZER = "Adam"
GCNN_DIHEDRAL_LR = 1e-2
GCNN_DIHEDRAL_WEIGHT_DECAY = 0.0


# TRAINING STUFF
SEEDS = [42, 15, 67, 1312, 8]
N_TRAIN = 20000
N_TEST = 10000
N_EPOCHS = 30

start_time = time.time()

for SEED in SEEDS:
    logging.info(f"\n==================== SEED {SEED} ====================")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    
    train_loader = None
    test_loader = None

    # Data Loader
    train_loader, test_loader = load_pcam_subset(N_TRAIN, N_TEST, r"./data/raw/", batch_size=32, seed=SEED)

    train_loader_augmented = combine_multiple_augmentations(train_loader, transforms, extend=False)
    ## MODELS
    # CNN
    run_name = f"CNN_{CNN_OPTIMIZER}_lr{CNN_LR}_wd{CNN_WEIGHT_DECAY}"
    cnn_model, cnn_results = train_model(
        model_name="CNN",
        name=run_name,
        model_hparams={"in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS, 
                    "kernel_size": KERNEL_SIZE, "num_hidden": CNN_NUM_HIDDEN, 
                    "hidden_channels": CNN_HIDDEN_CHANNELS},
        optimizer_name=CNN_OPTIMIZER,
        optimizer_hparams={"lr": CNN_LR, "weight_decay": CNN_WEIGHT_DECAY},
        save_name=f"{run_name}-pretrained",
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=N_EPOCHS
        )
    # save results
    results[run_name].append({"seed": SEED, "history": cnn_results})

    # CNN data augment
    run_name = f"CNN-augmented_{CNN_OPTIMIZER}_lr{CNN_LR}_wd{CNN_WEIGHT_DECAY}"
    cnn_model, cnn_results = train_model(
        model_name="CNN",
        name=run_name,
        model_hparams={"in_channels": IN_CHANNELS, "out_channels": OUT_CHANNELS, 
                    "kernel_size": KERNEL_SIZE, "num_hidden": CNN_NUM_HIDDEN, 
                    "hidden_channels": CNN_HIDDEN_CHANNELS},
        optimizer_name=CNN_OPTIMIZER,
        optimizer_hparams={"lr": CNN_LR, "weight_decay": CNN_WEIGHT_DECAY},
        save_name=f"{run_name}-pretrained",
        train_loader=train_loader_augmented,
        test_loader=test_loader,
        n_epochs=N_EPOCHS
        )
    # save results
    results[run_name].append({"seed": SEED, "history": cnn_results})

    # Steerable
    run_name = f"SteerableGCNN_{STEERABLE_OPTIMIZER}_lr{STEERABLE_LR}_wd{STEERABLE_WEIGHT_DECAY}"
    steerablegcnn_model, steerablegcnn_results = train_model(
        model_name="SteerableGCNN",
        name=run_name,
        model_hparams={
            "in_channels": IN_CHANNELS,
            "out_channels": OUT_CHANNELS,
            "kernel_size": KERNEL_SIZE,
            "num_hidden": STEERABLE_NUM_HIDDEN,
            "hidden_channels": HIDDEN_CHANNELS
        },
        optimizer_name=STEERABLE_OPTIMIZER,
        optimizer_hparams={"lr": STEERABLE_LR, "weight_decay": STEERABLE_WEIGHT_DECAY},
        save_name=f"{run_name}-pretrained",
        train_loader=train_loader,
        test_loader=test_loader,
        n_epochs=N_EPOCHS
        )
    # save results
    results[run_name].append({"seed": SEED, "history": steerablegcnn_results})

    for order in GCNN_GROUP_ORDERS:
        # GCNN
        cyclic_name = "GCNN_CYCLIC_" + str(order) + f"_{GCNN_CYCLIC_OPTIMIZER}_lr{GCNN_CYCLIC_LR}_wd{GCNN_CYCLIC_WEIGHT_DECAY}"
        cyclic_group = CyclicGroup(n=order).to(device)

        dihedral_name = "GCNN_DIHEDRAL_" + str(order) + f"_{GCNN_DIHEDRAL_OPTIMIZER}_lr{GCNN_DIHEDRAL_LR}_wd{GCNN_DIHEDRAL_WEIGHT_DECAY}"
        dihedral_group = DihedralGroup(n=order).to(device)

        num_elements = cyclic_group.elements().numel()
        HIDDEN_CHANNELS = round(CNN_HIDDEN_CHANNELS/np.log2(num_elements))

        # CYCLIC
        gcnn_model, gcnn_results = train_model(
            model_name="GCNN",
            name=cyclic_name,
            model_hparams={"in_channels": IN_CHANNELS, 
                        "out_channels": OUT_CHANNELS, 
                        "kernel_size": KERNEL_SIZE, 
                        "num_hidden": GCNN_NUM_HIDDEN,
                        "hidden_channels": HIDDEN_CHANNELS, 
                        "group":cyclic_group},
            optimizer_name=GCNN_CYCLIC_OPTIMIZER,
            optimizer_hparams={"lr": GCNN_CYCLIC_LR, "weight_decay": GCNN_CYCLIC_WEIGHT_DECAY},
            save_name=f"{cyclic_name}-pretrained",
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=N_EPOCHS
            )
        # save results
        results[cyclic_name].append({"seed": SEED, "history": gcnn_results})

        # DIHEDRAL
        gcnn_model, gcnn_results = train_model(
            model_name="GCNN",
            name=dihedral_name,
            model_hparams={"in_channels": IN_CHANNELS, 
                        "out_channels": OUT_CHANNELS, 
                        "kernel_size": KERNEL_SIZE, 
                        "num_hidden": GCNN_NUM_HIDDEN,
                            "hidden_channels": HIDDEN_CHANNELS, 
                            "group":dihedral_group},
            optimizer_name=GCNN_DIHEDRAL_OPTIMIZER,
            optimizer_hparams={"lr": GCNN_DIHEDRAL_LR, "weight_decay": GCNN_DIHEDRAL_WEIGHT_DECAY},
            save_name=f"{dihedral_name}-pretrained",
            train_loader=train_loader,
            test_loader=test_loader,
            n_epochs=N_EPOCHS
            )
        # save results
        results[dihedral_name].append({"seed": SEED, "history": gcnn_results})

end_time = time.time()

took = end_time - start_time
logging.info(f"Done in {took:.2f} seconds")

import statistics

logging.info("\n===== RESULTS SUMMARY =====")

for model_name, histories in results.items():
    f1_scores = [max(h["history"]["f1"]) for h in histories]
    aurocs = [h["history"]["auroc"][np.argmax(h["history"]["f1"])] for h in histories]  # AUROC at best-F1 epoch

    f1_mean = statistics.mean(f1_scores)
    f1_std = statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0.0
    auroc_mean = statistics.mean(aurocs)
    auroc_std = statistics.stdev(aurocs) if len(aurocs) > 1 else 0.0

    logging.info(
        f"{model_name} | "
        f"F1: {f1_mean:.4f} ± {f1_std:.4f} | "
        f"AUROC: {auroc_mean:.4f} ± {auroc_std:.4f}"
    )

save_results(results)

