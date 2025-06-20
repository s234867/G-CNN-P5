import csv
import os

def save_results(results, csv_path="results.csv", confusion_csv_path="preds.csv"):
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        if write_header:
            writer.writerow([
                "Model", "Seed", "Epoch",
                "Train_Loss", "Test_Loss",
                "Train_Acc", "Test_Acc",
                "F1", "AUROC", "Epoch_time"
            ])

        for model_name, histories in results.items():
            for entry in histories:
                seed = entry["seed"]
                history = entry["history"]
                for epoch, (train_loss, test_loss, train_acc, test_acc, f1, auroc, epoch_time) in enumerate(
                    zip(
                        history['train_loss'], history['test_loss'],
                        history['train_acc'], history['test_acc'],
                        history['f1'], history['auroc'],
                        history['epoch_time']
                    ), 1
                ):
                    writer.writerow([
                        model_name, seed, epoch,
                        round(train_loss, 4), round(test_loss, 4),
                        round(train_acc, 4), round(test_acc, 4),
                        round(f1, 4), round(auroc, 4),
                        round(epoch_time, 2)
                    ])
    
    write_conf_header = not os.path.exists(confusion_csv_path)

    # --- Save confusion preds/targets if available ---
    with open(confusion_csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)

        if write_conf_header:
            writer.writerow(["Model", "Seed", "Epoch", "Preds", "Targets"])

        for model_name, histories in results.items():
            for entry in histories:
                seed = entry["seed"]
                history = entry["history"]
                if "confusion_preds" in history and "confusion_targets" in history:
                    # Assume this is only stored for final epoch
                    writer.writerow([
                        model_name, seed, "final",
                        " ".join(map(str, history["confusion_preds"])),
                        " ".join(map(str, history["confusion_targets"]))
                    ])