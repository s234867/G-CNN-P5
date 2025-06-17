## save_data.py ##

# imports
import csv
import os


def save_results(results, seeds, csv_path="results.csv"):
    write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        write_header = not os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["Model", "Seed", "Epoch", "Train_Acc", "Test_Acc", "F1", "AUROC"])

        for model_name, histories in results.items():
            for seed, history in zip(seeds, histories):
                for epoch, (train_acc, test_acc, f1, auroc) in enumerate(
                    zip(history['train_acc'], history['test_acc'], history['f1'], history['auroc']), 1
                ):
                    writer.writerow([
                        model_name, seed, epoch,
                        round(train_acc, 4),
                        round(test_acc, 4),
                        round(f1, 4),
                        round(auroc, 4)
                    ])