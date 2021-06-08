
import os 
import argparse
import matplotlib.pyplot as plt 


ap = argparse.ArgumentParser()
ap.add_argument("-r", "--root", default="./data_to_plot", type=str, help="Path to dir containing output dirs of experiments whose results are to be plot")
ap.add_argument("-w", "--which", default="val", type=str, help="Whether to plot train metrics ('train') or validation metrics ('val')")
ap.add_argument("-m", "--metric", default="accuracy", type=str, help="The quantity to plot for chosen experiments; whether 'loss' or 'accuracy'")
args = vars(ap.parse_args())
if args["metric"] not in ["loss", "accuracy"]:
    raise ValueError(f"Argument 'metric' should be one of ('loss', 'accuracy'), got {args['metric']}")

data = {}
root = args["root"]

for folder in os.listdir(root):
    data[folder] = {"epoch": [], "loss": [], "accuracy": []}
    if not os.path.exists(os.path.join(root, folder, "trainlogs.txt")):
        raise NotImplementedError(f"Could not find 'trainlogs.txt' in {folder}. Please ensure this is the right output directory.")

    with open(os.path.join(root, folder, "trainlogs.txt"), "r") as f:
        lines = f.read().split("\n")

    if args["which"] == "val":
        val_lines = [l for l in lines if l.startswith("[VAL]")]
        for i in range(len(val_lines)):
            if i % 2 == 0:
                data[folder]["epoch"].append(int(val_lines[i][13:16])-1)
            else:
                data[folder]["loss"].append(float(val_lines[i][13:19]))
                data[folder]["accuracy"].append(float(val_lines[i][31:]))

    elif args["which"] == "train":
        train_lines = [l for l in lines if l.startswith("[TRAIN]")]
        for i in range(len(train_lines)):
            if i % 2 == 0:
                data[folder]["epoch"].append(int(train_lines[i][15:18])-1)
            else:
                data[folder]["loss"].append(float(train_lines[i][15:21]))
                data[folder]["accuracy"].append(float(train_lines[i][33:]))

    else:
        raise ValueError(f"Argument 'which' should be one of ('train', 'val'), got {args['which']}")

title_text = "Validation performance" if args["which"] == "val" else "Train performance"

# Plotting here
plt.figure(figsize=(10, 8))
for k, v in data.items():
    plt.plot(v["epoch"], v[args["metric"]], marker='.', label=k)
plt.xlabel("Epoch", fontsize=15)
plt.ylabel(args["metric"][0].upper() + args["metric"][1:], fontsize=15)
plt.title(title_text, fontsize=15, fontweight="bold")
plt.grid(alpha=0.4)
plt.legend()
plt.savefig("./performance_comparison.png", pad_inches=0.05)