"""
Helper functions for initializing expt
Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
import random
import numpy as np
import os
import torch
import yaml
import logging

COLORS = {"yellow": "\x1b[33m", "blue": "\x1b[94m", "green": "\x1b[32m", "end": "\033[0m"}

# progress bar without tqdm :P
def progress_bar(progress=0, status="", bar_len=20):
    status = status.ljust(50)
    if progress == 1:
        status = "{}".format("Done...".ljust(50))
    block = int(round(bar_len * progress))
    text = "\rProgress: [{}] {}% {}".format(
        COLORS["green"] + "#" * block + COLORS["end"] + "-" * (bar_len - block), round(progress * 100, 2), status
    )
    print(text, end="")


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.metrics = {}

    def add(self, metrics):
        if len(self.metrics) == 0:
            self.metrics = {key: [value] for key, value in metrics.items()}
        else:
            for key, value in metrics.items():
                if key in self.metrics:
                    self.metrics[key].append(value)
                else:
                    raise ValueError("Incorrect metric key")

    def return_metrics(self):
        metrics = {key: np.mean(values) for key, values in self.metrics.items()}
        return metrics

    def return_msg(self):
        metrics = self.return_metrics()
        msg = "".join([f"{key}: {round(value, 3)} " for key, value in metrics.items()])
        return msg


class Logger:
    def __init__(self, output_dir):
        # Reset logger and setup output file
        [logging.root.removeHandler(handler) for handler in logging.root.handlers[:]]
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            handlers=[logging.FileHandler(os.path.join(output_dir, "trainlogs.txt"))],
        )

    def print(self, msg, mode=""):
        if mode == "info":
            print(f'{COLORS["yellow"]}[INFO] {msg}{COLORS["end"]}')
        elif mode == "train":
            print(f'\n{COLORS["blue"]}[TRAIN] {msg}{COLORS["end"]}')
        elif mode == "val":
            print(f'\n{COLORS["green"]}[VAL] {msg}{COLORS["end"]}')
        else:
            print(f"{msg}")

    def write(self, msg, mode=""):
        if mode == "info":
            msg = f"[INFO] {msg}"
        elif mode == "train":
            msg = f"[TRAIN] {msg}"
        elif mode == "val":
            msg = f"[VAL] {msg}"
        else:
            msg = f"{msg}"
        logging.info(msg)


def open_config(file_path):
    """
    Opens a configuration file.
    """
    config = yaml.safe_load(open(file_path, "r"))
    return config


def init_experiment(args, seed=420):
    """
    Instantiates output directories, logging files
    and random seeds.
    """
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Some other stuff
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # open config
    config = open_config(args["config"])

    # Setup logging directory
    output_dir = os.path.join(config["dataset"], config["task"], args["output"])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger = Logger(output_dir)

    logger.print("Logging at {}".format(output_dir), mode="info")
    logger.print("-" * 50)
    logger.print("{:>25}".format("Configuration"))
    logger.print("-" * 50)
    logger.print(yaml.dump(config))
    logger.print("-" * 50)

    # write hyper params to seperate file
    with open(os.path.join(output_dir, "hyperparameters.txt"), "w") as logs:
        logs.write(yaml.dump(config))

    # setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.print(f"Found device {torch.cuda.get_device_name(0)}", mode="info")

    return config, output_dir, logger, device


def print_network(model, name=""):
    """
    Pretty prints the model.
    """
    print(name.rjust(35))
    print("-" * 70)
    print("{:>25} {:>27} {:>15}".format("Layer.Parameter", "Shape", "Param"))
    print("-" * 70)

    for param in model.state_dict():
        p_name = param.split(".")[-2] + "." + param.split(".")[-1]
        if p_name[:2] != "BN" and p_name[:2] != "bn":  # Not printing batch norm layers
            print(
                "{:>25} {:>27} {:>15}".format(
                    p_name,
                    str(list(model.state_dict()[param].squeeze().size())),
                    "{0:,}".format(np.product(list(model.state_dict()[param].size()))),
                )
            )
    print("-" * 70 + "\n")