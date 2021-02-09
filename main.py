"""
Script to use all other modules and run the thing.
Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
from datetime import datetime as dt
import argparse
import os
from utils import common
from data import augmentations, datasets
from models import models
import wandb

# available models
MODEL_HELPER = {
    "resnet": models.ResNetCLS,
}

# ===================================================================================================
# Trainer: helper class for training proceedure
# ===================================================================================================


class Trainer:
    def __init__(self, args):

        # Initialize experiment
        self.config, output_dir, self.logger, device = common.init_experiment(args, seed=420)

        # get datasets and transforms
        self.data_loaders = {}
        for key, params in self.config["dataloaders"].items():
            transform = augmentations.get_transform(self.config[params["transform"]], datasets.DATASET_HELPER[self.config["dataset"]]["norm"])
            dataset = datasets.get_dataset(
                name=self.config["dataset"],
                dataroot=os.path.join(output_dir.split("/")[0], "data"),
                split=params["split"],
                transform=transform,
            )
            shuffle = params.get("shuffle", False)
            self.data_loaders[key] = datasets.get_dataloader(
                dataset=dataset,
                batch_size=self.config["batch_size"],
                num_workers=self.config["num_workers"],
                shuffle=shuffle,
            )

        assert self.config["task"] in MODEL_HELPER.keys(), "Invalid task, choose from resnet"
        self.model = MODEL_HELPER[self.config["task"]](config=self.config, device=device, output_dir=output_dir)

        if os.path.exists(os.path.join(output_dir, "last.ckpt")):
            self.epoch_start = self.model.load_ckpt()
            self.logger.print(f"Loaded checkpoint. Resuming from {self.epoch_start} epochs...", mode="info")
            self.logger.write(f"Loaded checkpoint. Resuming from {self.epoch_start} epochs...", mode="info")
        else:
            self.epoch_start = 1
            self.logger.print(f"Checkpoint not found. Starting fresh...", mode="info")
            self.logger.write(f"Checkpoint not found. Starting fresh...", mode="info")

    def train(self):
        epoch_meter = common.AverageMeter()
        run = wandb.init(self.config["task"])
        self.logger.write(run.get_url(), mode="info")
        train_step = 0

        for epoch in range(self.epoch_start, self.config["epochs"] + 1):
            self.logger.print(f'Epoch [{epoch}/{self.config["epochs"]}]', mode="info")
            self.logger.write(f'Epoch [{epoch}/{self.config["epochs"]}]', mode="info")
            epoch_meter.reset()
            for indx, data in enumerate(self.data_loaders["train"]):
                train_metric = self.model.train_one_step(data)
                wandb.log({**train_metric, "train step": train_step})
                epoch_meter.add(train_metric)
                common.progress_bar(progress=indx / len(self.data_loaders["train"]), status=epoch_meter.return_msg())
                train_step += 1
            common.progress_bar(progress=1)
            self.logger.print(epoch_meter.return_msg(), mode="train")
            self.logger.write(epoch_meter.return_msg(), mode="train")
            wandb.log({"lr": self.model.optim.param_groups[0]["lr"], "train epoch": epoch})
            if epoch <= self.model.warmup_epochs:
                self.model.optim.param_groups[0]["lr"] = epoch / self.model.warmup_epochs * self.model.lr
            elif self.model.lr_scheduler is not None:
                self.model.lr_scheduler.step()
            self.model.save_ckpt(epoch)

            if epoch % self.config["eval_every"] == 0 or epoch == 1:
                metrics = self.model.validate(val_loader=self.data_loaders["val"])
                wandb.log({**metrics, "val epoch": epoch})

                msg = "".join([f"{key}: {round(value, 3)} " for key, value in metrics.items()])
                self.logger.print(msg, mode="val")
                self.logger.write(msg, mode="val")

                self.model.save_model(f"{epoch}.pth")

        self.logger.print("Training complete", mode="info")
        self.logger.write("Training complete", mode="info")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="Path to configuration file")
    parser.add_argument(
        "-o", "--output", default=dt.now().strftime("%Y-%m-%d_%H-%M"), type=str, help="Output directory"
    )
    args = vars(parser.parse_args())

    trainer = Trainer(args)
    trainer.train()