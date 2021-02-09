"""
Training functions and classes.
Authors: Mukund Varma T, Nishant Prabhu
"""

# Dependencies
from . import networks
from utils import common, train_utils
import torch
import os
import torch.nn.functional as F
import numpy as np
from data.datasets import DATASET_HELPER

class BaseModel:
    def __init__(self, config, device, output_dir):
        
        self.config = config
        self.device = device
        self.output_dir = output_dir
        
        # Models
        self.model = None

        # Optimizer, scheduler and criterion
        self.optim = None
        self.lr_scheduler, self.warmup_epochs = None, 0
        self.lr = config["optim"]["lr"]

        self.best = 0
        
    def load_ckpt(self):

        ckpt = torch.load(os.path.join(self.output_dir, "last.ckpt"))
        self.model.load_state_dict(ckpt["model"])
        self.optim.load_state_dict(ckpt["optim"])
        if self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        return ckpt["epoch"]

    def train_one_step(self, data):
        """ Trains model on one batch of data """

        img, target = data["img"].to(self.device), data["target"].to(self.device)
        pred = self.model(img)
        loss = F.nll_loss(pred, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        
        pred_choice = pred.max(1)[1]
        acc = pred_choice.eq(target.long().data).cpu().sum()/pred_choice.shape[0]

        return {"train loss": loss.item(), "train acc": acc.item()}
    
    def save_ckpt(self, epoch):
        ckpt = {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "epoch": epoch,
        }
        if self.lr_scheduler:
            ckpt["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(ckpt, os.path.join(self.output_dir, "last.ckpt"))

    def save_model(self, fname):
        model_save = {"model": self.model.state_dict()}
        torch.save(model_save, os.path.join(self.output_dir, fname))
    
    def validate(self, val_loader):
        # validate
        
        acc_cntr = []

        for indx, data in enumerate(val_loader):
            img, target = data["img"].to(self.device), data["target"].to(self.device)
            pred = self.model(img)
            pred_choice = pred.max(1)[1]
            acc = pred_choice.eq(target.long().data).cpu().sum()/pred_choice.shape[0]
            acc_cntr.append(acc.item())
            common.progress_bar(progress=indx / len(val_loader), status=f"test acc: {round(np.mean(acc_cntr), 3)}")
        common.progress_bar(progress=1)
        
        acc = np.mean(acc_cntr)
        if acc >= self.best:
            self.save_model("best.pth")
            self.best = acc

        return {"val acc": acc}

class ResNetCLS(BaseModel):
    def __init__(self, config, device, output_dir):
        super().__init__(config, device, output_dir)
        
        enc = networks.ResNet18(**config["encoder"])
        common.print_network(enc, "Encoder")
        cls_head = networks.ClassificationHead(enc.backbone_dim, n_classes=DATASET_HELPER[self.config["dataset"]]["classes"])
        common.print_network(cls_head, "Classification Head")
        self.model = torch.nn.Sequential(enc, cls_head).to(device)
        self.optim = train_utils.get_optimizer(
            config=config["optim"], params=self.model.parameters()
        )
        self.lr_scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config={**config["lr_scheduler"], "epochs": config["epochs"]}, optimizer=self.optim
        )
        self.lr = config["optim"]["lr"]