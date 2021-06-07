
"""
Trainer class and main script.

Authors: Mukund Varma T, Nishant Prabhu
"""

import os
import time
import pickle
import wandb
import argparse
import torch
import itertools
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import interpolate
from datetime import datetime as dt
from models import attention, networks
from utils import common, train_utils, data_utils, losses
from matplotlib.patches import Rectangle
from PIL import Image
from torchvision import transforms as T


class Trainer:
    '''
    Helper class for training models, checkpointing and logging.
    '''

    def __init__(self, args):
        # Initialize experiment
        self.args = args
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)

        # Networks and optimizers
        self.encoder = networks.Encoder(self.config['encoder']).to(self.device)
        self.feature_pool = networks.FeaturePooling(self.config['feature_pooling']).to(self.device)
        self.clf_head = networks.ClassificationHead(self.config['clf_head']).to(self.device)

        self.optim = train_utils.get_optimizer(
            config = self.config['optimizer'],
            params = list(self.encoder.parameters())+list(self.feature_pool.parameters())+list(self.clf_head.parameters()))

        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**self.config['scheduler'], 'epochs': self.config['epochs']},
            optimizer = self.optim)

        # Total parameters
        enc_params = common.count_parameters(self.encoder)
        fp_params = common.count_parameters(self.feature_pool)
        clf_params = common.count_parameters(self.clf_head)
        print(f"\nModel parameter count: {round((enc_params + fp_params + clf_params)/1e06, 2)}M\n")

        # Dataloaders
        self.train_loader, self.val_loader = data_utils.get_dataloader({
            **self.config['dataset'],
            'batch_size': self.config['batch_size']
        })

        # Warmup handling
        if self.warmup_epochs > 0:
            self.warmup_rate = self.optim.param_groups[0]['lr'] / self.warmup_epochs

        # Losses and performance monitoring
        self.criterion = losses.ClassificationLoss()
        self.best_val_acc = 0
        run = wandb.init('self-attention-cnn')

        # Other
        self.logger.write(run.get_url(), mode='info')

        # Check for any saved state in the output directory and load
        if os.path.exists(os.path.join(self.output_dir, 'last_state.ckpt')):
            self.done_epochs = self.load_state()
            self.logger.record(f"Loaded saved state. Resuming from {self.done_epochs} epochs", mode="info")
        else:
            self.done_epochs = 0
            self.logger.record("No saved state found. Starting fresh", mode="info")

        # Check if a model has to be loaded from ckpt
        if args['load'] is not None:
            self.load_model(args['load'])
            self.output_dir = args['load']


    def train_one_step(self, data):

        img, labels = data[0].to(self.device), data[1].to(self.device)
        out = self.clf_head(self.encoder(self.feature_pool(img), return_attn=False))

        # Loss and update
        self.optim.zero_grad()
        loss = self.criterion(out, labels)
        loss.backward()
        self.optim.step()

        # Correct predictions
        pred = out.argmax(dim=-1)
        acc = pred.eq(labels.view_as(pred)).sum().item() / img.size(0)

        return {'Loss': loss.item(), 'Accuracy': acc}


    def validate_one_step(self, data):

        img, labels = data[0].to(self.device), data[1].to(self.device)
        out = self.clf_head(self.encoder(self.feature_pool(img), return_attn=False))

        loss = self.criterion(out, labels)
        pred = out.argmax(dim=-1)
        acc = pred.eq(labels.view_as(pred)).sum().item() / img.size(0)

        return {'Loss': loss.item(), 'Accuracy': acc}


    def save_state(self, epoch):                                        # For resuming from run breakages, runtime errors, etc.
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'conv': self.feature_pool.state_dict(),
            'clf': self.clf_head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.output_dir, f'state_{epoch}.ckpt'))


    def save_data(self):
        data = {
            'encoder': self.encoder.state_dict(),
            'conv': self.feature_pool.state_dict(),
            'clf': self.clf_head.state_dict()
        }
        torch.save(data, os.path.join(self.output_dir, 'best_model.ckpt'))


    def load_model(self, expt_dir):
        try:
            ckpt = torch.load(os.path.join(expt_dir, 'best_model.ckpt'))
            self.encoder.load_state_dict(ckpt['encoder'])
            self.feature_pool.load_state_dict(ckpt['conv'])
            self.clf_head.load_state_dict(ckpt['clf'])
            print('[INFO] Successfully loaded saved model!')
        except Exception as e:
            print(e)


    def load_state(self):
        last_state = torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))
        done_epochs = last_state['epoch']-1
        self.encoder.load_state_dict(last_state['encoder'])
        self.feature_pool.load_state_dict(last_state['conv'])
        self.clf_head.load_state_dict(last_state['clf'])
        self.optim.load_state_dict(last_state['optim'])
        self.scheduler.load_state_dict(last_state['scheduler'])
        return done_epochs


    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()


    def visualize_attention(self, epoch, one_pix=True):
        ''' Generate attention scores on 1 image and plot them '''

        # Disable any dropout, Batch norms, etc.
        self.encoder.eval()
        self.feature_pool.eval()

        # Load image
        transform = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        img = Image.open(self.args["image"])
        img = transform(img).unsqueeze(0).to(self.device)

        for batch in self.val_loader:
            with torch.no_grad():
                fvecs, attn_scores = self.encoder(self.feature_pool(img), return_attn=True)
            b, h, w, _ = self.feature_pool(img).size()

            # Obtain original image by inverting transform
            img_normal = data_utils.inverse_transform(img.squeeze(0), [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

            # attn_scores is a dict with num_encoder_blocks items
            # Each item value has size (batch_size, num_heads, num_pixels, num_pixels)
            heads = self.config['encoder']['num_heads']
            layers = self.config['encoder']['num_encoder_blocks']
            fig = plt.figure(figsize=(heads, layers))
            ax = fig.add_subplot(layers+1, heads, 1)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img_normal.permute(1, 2, 0).cpu().numpy())
            if one_pix:
                rect = Rectangle(xy=(7.5, 7.5), width=2, height=2, linewidth=1, edgecolor='black', facecolor='none')
                ax.add_patch(rect)
            count = 1

            for j, (name, attn) in enumerate(attn_scores.items()):

                # Average attention over pixels
                # Attention has size (1, h, w, heads, h, w)
                # After the line below, it'll have shape (heads, h, w)
                attn = attn.view(b, -1, heads, h, w)                                                    # (n, heads, h, w)
                attn = attn.contiguous().view(b, h*w, heads, -1)                                        # (b, n, heads, n)

                # Central pixel has flat index 120 for 16x16 image
                if one_pix:
                    attn = attn[:, 120, :, :][0].permute(1, 0).detach().cpu().numpy()                   # (n, heads)
                else:
                    attn = attn.mean(dim=1).permute(0, 2, 1).detach().cpu().numpy()[0]                  # (n, heads)

                for i in range(attn.shape[1]):
                    ax = fig.add_subplot(layers+1, heads, count+heads)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    ax.imshow(attn[:, i].reshape(h, w), cmap='Reds')
                    if one_pix:
                        rect = Rectangle(xy=(7.5, 7.5), width=1, height=1, linewidth=1, edgecolor='black', facecolor='none')
                        ax.add_patch(rect)
                    if i == 0:
                        ax.set_ylabel(f'Layer {j+1}', labelpad=10)
                    count += 1

            plt.tight_layout(pad=0.5)
            plt.savefig(os.path.join(self.output_dir, "viz_image.png"), pad_inches=0.05)
            break


    def throughput(self):
        self.encoder.eval()
        self.feature_pool.eval()
        self.clf_head.eval()
        dummy = torch.randn(16, 3, 32, 32, dtype=torch.float).to(self.device)
        repetitions = 100
        total_time = 0
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = self.clf_head(self.encoder(self.feature_pool(dummy), return_attn=False))
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                total_time += curr_time
        Throughput = (repetitions*16)/total_time
        self.logger.print(f"Final throughput: {Throughput}")


    def inference_time(self):
        self.encoder.eval()
        self.feature_pool.eval()
        self.clf_head.eval()
        dummy = torch.randn(1, 3, 32, 32, dtype=torch.float).to(self.device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))

        # GPU-WARM-UP
        for _ in range(10):
            _ = self.clf_head(self.encoder(self.feature_pool(dummy), return_attn=False))

        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.clf_head(self.encoder(self.feature_pool(dummy), return_attn=False))
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        self.logger.print(f"Inference time: {mean_syn} +/- {std_syn}")


    def plot_grid_query_pix(self, width, ax=None):
        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.set_xticks(np.arange(-width / 2, width / 2))  # , minor=True)
        ax.set_aspect(1)
        ax.set_yticks(np.arange(-width / 2, width / 2))  # , minor=True)
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            labelbottom=False,
            labelleft=False,
        )
        ax.grid(True, alpha=0.5)

        # query pixel
        querry_pix = Rectangle(xy=(-0.5, -0.5), width=1, height=1, edgecolor="black", fc='None', lw=2)

        ax.add_patch(querry_pix)
        ax.set_xlim(-width / 2, width / 2)
        ax.set_ylim(-width / 2, width / 2)
        ax.set_aspect("equal")

    def plot_attention_layer(self, layer_idx, width, ax=None):
        """Plot the 2D attention probabilities of all heads on an image
        of layer layer_idx
        """
        if ax is None:
            fig, ax = plt.subplots()

        attn = self.encoder.blocks[layer_idx].attention
        attention_probs, _ = attn.get_attention_probs(width + 2, width + 2)

        contours = np.array([0.9, 0.5])
        linestyles = [":", "-"]
        flat_colors = ["#3498db", "#f1c40f", "#2ecc71", "#e74c3c", "#e67e22", "#9b59b6", "#34495e", "#1abc9c", "#95a5a6"]

        if ax is None:
            fig, ax = plt.subplots()

        shape = attention_probs.shape
        # remove batch size if present
        if len(shape) == 6:
            shape = shape[1:]
        height, width, num_heads, _, _ = shape

        attention_at_center = attention_probs[width // 2, height // 2]
        attention_at_center = attention_at_center.detach().cpu().numpy()

        # compute integral of distribution for thresholding
        n = 1000
        t = np.linspace(0, attention_at_center.max(), n)
        integral = ((attention_at_center >= t[:, None, None, None]) * attention_at_center).sum(
            axis=(-1, -2)
        )

        self.plot_grid_query_pix(width - 2, ax)

        for h, color in zip(range(num_heads), itertools.cycle(flat_colors)):
            f = interpolate.interp1d(integral[:, h], t, fill_value=(1, 0), bounds_error=False)
            t_contours = f(contours)

            # remove duplicate contours if any
            keep_contour = np.concatenate([np.array([True]), np.diff(t_contours) > 0])
            t_contours = t_contours[keep_contour]

            for t_contour, linestyle in zip(t_contours, linestyles):
                ax.contour(
                    np.arange(-width // 2, width // 2) + 1,
                    np.arange(-height // 2, height // 2) + 1,
                    attention_at_center[h],
                    [t_contour],
                    extent=[- width // 2, width // 2 + 1, - height // 2, height // 2 + 1],
                    colors=color,
                    linestyles=linestyle
                )
        return ax

    def plot_attention_positions_all_layers(self, epoch, width, global_step=None):
        fig = plt.figure(figsize=(12, 2))
        for layer_idx in range(len(self.encoder.blocks)):
            ax = fig.add_subplot(1, len(self.encoder.blocks), layer_idx+1)
            self.plot_attention_layer(layer_idx, width, ax=ax)
            if layer_idx == 0:
                ax.set_ylabel(f"Epoch {epoch}", fontsize=15)

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'epoch_{epoch}.pdf'))
        # plt.show()


    def gaussian_visualization(self):
        for f in os.listdir(self.output_dir):
            if f.startswith('state'):
                state = torch.load(os.path.join(self.output_dir, f))
                done_epochs = state['epoch']
                self.encoder.load_state_dict(state['encoder'])
                self.feature_pool.load_state_dict(state['conv'])
                self.clf_head.load_state_dict(state['clf'])
                self.plot_attention_positions_all_layers(done_epochs, 16, None)


    def train(self):
        print()
        # Training loop
        for epoch in range(self.config['epochs'] - self.done_epochs + 1):
            train_meter = common.AverageMeter()
            val_meter = common.AverageMeter()
            self.logger.record('Epoch [{:3d}/{}]'.format(epoch+1, self.config['epochs']), mode='train')
            self.adjust_learning_rate(epoch+1)

            for idx, batch in enumerate(self.train_loader):
                train_metrics = self.train_one_step(batch)
                wandb.log({'Loss': train_metrics['Loss'], 'Epoch': epoch+1})
                train_meter.add(train_metrics)
                common.progress_bar(progress=idx/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1, status=train_meter.return_msg())
            self.logger.write(train_meter.return_msg(), mode='train')
            wandb.log({'Train accuracy': train_meter.return_metrics()['Accuracy'], 'Epoch': epoch+1})
            wandb.log({'Learning rate': self.optim.param_groups[0]['lr'], 'Epoch': epoch+1})

            # Save state
            if epoch % 50 == 0:
                self.save_state(epoch)

            # Validation
            if epoch % self.config['eval_every'] == 0:
                self.logger.record('Epoch [{:3d}/{}]'.format(epoch+1, self.config['epochs']), mode='val')

                for idx, batch in enumerate(self.val_loader):
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=idx/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1, status=val_meter.return_msg())
                self.logger.write(val_meter.return_msg(), mode='val')
                val_metrics = val_meter.return_metrics()
                wandb.log({'Validation loss': val_metrics['Loss'], 'Validation accuracy': val_metrics['Accuracy'], 'Epoch': epoch+1})

                if val_metrics['Accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['Accuracy']
                    self.save_data()

                # Visualize attention
                # self.visualize_attention(epoch)

        self.logger.record('Training complete!', mode='info')


class ResnetTrainer:
    '''
    For training a normal ResNet
    '''
    def __init__(self, args):
        # Initialize experiment
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)

        # Networks and optimizers
        self.model = networks.ResnetClassifier(self.config['resnet']).to(self.device)
        self.optim = train_utils.get_optimizer(config=self.config['optimizer'], params=self.model.parameters())
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**self.config['scheduler'], 'epochs': self.config['epochs']},
            optimizer = self.optim)

        # Parameter count
        num_params = common.count_parameters(self.model)
        print(f"\nModel parameter count: {round(num_params/1e06, 2)}M\n")

        # Warmup handling
        if self.warmup_epochs > 0:
            self.warmup_rate = self.optim.param_groups[0]['lr'] / self.warmup_epochs

        # Dataloaders
        self.train_loader, self.val_loader = data_utils.get_dataloader({
            **self.config['dataset'],
            'batch_size': self.config['batch_size']
        })

        # Losses and performance monitoring
        self.criterion = nn.NLLLoss()
        self.best_val_acc = 0
        run = wandb.init('self-attention-cnn')

        # Other
        self.logger.write(run.get_url(), mode='info')

        # Check for any saved state in the output directory and load
        if os.path.exists(os.path.join(self.output_dir, 'last_state.ckpt')):
            self.done_epochs = self.load_state()
            self.logger.print(f"Loaded saved state. Resuming from {self.done_epochs} epochs", mode="info")
            self.logger.write(f"Loaded saved state. Resuming from {self.done_epochs} epochs", mode="info")
        else:
            self.done_epochs = 0
            self.logger.print("No saved state found. Starting fresh", mode="info")
            self.logger.write("No saved state found. Starting fresh", mode="info")


    def train_one_step(self, data):

        img, labels = data[0].to(self.device), data[1].to(self.device)
        out = self.model(img)

        # Loss and update
        self.optim.zero_grad()
        loss = self.criterion(out, labels)
        loss.backward()
        self.optim.step()

        # Correct predictions
        pred = out.argmax(dim=-1)
        acc = pred.eq(labels.view_as(pred)).sum().item() / img.size(0)
        return {'Loss': loss.item(), 'Accuracy': acc}


    def validate_one_step(self, data):

        img, labels = data[0].to(self.device), data[1].to(self.device)
        out = self.model(img)

        loss = self.criterion(out, labels)
        pred = out.argmax(dim=-1)
        acc = pred.eq(labels.view_as(pred)).sum().item() / img.size(0)
        return {'Loss': loss.item(), 'Accuracy': acc}


    def save_state(self, epoch):                                        # For resuming from run breakages, runtime errors, etc.
        state = {
            'epoch': epoch,
            'model': self.model.state_dict(),
            'optim': self.optim.state_dict(),
        }
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt'))


    def save_data(self):
        data = {
            'model': self.model.state_dict(),
        }
        torch.save(data, os.path.join(self.output_dir, 'best_model.ckpt'))


    def load_state(self):
        last_state = torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))
        done_epochs = last_state['epoch'] - 1
        self.model.load_state_dict(last_state['model'])
        self.optim.load_state_dict(last_state['optim'])
        return done_epochs


    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        elif self.scheduler is not None:
            self.scheduler.step()
        else:
            pass


    def train(self):
        print()
        # Training loop
        for epoch in range(self.config['epochs'] - self.done_epochs + 1):
            train_meter = common.AverageMeter()
            val_meter = common.AverageMeter()
            self.logger.record('Epoch [{:3d}/{}]'.format(epoch+1, self.config['epochs']), mode='train')
            self.adjust_learning_rate(epoch+1)

            for idx, batch in enumerate(self.train_loader):
                train_metrics = self.train_one_step(batch)
                wandb.log({'Loss': train_metrics['Loss'], 'Epoch': epoch+1})
                train_meter.add(train_metrics)
                common.progress_bar(progress=idx/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1, status=train_meter.return_msg())
            self.logger.write(train_meter.return_msg(), mode='train')
            wandb.log({'Train accuracy': train_meter.return_metrics()['Accuracy'], 'Epoch': epoch+1})
            wandb.log({'Learning rate': self.optim.param_groups[0]['lr'], 'Epoch': epoch+1})

            # Save state
            self.save_state(epoch)

            # Validation
            if epoch % self.config['eval_every'] == 0:
                self.logger.record('Epoch [{:3d}/{}]'.format(epoch+1, self.config['epochs']), mode='val')

                for idx, batch in enumerate(self.val_loader):
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=idx/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1, status=val_meter.return_msg())
                self.logger.write(val_meter.return_msg(), mode='val')
                val_metrics = val_meter.return_metrics()
                wandb.log({'Validation loss': val_metrics['Loss'], 'Validation accuracy': val_metrics['Accuracy'], 'Epoch': epoch+1})

                if val_metrics['Accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['Accuracy']
                    self.save_data()

        self.logger.record('\nTraining complete!', mode='info')



if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='Path to configuration file')
    ap.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str, help='Path to output directory')
    ap.add_argument('-l', '--load', default=None, help='Path to output dir from which pretrained models will be loaded')
    ap.add_argument('-i', '--image', default="imgs/cat_s_000019.png", help='Path to image on which attention visualization will be performed')
    ap.add_argument('-t', '--task', default='train', type=str, help='Task to perform, choose from (train, resnet_train, viz, viz_1, gauss_viz)')
    args = vars(ap.parse_args())

    if args['task'] == 'train':
        trainer = Trainer(args)
        trainer.train()

    elif args['task'] == 'resnet_train':
        trainer = ResnetTrainer(args)
        trainer.train()

    elif args['task'] == 'viz_1':
        if args['load'] is None:
            raise NotImplementedError("Please load a trained model using --load for visualization tasks")
        trainer = Trainer(args)
        trainer.visualize_attention(1, one_pix=True)

    elif args['task'] == 'viz':
        if args['load'] is None:
            raise NotImplementedError("Please load a trained model using --load for visualization tasks")
        trainer = Trainer(args)
        trainer.visualize_attention(1, one_pix=False)

    elif args['task'] == 'time':
        trainer = Trainer(args)
        trainer.throughput()
        trainer.inference_time()

    elif args['task'] == 'gauss_viz':
        trainer = Trainer(args)
        trainer.gaussian_visualization()

    else:
        raise NotImplementedError(f"Unrecognized task {args['task']}")
