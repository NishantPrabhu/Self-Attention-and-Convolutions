
""" 
Training function and main script.

Authors: Nishant Prabhu, Mukund Varma T
"""

import os 
import time
import torch 
import wandb
import argparse
import numpy as np 
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from datetime import datetime as dt 
from models import networks
from utils import common, train_utils, data_utils, losses
from PIL import Image
from torchvision import transforms as T


class Trainer:
    '''
    Helper class for training models, checkpointing and logging.
    '''
    def __init__(self, args):
        # Initialize experiment
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)

        # Networks and optimizers
        self.encoder = networks.Encoder(self.config['encoder']).to(self.device)
        self.patcher = networks.PatchExtraction(self.config['patcher']).to(self.device)
        self.clf_head = networks.ClassificationHead(self.config['clf_head']).to(self.device)

        self.optim = train_utils.get_optimizer(
            config = self.config['optimizer'], 
            params = list(self.encoder.parameters())+list(self.patcher.parameters())+list(self.clf_head.parameters()))

        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**self.config['scheduler'], 'epochs': self.config['epochs']}, 
            optimizer = self.optim)

        # Parameter count
        enc_params = common.count_parameters(self.encoder)
        patch_params = common.count_parameters(self.patcher)
        clf_params = common.count_parameters(self.clf_head)
        total_params = (enc_params + patch_params + clf_params)
        self.logger.record(f"Trainable parameters: {round(total_params / 1e06, 2)}M", mode='info')

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
            self.logger.print(f"Loaded saved state. Resuming from {self.done_epochs} epochs", mode="info")
            self.logger.write(f"Loaded saved state. Resuming from {self.done_epochs} epochs", mode="info")
        else:
            self.done_epochs = 0
            self.logger.print("No saved state found. Starting fresh", mode="info")
            self.logger.write("No saved state found. Starting fresh", mode="info")

        # Load model if specified
        if 'load' in args.keys():
            if os.path.exists(os.path.join(args['load'], 'best_model.ckpt')):
                self.load_model(args['load'])
                self.logger.print(f"Succesfully loaded model from {args['load']}")
            else:
                self.logger.print(f"No saved model found at {args['load']}; please check your input!")


    def train_one_step(self, data):
        ''' Trains model on single batch of data '''

        img, labels = data[0].to(self.device), data[1].to(self.device)
        out = self.clf_head(self.encoder(self.patcher(img), return_attn=False))
        
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
        ''' Evaluates model on single batch of data '''

        img, labels = data[0].to(self.device), data[1].to(self.device)
        with torch.no_grad():
            out = self.clf_head(self.encoder(self.patcher(img), return_attn=False))
        
        loss = self.criterion(out, labels)
        pred = out.argmax(dim=-1)	
        acc = pred.eq(labels.view_as(pred)).sum().item() / img.size(0)
        
        return {'Loss': loss.item(), 'Accuracy': acc}


    def save_state(self, epoch):                                        # For resuming from run breakages, runtime errors, etc.
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'patcher': self.patcher.state_dict(),
            'clf': self.clf_head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt'))


    def save_data(self):
        data = {
            'encoder': self.encoder.state_dict(),
            'patcher': self.patcher.state_dict(),
            'clf': self.clf_head.state_dict()
        }
        torch.save(data, os.path.join(self.output_dir, 'best_model.ckpt'))


    def load_state(self):
        done_epochs = torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))['epoch']-1
        self.encoder.load_state_dict(torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))['encoder'])
        self.patcher.load_state_dict(torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))['patcher']) 
        self.clf_head.load_state_dict(torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))['clf'])
        self.optim.load_state_dict(torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))['optim'])
        self.scheduler.load_state_dict(torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))['scheduler'])
        return done_epochs


    def load_model(self, output_dir):
        model = torch.load(os.path.join(output_dir, 'best_model.ckpt'))
        self.encoder.load_state_dict(model['encoder'])
        self.patcher.load_state_dict(model['patcher'])
        self.clf_head.load_state_dict(model['clf'])


    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()


    def throughput(self):
        self.encoder.eval()
        self.patcher.eval()
        self.clf_head.eval()
        dummy = torch.randn(16, 3, 32, 32, dtype=torch.float).to(self.device)
        repetitions = 100
        total_time = 0
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = self.clf_head(self.encoder(self.patcher(dummy), return_attn=False))
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                total_time += curr_time
        Throughput = (repetitions*16)/total_time
        self.logger.print(f"Final throughput: {Throughput}")


    def inference_time(self):
        self.encoder.eval()
        self.patcher.eval()
        self.clf_head.eval()
        dummy = torch.randn(1, 3, 32, 32, dtype=torch.float).to(self.device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        
        # GPU-WARM-UP
        for _ in range(10):
            _ = self.clf_head(self.encoder(self.patcher(dummy), return_attn=False))
        
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.clf_head(self.encoder(self.patcher(dummy), return_attn=False))
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        self.logger.print(f"Inference time: {mean_syn} +/- {std_syn}")


    def visualize_attention(self, epoch, one_pix=True):
        ''' Generate attention scores on 1 image and plot them '''

        # Disable any dropout, Batch norms, etc.
        self.encoder.eval()
        self.patcher.eval()
        # batch = next(iter(self.val_loader))
        
        # Load image
        transform = T.Compose([T.Resize(32), T.ToTensor(), T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        img = Image.open('imgs/english_toy_spaniel_s_000685.png')
        img = transform(img).unsqueeze(0)

        for k, idx in enumerate([16, 17]):
            img = img.to(self.device)			
            with torch.no_grad():
                fvecs, attn_scores = self.encoder(self.patcher(img), return_attn=True)

            # Obtain original image by inverting transform
            img_normal = data_utils.inverse_transform(img.squeeze(0), [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

            # attn_scores is a dict with num_encoder_blocks items 
            # Each item value has size (batch_size, num_heads, num_pixels, num_pixels)
            heads = self.config['encoder']['num_heads']
            layers = self.config['encoder']['num_encoder_blocks']
            h = self.config['patcher']['patch_grid_size']

            fig = plt.figure(figsize=(heads, 7))
            ax = fig.add_subplot(layers+1, heads, 1)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img_normal.permute(1, 2, 0).cpu().numpy())
            if one_pix:
                rect = Rectangle(xy=(5.5, 5.5), width=2, height=2, linewidth=1, edgecolor='red', facecolor='none')
                ax.add_patch(rect)
            count = 1

            for j, (name, attn) in enumerate(attn_scores.items()):

                # Attention has size (1, heads, grid_size^2 + 1, grid_size^2 + 1)
                # After the line below, it'll have shape (heads, grid_size^2, grid_size^2)
                attn = attn.squeeze(0)[:, 1:, 1:]                                           # (heads, grid_size^2, grid_size^2)
                if not one_pix:
                    attn = attn / attn.sum(dim=1, keepdim=True)                             # (heads, grid_size^2, grid_size^2)

                if one_pix:
                    attn = attn[:, :, 34].detach().cpu().numpy()                           # (heads, grid_size^2)
                else:
                    attn = attn.mean(dim=-1).detach().cpu().numpy()                         # (heads, grid_size^2)
                
                for i in range(attn.shape[0]):
                    ax = fig.add_subplot(layers+1, heads, count+heads)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    ax.imshow(attn[i, :].reshape(h, h), cmap='Reds')
                    if i == 0:
                        ax.set_ylabel(f'Layer {j+1}', labelpad=10)
                    count += 1

            plt.tight_layout(pad=0.5)
            # plt.show()
            plt.savefig('/home/nishant/Desktop/vit_2.pdf', pad_inches=0.05)


    def train(self):
        ''' Trains the model '''
        
        print()
        # Training loop
        for epoch in range(self.config['epochs'] - self.done_epochs + 1):

            train_meter = common.AverageMeter()
            val_meter = common.AverageMeter()
            self.logger.record('Epoch [{:3d}/{}]'.format(self.done_epochs+epoch+1, self.config['epochs']), mode='train')
            self.adjust_learning_rate(self.done_epochs+epoch+1)

            for idx, batch in enumerate(self.train_loader):
                train_metrics = self.train_one_step(batch)
                wandb.log({'Loss': train_metrics['Loss'], 'Epoch': self.done_epochs+epoch+1})
                train_meter.add(train_metrics)
                common.progress_bar(progress=idx/len(self.train_loader), status=train_meter.return_msg())

            common.progress_bar(progress=1, status=train_meter.return_msg())
            self.logger.write(train_meter.return_msg(), mode='train')
            wandb.log({'Train accuracy': train_meter.return_metrics()['Accuracy'], 'Epoch': self.done_epochs+epoch+1})
            wandb.log({'Learning rate': self.optim.param_groups[0]['lr'], 'Epoch': self.done_epochs+epoch+1})

            # Save state
            self.save_state(self.done_epochs+epoch)

            # Validation
            if epoch % self.config['eval_every'] == 0:

                self.logger.record('Epoch [{:3d}/{}]'.format(self.done_epochs+epoch+1, self.config['epochs']), mode='val')
                
                for idx, batch in enumerate(self.val_loader):
                    val_metrics = self.validate_one_step(batch)
                    val_meter.add(val_metrics)
                    common.progress_bar(progress=idx/len(self.val_loader), status=val_meter.return_msg())

                common.progress_bar(progress=1, status=val_meter.return_msg())
                self.logger.write(val_meter.return_msg(), mode='val')
                val_metrics = val_meter.return_metrics()
                wandb.log({
                    'Validation loss': val_metrics['Loss'], 
                    'Validation accuracy': val_metrics['Accuracy'], 
                    'Epoch': self.done_epochs + epoch + 1
                })

                if val_metrics['Accuracy'] > self.best_val_acc:
                    self.best_val_acc = val_metrics['Accuracy']
                    self.save_data()

        self.logger.record('Training complete!', mode='info')


if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='Path to configuration file')
    ap.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str, help='Path to output file')
    ap.add_argument('-l', '--load', type=str, help='Directory from which best model will be loaded')
    ap.add_argument('-t', '--task', type=str, required=True, help='The task to perform, choose from (train, viz, viz_1, time)')
    args = vars(ap.parse_args())

    # For training attention networks
    trainer = Trainer(args)

    # Perform specified task
    if args['task'] == 'train':
        trainer.train()

    elif args['task'] == 'viz':
        trainer.visualize_attention(1, False)

    elif args['task'] == 'viz_1':
        trainer.visualize_attention(1, True)

    elif args['task'] == 'time':
        trainer.throughput()
        trainer.inference_time()