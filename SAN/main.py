
""" 
Trainer class and main script.

Authors: Mukund Varma T, Nishant Prabhu
"""

import os
import wandb
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from models import networks
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
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)

        # Networks and optimizers
        self.encoder = networks.Encoder(self.config['encoder']).to(self.device)

        self.optim = train_utils.get_optimizer(
            config=self.config['optimizer'],
            params=self.encoder.parameters())
        
        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config={**self.config['scheduler'], 'epochs': self.config['epochs']},
            optimizer=self.optim)

        # Total trainable params
        total = common.count_parameters(self.encoder)
        if total / 1e06 >= 1:
            self.logger.record(f"Total trainable parameters: {round(total/1e06, 2)}M", mode='info')
        else:
            self.logger.record(f"Total trainable parameters: {total}", mode='info')

        # Dataloaders
        self.train_loader, self.val_loader = data_utils.get_dataloader({
            **self.config['dataset'], 
            'batch_size': self.config['batch_size']
        })

        # Warmup handling
        if self.warmup_epochs > 0:
            self.warmup_rate = self.optim.param_groups[0]['lr'] / self.warmup_epochs

        # Losses and performance monitoring
        self.criterion = losses.ClassificationLoss(self.config['criterion']['smoothing'])
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

        # Load best model if any argument is provided
        if 'load' in args.keys():
            if os.path.exists(os.path.join(args['load'], 'best_model.ckpt')):
                self.load_model(args['load'])
                self.logger.print(f"Successfully loaded model at {args['load']}", mode='info')
            else:
                self.logger.print(f"No saved model found at {args['load']}; please check your input!", mode='info')


    def train_one_step(self, data):

        img, labels = data[0].to(self.device), data[1].to(self.device)
        out = self.encoder(img, False)
        
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
        with torch.no_grad():
            out = self.encoder(img, False)
        
        loss = self.criterion(out, labels)
        pred = out.argmax(dim=-1)	
        acc = pred.eq(labels.view_as(pred)).sum().item() / img.size(0)
        
        return {'Loss': loss.item(), 'Accuracy': acc}


    def save_state(self, epoch):                                # For resuming from run breakages, runtime errors, etc.
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt'))


    def save_data(self):
        data = {
            'encoder': self.encoder.state_dict(),
        }
        torch.save(data, os.path.join(self.output_dir, 'best_model.ckpt'))


    def load_state(self):
        last_state = torch.load(os.path.join(self.output_dir, 'last_state.ckpt'))
        done_epochs = last_state['epoch']-1
        self.encoder.load_state_dict(last_state['encoder'])
        self.optim.load_state_dict(last_state['optim'])
        self.scheduler.load_state_dict(last_state['scheduler'])
        return done_epochs


    def load_model(self, output_dir):
        best = torch.load(os.path.join(output_dir, 'best_model.ckpt'))
        self.encoder.load_state_dict(best['encoder'])


    def adjust_learning_rate(self, epoch):
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()


    def visualize_attention(self, epoch):
        # Disable any dropout, Batch norms, etc.
        self.encoder.eval()
        # batch = next(iter(self.val_loader))
        layers = len(self.config['encoder']['layers'])

        # Load image
        transform = T.Compose([T.ToTensor(), T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
        img = Image.open('imgs/{}'.format(os.listdir("imgs")[0]))
        img = transform(img).unsqueeze(0)

        for k, idx in enumerate([16, 17]):
            img = img.to(self.device)			
            with torch.no_grad():
                fvecs, attn_scores = self.encoder(img, return_attn=True)            # attn_scores -> (bs, heads, kernel^2, h * w)
            _, _, h, w = img.size()
            _, heads, patches, _ = attn_scores['pass_1'].size()

            # Obtain original image by inverting transform
            img_normal = data_utils.inverse_transform(img.squeeze(0), [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

            # attn_scores is a dict with tensors of size (bs, heads, kernel^2, h * w) 
            fig = plt.figure(figsize=(patches, 5))
            ax = fig.add_subplot(layers+1, patches, 1)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img_normal.permute(1, 2, 0).cpu().numpy())
            count = 1

            for l_idx in range(layers):
                attn = attn_scores[f'pass_{l_idx+1}']
                attn = attn.squeeze(0).mean(dim=0)                                              # (kernel^2, h * w) 
                for i in range(patches):
                    val = attn[i].view(h, w).detach().cpu().numpy()                             # (h, w)                
                    ax = fig.add_subplot(layers+1, patches, count+patches)
                    ax.get_xaxis().set_ticks([])
                    ax.get_yaxis().set_ticks([])
                    ax.imshow(val, cmap='Reds')
                    count += 1

            plt.tight_layout(pad=0.5)
            # plt.show()
            plt.savefig('/home/nishant/Desktop/pairwise_hier.pdf', pad_inches=0.05)


    def throughput(self):
        self.encoder.eval()
        dummy = torch.randn(16, 3, 32, 32, dtype=torch.float).to(self.device)
        repetitions = 100
        total_time = 0
        with torch.no_grad():
            for rep in range(repetitions):
                starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                starter.record()
                _ = self.encoder(dummy)
                ender.record()
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)/1000
                total_time += curr_time
        Throughput = (repetitions*16)/total_time
        self.logger.print(f"Final throughput: {Throughput}")


    def inference_time(self):
        self.encoder.eval()
        dummy = torch.randn(1, 3, 32, 32, dtype=torch.float).to(self.device)
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        repetitions = 300
        timings = np.zeros((repetitions, 1))
        
        # GPU-WARM-UP
        for _ in range(10):
            _ = self.encoder(dummy)
        
        # MEASURE PERFORMANCE
        with torch.no_grad():
            for rep in range(repetitions):
                starter.record()
                _ = self.encoder(dummy)
                ender.record()
                # WAIT FOR GPU SYNC
                torch.cuda.synchronize()
                curr_time = starter.elapsed_time(ender)
                timings[rep] = curr_time
        mean_syn = np.sum(timings) / repetitions
        std_syn = np.std(timings)
        self.logger.print(f"Inference time: {mean_syn} +/- {std_syn}")


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

        self.logger.record('Training complete!', mode='info')


if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='Path to configuration file')
    ap.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str, help='Path to output file')
    ap.add_argument('-l', '--load', type=str, help='Name of the save directory from which best model should be loaded')
    ap.add_argument('-t', '--task', type=str, default='train', help='Task to be performed, choose from (train, viz, time)')
    args = vars(ap.parse_args())

    # For training attention networks
    trainer = Trainer(args)

    # Perform specified task
    if args['task'] == 'train': 
        trainer.train()

    elif args['task'] == 'viz':
        trainer.visualize_attention(1)

    elif args['task'] == 'time':
        trainer.throughput()
        trainer.inference_time()

    else:
        raise ValueError(f"Unrecognized task {args['task']}")