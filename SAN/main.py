
""" 
Trainer class and main script.

Authors: Mukund Varma T, Nishant Prabhu
"""

import os
import wandb
import argparse
import torch
from models import san_networks
from datetime import datetime as dt
from utils import common, train_utils, data_utils, losses


class Trainer:
    '''
    Helper class for training models, checkpointing and logging.
    '''

    def __init__(self, args):

        # Initialize experiment
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)

        # Networks and optimizers
        self.encoder = san_networks.Encoder(self.config['encoder']).to(self.device)

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


    def train_one_step(self, data):

        img, labels = data[0].to(self.device), data[1].to(self.device)
        out = self.encoder(img)
        
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
            out = self.encoder(img)
        
        loss = self.criterion(out, labels)
        pred = out.argmax(dim=-1)	
        acc = pred.eq(labels.view_as(pred)).sum().item() / img.size(0)
        
        return {'Loss': loss.item(), 'Accuracy': acc}


    def save_state(self, epoch):
        ''' For resuming from run breakages, etc '''
        
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


    def adjust_learning_rate(self, epoch):
        
        if epoch < self.warmup_epochs:
            for group in self.optim.param_groups:
                group['lr'] = 1e-12 + (epoch * self.warmup_rate)
        else:
            self.scheduler.step()


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
    args = vars(ap.parse_args())

    # For training attention networks
    trainer = Trainer(args)

    # Train 
    trainer.train()