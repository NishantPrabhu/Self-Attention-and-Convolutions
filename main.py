
""" 
Trainer class and main script.

Authors: Mukund Varma T, Nishant Prabhu
"""

import os 
import wandb
import argparse 
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime as dt 
from models import attention, networks
from helpers import common, train_utils, data_utils


class Trainer:
    '''
    Helper class for training models, checkpointing and logging.
    '''

    def __init__(self, args):

        # Initialize experiment
        self.config, self.output_dir, self.logger, self.device = common.init_experiment(args, seed=420)

        # Networks and optimizers
        self.encoder = networks.BertEncoder(self.config['bert_encoder']).to(self.device)
        self.feature_pool = networks.FeaturePooling(self.config['feature_pooling']).to(self.device)
        self.clf_head = networks.ClassificationHead(self.config['clf_head']).to(self.device)

        self.optim = train_utils.get_optimizer(
            config = self.config['optimizer'], 
            params = list(self.encoder.parameters())+list(self.feature_pool.parameters())+list(self.clf_head.parameters()))

        self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
            config = {**self.config['scheduler'], 'epochs': self.config['epochs']}, 
            optimizer = self.optim)

        # Dataloaders
        self.train_loader, self.val_loader = data_utils.get_dataloader({
            **self.config['dataset'], 
            'batch_size': self.config['batch_size']
        })

        # Warmup handling
        if self.warmup_epochs > 0:
            self.warmup_rate = self.optim.param_groups[0]['lr'] / self.warmup_epochs

        # Losses and performance monitoring
        self.criterion = nn.NLLLoss()
        self.best_val_acc = 0
        run = wandb.init('self-attention-cnn')
        
        # Other
        self.logger.write(run.get_url(), mode='info')

        # Check for any saved state in the output directory and load
        if os.path.exists(os.path.join(self.output_dir, 'last.ckpt')):
            self.done_epochs = self.load_state()
            self.logger.print(f"Loaded saved state. Resuming from {self.done_epochs} epochs", mode="info")
            self.logger.write(f"Loaded saved state. Resuming from {self.done_epochs} epochs", mode="info")
        else:
            self.done_epochs = 0
            self.logger.print(f"No saved state found. Starting fresh", mode="info")
            self.logger.write(f"No saved state found. Starting fresh", mode="info")


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


    def save_state(self, epoch):
        ''' For resuming from run breakages, etc '''
        
        state = {
            'epoch': epoch,
            'encoder': self.encoder.state_dict(),
            'conv': self.feature_pool.state_dict(),
            'clf': self.clf_head.state_dict(),
            'optim': self.optim.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt'))


    def save_data(self):
        
        data = {
            'encoder': self.encoder.state_dict(),
            'conv': self.feature_pool.state_dict(),
            'clf': self.clf_head.state_dict()
        }
        torch.save(data, os.path.join(self.output_dir, 'best_model.ckpt'))


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


    def visualize_attention(self, epoch):
        ''' Generate attention on a batch of 100 images and plot them '''	

        batch = next(iter(self.val_loader))
        img = batch[0][1].unsqueeze(0).to(self.device)			

        with torch.no_grad():
            fvecs, attn_scores = self.encoder(self.feature_pool(img), return_attn=True)
        _, h, w, _ = self.feature_pool(img).size() 

        # attn_scores is a dict with num_encoder_blocks items 
        # Each item value has size (batch_size, num_heads, num_pixels, num_pixels)
        heads = self.config['bert_encoder']['num_heads']
        layers = self.config['bert_encoder']['num_encoder_blocks']
        fig = plt.figure(figsize=(10, 7))
        count = 1

        for name, attn in attn_scores.items():

            # Average attention over pixels
            # Attention has size (1, h, w, heads, h, w)
            # After the line below, it'll have shape (heads, h, w)
            attn = attn.mean(dim=0).view(-1, heads, h, w).mean(dim=0).detach().cpu().numpy()
            
            for i in range(attn.shape[0]):
                fig.add_subplot(layers, heads, count)
                plt.imshow(attn[i], cmap='Reds')
                plt.axis('off')
                count += 1

        plt.tight_layout(pad=1)
        plt.savefig(os.path.join(self.output_dir, f'attn_map_{epoch+1}.png'), pad_inches=0.05)


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

                # Visualize attention
                self.visualize_attention(epoch)

        self.logger.record('Training complete!', mode='info')



if __name__ == '__main__':

    # Parse arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-c', '--config', required=True, help='Path to configuration file')
    ap.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str, help='Path to output file')
    args = vars(ap.parse_args())

    # Initialize trainer
    trainer = Trainer(args)

    # Train 
    trainer.train()