
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
from utils import common, train_utils, data_utils, losses



class Trainer:
    '''
    Helper class for training models, checkpointing and logging.
    '''

    def __init__(self, args):
        # Initialize experiment
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
        torch.save(state, os.path.join(self.output_dir, 'last_state.ckpt'))


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


    def visualize_attention(self, epoch):
        ''' Generate attention on a batch of 100 images and plot them '''	
        batch = next(iter(self.val_loader))
        img = batch[0][1].unsqueeze(0).to(self.device)			

        with torch.no_grad():
            fvecs, attn_scores = self.encoder(self.feature_pool(img), return_attn=True)
        _, h, w, _ = self.feature_pool(img).size() 

        # Obtain original image by inverting transform
        img_normal = data_utils.inverse_transform(img.squeeze(0), [0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

        # attn_scores is a dict with num_encoder_blocks items 
        # Each item value has size (batch_size, num_heads, num_pixels, num_pixels)
        heads = self.config['encoder']['num_heads']
        layers = self.config['encoder']['num_encoder_blocks']
        fig = plt.figure(figsize=(11, 7))
        ax = fig.add_subplot(layers+1, heads, 5)
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        ax.imshow(img_normal.permute(1, 2, 0).cpu().numpy())
        count = 1

        for j, (name, attn) in enumerate(attn_scores.items()):

            # Average attention over pixels
            # Attention has size (1, h, w, heads, h, w)
            # After the line below, it'll have shape (heads, h, w)
            attn = attn.mean(dim=0).view(-1, heads, h, w)[int(h * w//2)].detach().cpu().numpy()
            
            for i in range(attn.shape[0]):
                ax = fig.add_subplot(layers+1, heads, count+heads)
                ax.get_xaxis().set_ticks([])
                ax.get_yaxis().set_ticks([])
                ax.imshow(attn[i], cmap='Reds')
                if i == 0:
                    ax.set_ylabel(f'Layer {j+1}', labelpad=10)
                count += 1

        plt.tight_layout(pad=0.5)
        plt.show()
        # plt.savefig(os.path.join(self.output_dir, 'attn_map_final.png'), pad_inches=0.05)

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
    ap.add_argument('-o', '--output', default=dt.now().strftime('%Y-%m-%d_%H-%M'), type=str, help='Path to output file')
    ap.add_argument('-l', '--load', default=None, help='Path to output dir from which pretrained models will be loaded')
    ap.add_argument('-t', '--task', default='train', type=str, help='Which task to perform, choose from (trian, resnet_train, viz)')
    args = vars(ap.parse_args())

    # Initialize trainer 
    trainer = Trainer(args)

    if args['task'] == 'train':
        trainer.train()

    elif args['task'] == 'resnet_train':
        trainer = ResnetTrainer(args)

    elif args['task'] == 'viz':
        trainer.visualize_attention(1)

    else:
        raise NotImplementedError(f"Unrecognized task {args['task']}")