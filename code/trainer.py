
""" 
Training functions.
"""

import torch 
import torch.nn as nn
import networks 
import train_utils
import numpy as np
import wandb
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


class Trainer:

	def __init__(self, config, device):
		
		self.config = config
		self.device = device

		# Networks and optimizers
		self.encoder = networks.BertEncoder(self.config['bert_encoder']).to(self.device)
		self.conv_bottom = networks.ConvBottom(**self.config['conv_bottom']).to(self.device)
		self.clf_head = networks.ClassificationHead(**self.config['clf_head']).to(self.device)
		
		self.optim = train_utils.get_optimizer(
			config = self.config['optimizer'], 
			params = list(self.encoder.parameters())+list(self.conv_bottom.parameters())+list(self.clf_head.parameters())
		)
		self.scheduler, self.warmup_epochs = train_utils.get_scheduler(
			config = {**self.config['scheduler'], 'epochs': self.config['epochs']}, 
			optimizer = self.optim
		)

		# Warmup handling
		if self.warmup_epochs > 0:
			self.warmup_rate = self.optim.param_groups[0]['lr']/self.warmup_epochs

		# Losses and performance monitoring
		self.criterion = nn.NLLLoss()
		self.best_val_acc = 0
		wandb.init('self-attention-cnn-test')

	
	def train_one_step(self, data):

		img, labels = data[0].to(self.device), data[1].to(self.device)
		out = self.clf_head(self.encoder(self.conv_bottom(img), return_attn_scores=False))
		
		# Loss and update
		self.optim.zero_grad()
		loss = self.criterion(out, labels)	
		loss.backward()
		self.optim.step()

		# Correct predictions
		pred = out.argmax(dim=-1)	
		correct = pred.eq(labels.view_as(pred)).sum().item()
		
		return loss.item(), correct


	def validate_one_step(self, data):

		img, labels = data[0].to(self.device), data[1].to(self.device)
		out = self.clf_head(self.encoder(self.conv_bottom(img), return_attn_scores=False))
		
		loss = self.criterion(out, labels)
		pred = out.argmax(dim=-1)	
		correct = pred.eq(labels.view_as(pred)).sum().item()
		
		return loss.item(), correct


	def save_state(self, epoch):
		''' For resuming from run breakages, etc '''
		
		state = {
			'epoch': epoch,
			'encoder': self.encoder.state_dict(),
			'conv': self.conv_bottom.state_dict(),
			'clf': self.clf_head.state_dict(),
			'optim': self.optim.state_dict(),
			'scheduler': self.scheduler.state_dict()
		}
		torch.save(state, '../saved_data/last_state.ckpt')


	def save_data(self):
		
		data = {
			'encoder': self.encoder.state_dict(),
			'conv': self.conv_bottom.state_dict(),
			'clf': self.clf_head.state_dict()
		}
		torch.save(data, '../saved_data/best_model.ckpt')


	def load_state(self):
		
		if os.path.exists('../saved_data/last_state.ckpt'):
			last_state = torch.load('../saved_data/last_state.ckpt')
			done_epochs = last_state['epoch']-1
			self.encoder.load_state_dict(last_state['encoder'])
			self.conv_bottom.load_state_dict(last_state['conv']) 
			self.clf_head.load_state_dict(last_state['clf'])
			self.optim.load_state_dict(last_state['optim'])
			self.scheduler.load_state_dict(last_state['scheduler'])
			print("\n[INFO] Loaded last saved state successfully\n")
			return done_epochs
		
		else:
			print("\n[INFO] No saved state found, starting fresh\n")
			return 0


	def adjust_learning_rate(self, epoch):
		
		if epoch < self.warmup_epochs:
			for group in self.optim.param_groups:
				group['lr'] = 1e-12 + (epoch * self.warmup_rate)
		else:
			self.scheduler.step()


	def visualize_attention(self, epoch, val_loader):
		''' Generate attention on a batch of 100 images and plot them '''	

		batch = next(iter(val_loader))
		img = batch[0].to(self.device)
		fvecs, attn_scores = self.encoder(self.conv_bottom(img), return_attn_scores=True)

		# attn_scores is a dict with num_encoder_blocks items 
		# Each item value has size (batch_size, num_heads, num_pixels, num_pixels)
		heads = self.config['bert_encoder']['mha_heads']
		layers = self.config['bert_encoder']['num_encoder_blocks']
		fig = plt.figure(figsize=(10, 7))
		count = 1

		for name, attn in attn_scores.items():

			# Average attention over batch
			attn = attn.mean(dim=0).detach().cpu().numpy()		# Size (n_heads, n_pix, n_pix)
			
			for i in range(attn.shape[0]):
				fig.add_subplot(layers, heads, count)
				plt.imshow(attn[i], cmap='gray')
				plt.axis('off')
				count += 1

		plt.tight_layout(pad=1)
		plt.savefig(f'../saved_data/plots/attention_maps_{epoch+1}', pad_inches=0.05)


	def train(self, train_loader, val_loader):

		# Load last state if it exists
		done_epochs = self.load_state()

		# Train
		for epoch in range(self.config['epochs']-done_epochs):

			# Change learning rate
			self.adjust_learning_rate(epoch)

			train_losses, train_correct_count = [], 0
			pbar = tqdm(total=len(train_loader), desc=f"[Train epoch] {epoch+1} - [LR] {round(self.optim.param_groups[0]['lr'], 4)}")

			for batch in train_loader:
				train_loss, train_correct = self.train_one_step(batch)
				train_losses.append(train_loss)
				train_correct_count += train_correct
				wandb.log({'Train loss': train_loss})
				pbar.update(1)

			av_train_loss, train_acc = np.mean(train_losses), train_correct_count/len(train_loader.dataset)
			pbar.set_description(f'[Train epoch] {epoch+1} - [Train loss] {round(av_train_loss, 4)} - [Train acc] {round(train_acc, 4)}')
			pbar.close()
			wandb.log({'Train accuracy': train_acc, 'Epoch': epoch+1})

			if epoch % self.config['eval_every'] == 0:
				val_losses, val_correct_count = [], 0
				pbar = tqdm(total=len(val_loader), desc=f'[Valid epoch] {epoch+1}')
				
				for batch in val_loader:
					val_loss, val_correct = self.validate_one_step(batch)
					val_losses.append(val_loss)
					val_correct_count += val_correct
					pbar.update(1)

				av_val_loss, val_acc = np.mean(val_losses), val_correct_count/len(val_loader.dataset)
				pbar.set_description(f'[Valid epoch] {epoch+1} - [Valid loss] {round(av_val_loss, 4)} - [Valid acc] {round(val_acc, 4)}')
				pbar.close()
				wandb.log({'Validation loss': av_val_loss, 'Validation accuracy': val_acc, 'Epoch': epoch+1})

				if val_acc > self.best_val_acc:
					self.best_val_acc = val_acc
					self.save_data()

				# Visualize attention
				self.visualize_attention(epoch, val_loader)