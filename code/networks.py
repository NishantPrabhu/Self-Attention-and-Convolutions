
""" 
BERT definition
[NOTE] Experimental script

"""

import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models


MODEL_HELPER = {
	'resnet18': models.resnet18
}


class BertSelfAttention(nn.Module):

	def __init__(self, in_dim, model_dim, heads=9, pre_norm=False):
		
		super(BertSelfAttention, self).__init__()
		self.model_dim = model_dim
		self.heads = heads
		self.query = nn.Linear(in_dim, heads*model_dim)
		self.key = nn.Linear(in_dim, heads*model_dim)
		self.value = nn.Linear(in_dim, heads*model_dim)
		self.out = nn.Linear(heads*model_dim, in_dim)
		self.norm = nn.LayerNorm(in_dim)
		self.pre_norm = pre_norm


	def forward(self, x):
		''' Input has shape (bs, h, w, in_dim) '''
		
		if self.pre_norm:
			x = self.norm(x)

		Q = self.query(x)															# Q -> (bs, n, heads*model_dim)
		K = self.key(x)																# K -> (bs, n, heads*model_dim)
		V = self.value(x)															# V -> (bs, n, heads*model_dim)

		Q_ = torch.cat(Q.split(self.model_dim, -1), dim=0)							# Q_ -> (bs*heads, n, model_dim)
		K_ = torch.cat(K.split(self.model_dim, -1), dim=0)							# K_ -> (bs*heads, n, model_dim)
		V_ = torch.cat(V.split(self.model_dim, -1), dim=0)							# V_ -> (bs*heads, n, model_dim)
		correlation = torch.einsum('bid,bjd->bij', Q_, K_)		  					
		att_scores = F.softmax(correlation/math.sqrt(self.model_dim), dim=-1)  		# att_scores -> (bs*heads, n, n)
		
		logits = torch.bmm(att_scores, V_)											# logits -> (bs*heads, n, model_dim)
		logits = torch.cat(logits.split(Q.size(0), 0), dim=2)						# logits -> (bs, n, heads*model_dim)
		logits = self.out(logits)													# logits -> (bs, n, in_dim)

		# Residual connection
		out = logits + x
		if not self.pre_norm:
			out = self.norm(out)

		# Output attention scores as (batch_size, heads, n, n)
		headwise_att_scores = att_scores.split(out.size(0), 0)
		headwise_att_scores = [h.unsqueeze(1) for h in headwise_att_scores]
		headwise_att_scores = torch.cat(headwise_att_scores, dim=1)
		
		return out, headwise_att_scores


class Feedforward(nn.Module):

	def __init__(self, in_dim, hidden_dim, pre_norm=False):
		
		super(Feedforward, self).__init__()
		self.fc1 = nn.Linear(in_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_dim, in_dim)
		self.norm = nn.LayerNorm(in_dim)
		self.pre_norm = pre_norm


	def forward(self, x):
		''' Input has shape (bs, n, in_dim) '''

		if self.pre_norm:
			x = self.norm(x)
		
		out = self.fc2(self.relu(self.fc1(x))) + x	
		if not self.pre_norm:
			out = self.norm(out)
		
		return out


class EncoderBlock(nn.Module):

	def __init__(self, in_dim, model_dim, ff_dim, mha_heads=9, pre_norm=False):
		super(EncoderBlock, self).__init__()
		self.attention = BertSelfAttention(in_dim, model_dim, mha_heads, pre_norm)
		self.feedfwd = Feedforward(in_dim, ff_dim, pre_norm)

	def forward(self, x):
		x, att_scores = self.attention(x)
		x = self.feedfwd(x)
		return x, att_scores


class BertEncoder(nn.Module):

	def __init__(self, config):
		
		super(BertEncoder, self).__init__()
		self.heads = config['mha_heads']
		self.num_blocks = config['num_encoder_blocks']
		self.blocks = nn.ModuleList([
			EncoderBlock(
				in_dim=config['in_dim'], 
				model_dim=config['model_dim'], 
				ff_dim=config['ff_dim'],
				mha_heads=config['mha_heads'], 
				pre_norm=config['pre_norm']
			) for i in range(self.num_blocks)
		])


	def forward(self, x, return_attn_scores=False):

		att_scores = {}
		for i in range(self.num_blocks):
			x, att = self.blocks[i](x)
			att_scores[f'layer_{i}'] = att

		# Mean pool the output along pixel dimension
		out = x.mean(dim=1)

		if return_attn_scores:
			return out, att_scores
		else:
			return out


class ConvBottom(nn.Module):

	def __init__(self, name='resnet18', block=1, pretrained=False):
		super(ConvBottom, self).__init__()	
		assert name in list(MODEL_HELPER.keys()), f'name should be one of {list(MODEL_HELPER.keys())}'
		self.base_model = MODEL_HELPER[name](pretrained=pretrained)
		self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(64)
		self.relu1 = nn.ReLU()
		self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
		self.res_layers = list(self.base_model.children())[4:4+block]
		self.bottom = nn.Sequential(self.conv1, self.bn1, self.relu1, self.maxpool1, *self.res_layers)

	def forward(self, x):
		out = self.bottom(x)			# For resnet18 and CIFAR10, output has size (bs, 64, 8, 8)	
		bs, c, h, w = out.size()
		return out.view(bs, -1, c)


class ClassificationHead(nn.Module):

	def __init__(self, in_dim, n_classes):
		super(ClassificationHead, self).__init__()
		self.fc = nn.Linear(in_dim, n_classes)

	def forward(self, x):
		return F.log_softmax(self.fc(x), dim=-1)