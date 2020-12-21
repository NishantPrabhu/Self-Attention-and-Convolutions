
""" 
BERT definition
[NOTE] Experimental script

"""

import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 


class MultiHeadSelfAttention(nn.Module):

	def __init__(self, in_dim, hidden_dim, heads=9, pre_norm=False):
		
		super(MultiHeadSelfAttention, self).__init__()
		self.hidden_dim = hidden_dim
		self.heads = heads
		self.query = nn.Linear(in_dim, heads*hidden_dim)
		self.key = nn.Linear(in_dim, heads*hidden_dim)
		self.value = nn.Linear(in_dim, heads*hidden_dim)
		self.out = nn.Linear(heads*hidden_dim, in_dim)
		self.norm = nn.LayerNorm(in_dim)
		self.pre_norm = pre_norm


	def forward(self, x):
		''' Input has shape (bs, n, in_dim) '''
		
		if self.pre_norm:
			x = self.norm(x)

		Q = self.query(x)																			  # Q -> (bs, n, heads*hidden_dim)
		K = self.key(x)																				  # K -> (bs, n, heads*hidden_dim)
		V = self.value(x)																			  # V -> (bs, n, heads*hidden_dim)

		Q_ = torch.cat(Q.split(self.hidden_dim, 2), dim=0)										      # Q_ -> (bs*heads, n, hidden_dim)
		K_ = torch.cat(K.split(self.hidden_dim, 2), dim=0)											  # K_ -> (bs*heads, n, hidden_dim)
		V_ = torch.cat(V.split(self.hidden_dim, 2), dim=0)											  # V_ -> (bs*heads, n, hidden_dim)
		att_scores = F.softmax(torch.bmm(Q_, K_.transpose(1, 2))/math.sqrt(self.hidden_dim), dim=-1)  # att_scores -> (bs*heads, n, n)
		
		logits = torch.bmm(att_scores, V_)															  # logits -> (bs*heads, n, hidden_dim)
		logits = torch.cat(logits.split(Q.size(0), 0), dim=2)										  # logits -> (bs, n, heads*hidden_dim)
		logits = self.out(logits)																	  # logits -> (bs, n, in_dim)

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

	def __init__(self, in_dim, hidden_dim, mha_heads=9, pre_norm=False):
		super(EncoderBlock, self).__init__()
		self.attention = MultiHeadSelfAttention(in_dim, hidden_dim, mha_heads, pre_norm)
		self.feedfwd = Feedforward(in_dim, hidden_dim, pre_norm)

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
				hidden_dim=config['model_dim'], 
				mha_heads=config['mha_heads'], 
				pre_norm=config['pre_norm']
			) for i in range(self.num_blocks)
		])


	def forward(self, x):

		att_scores = {}
		for i in range(self.num_blocks):
			x, att = self.blocks[i](x)
			att_scores[f'block_{i}'] = att

		return x, att_scores



if __name__ == '__main__':

	config = {
		'in_dim': 3,
		'model_dim': 64,
		'mha_heads': 9,
		'pre_norm': False,
		'num_encoder_blocks': 6
	}

	encoder = BertEncoder(config)
	x = torch.rand((4, 16, 3))
	out, att_scores = encoder(x)
	print(out.size())
	print()
	for k, v in att_scores.items():
		print(v.size())