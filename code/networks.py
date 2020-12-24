
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

	def __init__(self, config):
		
		super(BertSelfAttention, self).__init__()
		self.in_dim = config['in_dim'] 
		self.model_dim = config['model_dim']
		self.heads = config['mha_heads']
		self.pre_norm = config['pre_norm']
		self.query = nn.Linear(self.in_dim, self.heads*self.model_dim, bias=False)
		self.key = nn.Linear(self.in_dim, self.heads*self.model_dim, bias=False)
		self.value = nn.Linear(self.in_dim, self.heads*self.model_dim, bias=False)
		self.out = nn.Linear(self.heads*self.model_dim, self.in_dim, bias=False)
		self.norm = nn.LayerNorm(self.in_dim)


	def forward(self, x, return_attn_scores=False):
		''' Input has shape (bs, h, w, in_dim) '''
		
		bs, h, w, in_dim = x.size()
		sqrt_normalizer = math.sqrt(self.model_dim)

		if self.pre_norm:
			x = self.norm(x)

		Q = self.query(x).view(bs, h, w, self.heads, -1)							# Q -> (bs, h, w, heads, model_dim)
		K = self.key(x).view(bs, h, w, self.heads, -1)								# K -> (bs, h, w, heads, model_dim)
		V = self.value(x).view(bs, h, w, self.heads, -1)							# V -> (bs, h, w, heads, model_dim)

		# Q_ = torch.cat(Q.split(self.model_dim, -1), dim=0)						# Q_ -> (bs*heads, h, w, model_dim)
		# K_ = torch.cat(K.split(self.model_dim, -1), dim=0)						# K_ -> (bs*heads, h, w, model_dim)
		# V_ = torch.cat(V.split(self.model_dim, -1), dim=0)						# V_ -> (bs*heads, h, w, model_dim)

		correlation = torch.einsum('bijhd,bklhd->bijhkl', Q, K)						# (bs, h, w, heads, h, w)
		corr_size = correlation.size()		  					
		correlation = correlation.view(*corr_size[:-2], -1)/sqrt_normalizer		  	# (bs, h, w, heads, h*w)
		attn_probs = F.softmax(correlation, dim=-1).view(corr_size)					# (bs, h, w, heads, h, w)		
		
		logits = torch.einsum('bijhkl,bklhd->bijhd', attn_probs, V)					# logits -> (bs, h, w, heads, model_dim)
		logits = logits.contiguous().view(bs, h, w, -1)								# logits -> (bs, h, w, heads*model_dim)
		logits = self.out(logits)													# logits -> (bs, h, w, in_dim)

		# Residual connection
		out = logits + x
		if not self.pre_norm:
			out = self.norm(out)
		
		return out, attn_probs


class Learned2dRelativeSelfAttention(nn.Module):

	def __init__(self, config):
		
		super().__init__()
		self.use_attention_data = config.get('use_attention_data', False)  
		self.query_positional_score = config.get('query_positional_score', False)
		self.heads = config['mha_heads']
		self.in_dim = config['in_dim']
		self.model_dim = config['model_dim']
		
		max_position_embeddings = config['max_position_embedding'] 
		position_embedding_size = config['model_dim']

		if self.query_positional_score:
			position_embedding_size = config['model_dim'] // 2
		if config['position_encoding_size'] != -1:
			position_embedding_size = config['position_encoding_size']

		self.row_embeddings = nn.Embedding(2*max_position_embeddings-1, position_embedding_size)
		self.col_embeddings = nn.Embedding(2*max_position_embeddings-1, position_embedding_size)

		if not self.query_positional_score:
			self.head_keys_row = nn.Linear(position_embedding_size, self.heads, bias=False)
			self.col_keys_row = nn.Linear(position_embedding_size, self.heads, bias=False)

		# Linear transforms for query and key
		if self.use_attention_data or self.query_positional_score:
			self.query = nn.Linear(self.in_dim, self.model_dim*self.heads)

		if self.use_attention_data:
			self.key = nn.Linear(self.in_dim, self.model_dim*self.heads)

		self.dropout = nn.Dropout(config['attention_dropout_prob'])
		self.value = nn.Linear(self.in_dim*self.heads, self.in_dim)

		# Relative positional encoding
		deltas = torch.arange(max_position_embeddings).view(-1, 1) + torch.arange(max_position_embeddings).view(1, -1)
		relative_indices = deltas + max_position_embeddings - 1
		self.register_buffer("relative_indices", relative_indices)


	def compute_attention_scores(self, hidden_state, input_nhwc=True):
		''' 
		Input is hidden state of size (bs, c, h, w)
		Output is attention tensor of size (bs, w, h, n_heads, w, h)
		
		Options for attention scores:

		* Positional only
			Settings: use_attention_data=False, query_positional_score=False
			w_q^T r
			where w_q is a learned vector per head

		* Query and positional encoding (without query key attention scores)
			Uses only positional relation of query with other pixels.
			Settings: use_attention_data=False, query_positional_score=True
			X * W_Q * r

		* With data
			Uses position and content relations of query with other pixels.
			Settings: use_attention_data=True, query_positional_scores=True
			X * W_Q * W_K^T * X^T + X * W_Q * r
		'''
		if not input_nhwc:
			hidden_state = hidden_state.permute(0, 2, 3, 1)

		bs, h, w, in_dim = hidden_state.size()

		# Compute query data if needed
		if self.use_attention_data or self.query_positional_score:
			q = self.query(hidden_state)
			q = q.view(bs, w, h, self.heads, self.model_dim)

		# Compute key data if needed
		if self.use_attention_data:
			k = self.key(hidden_state)
			k = k.view(bs, w, h, self.heads, self.model_dim)

		# Compute attention scores based on position
		relative_indices = self.relative_indices[:w, :w].reshape(-1,)
		row_embeds = self.row_embeddings(relative_indices)

		relative_indices = self.relative_indices[:h, :h].reshape(-1,)
		col_embeds = self.col_embeddings(relative_indices)

		# Container for attention scores
		attention_scores_per_type = {}
		sqrt_normalizer = math.sqrt(self.model_dim)

		if not self.query_positional_score:
			# Key and query data are not used here
			row_scores = self.head_keys_row(row_embeds).view(1, w, 1, w, self.heads)
			col_scores = self.head_keys_col(col_embeds).view(h, 1, h, 1, self.heads)

			# For each pixel in the hidden_states' (h, w), we have n_heads number
			# of attention maps, each of size (h, w); so every pixel has a
			# positional and/or content based relation with every other pixel
			attention_scores = row_scores + col_scores 												# (h, w, h, w, n_heads)
			attention_scores = attention_scores.permute(0, 1, 4, 2, 3)								# (h, w, n_heads, h, w)
			attention_scores = attention_scores.unsqueeze(0)										# (1, h, w, n_heads, h, w)
			attention_scores_per_type['position_only'] = attention_scores

		else: 
			# Query positional scores only
			# First half of model_dim learns row encoding, second half learns col encoding
			q_row = q[:, :, :, :, :self.model_dim//2]
			q_col = q[:, :, :, :, self.model_dim//2:]

			row_scores = torch.einsum('bijhd,ikd->bijhk', q_row, row_embeds.view(w, w, -1))			# (bs, w, h, n_heads, w)
			col_scores = torch.einsum('bijhd,jld->bijhl', q_col, col_embeds.view(h, h, -1))			# (bs, w, h, n_heads, h)

			attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)					# (bs, w, h, n_heads, w, h)
			attention_scores = attention_scores.permute(0, 2, 1, 3, 5, 4)/sqrt_normalizer			# (bs, h, w, n_heads, h, w)
			attention_scores_per_type['query_position'] = attention_scores

		if self.use_attention_data:
			# Content information also used in computing attention
			# Content information is simply added to the positional information
			attn_content_scores = torch.einsum('bijhd,bklhd->bijhkl', q, k)							# (bs, w, h, n_heads, w, h)
			attn_content_scores = attn_content_scores.permute(0, 2, 1, 3, 5, 4)/sqrt_normalizer		# (bs, h, w, n_heads, h, w)
			attention_scores = attention_scores + attn_content_scores
			attention_scores_per_type['query_with_content'] = attention_scores

		return attention_scores, attention_scores_per_type


	def forward(self, hidden_state, return_attn_scores=False):

		assert len(hidden_state.size()) == 4, f'Hidden state should be 4d tensor, got size {hidden_state.size()}'
		bs, h, w, c = hidden_state.size()

		# Attention scores will have size (1 or bs, h, w, n_heads, h, w)
		attention_scores, attention_scores_per_type = self.compute_attention_scores(hidden_state)
		attn_size = attention_scores.size()
		attention_probs = F.softmax(attention_scores.contiguous().view(*attn_size[:-2], -1), dim=-1).view(attn_size)

		# If first dimensions is not batch size, view it so
		if attn_size[0] != bs:
			attention_probs = attention_probs.expand(bs, *attn_size[1:])

		attention_probs = self.dropout(attention_probs)
		input_values = torch.einsum('bijhkl,bkld->bijhd', attention_probs, hidden_state)
		input_values = input_values.contiguous().view(bs, h, w, -1)
		output_value = self.value(input_values)							# (bs, h, w, in_dim)

		if return_attn_scores:
			attention_scores_per_type['attention_scores'] = attention_scores
			attention_scores_per_type['attention_probs'] = attention_probs
			return output_value, attention_probs
		else:
			return output_value


class Feedforward(nn.Module):

	def __init__(self, config):
		
		super(Feedforward, self).__init__()
		in_dim = config['in_dim']
		hidden_dim = config['ff_dim']
		self.pre_norm = config['pre_norm']
		self.fc1 = nn.Linear(in_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_dim, in_dim)
		self.norm = nn.LayerNorm(in_dim)


	def forward(self, x):
		''' Input has shape (bs, n, in_dim) '''

		if self.pre_norm:
			x = self.norm(x)
		
		out = self.fc2(self.relu(self.fc1(x))) + x	
		
		if not self.pre_norm:
			out = self.norm(out)
		
		return out


class EncoderBlock(nn.Module):

	def __init__(self, config):
		super(EncoderBlock, self).__init__()

		# Choose attention type
		name = config.get('name', None)

		if name == 'bert':
			self.attention = BertSelfAttention(config)	
		elif name == 'relative':
			self.attention = Learned2dRelativeSelfAttention(config)
		else:
			raise NotImplementedError(f'Attention mechanism "{name}" either unspecified or invalid')

		self.feedfwd = Feedforward(config)


	def forward(self, x):
		x, att_scores = self.attention(x, return_attn_scores=True)
		x = self.feedfwd(x)
		return x, att_scores


class BertEncoder(nn.Module):

	def __init__(self, config):
		
		super(BertEncoder, self).__init__()
		self.num_blocks = config['num_encoder_blocks']
		self.blocks = nn.ModuleList([EncoderBlock(config) for i in range(self.num_blocks)])


	def forward(self, x, return_attn_scores=False):

		att_scores = {}
		for i in range(self.num_blocks):
			x, att = self.blocks[i](x)
			att_scores[f'layer_{i}'] = att

		if return_attn_scores:
			return x, att_scores
		else:
			return x


class ConvBottom(nn.Module):

	def __init__(self, config):
		
		super(ConvBottom, self).__init__()
		name = config.get('name', None)	
		assert name in list(MODEL_HELPER.keys()), f'name should be one of {list(MODEL_HELPER.keys())}'
		
		self.base_model = MODEL_HELPER[name](pretrained=config['pretrained'])
		self.res_layers = list(self.base_model.children())[0:4+config['block']]
		self.bottom = nn.Sequential(*self.res_layers)


	def forward(self, x):

		out = self.bottom(x)			# For resnet18 and CIFAR10, output has size (bs, 64, 8, 8)
		return out.permute(0, 2, 3, 1)


class ClassificationHead(nn.Module):

	def __init__(self, config):
		
		super(ClassificationHead, self).__init__()
		self.fc = nn.Linear(config['in_dim'], config['n_classes'])


	def forward(self, x):
		''' Size of input: (bs, h, w, c) '''

		bs, h, w, c = x.size()
		x = x.view(bs, -1, c).mean(dim=1)
		return F.log_softmax(self.fc(x), dim=-1)