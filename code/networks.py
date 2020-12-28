
""" 
BERT definition
[NOTE] Some stuff doesn't work correctly

"""

import math
import torch
import numbers
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import models


MODEL_HELPER = {
	'resnet18': {'net': models.resnet18, 'features': 64},
	'resnet50': {'net': models.resnet50, 'features': 256}
}


def gaussian_kernel_2d(mean, std_inv, size):
    """Create a 2D gaussian kernel
    Args:
        mean: center of the gaussian filter (shift from origin)
            (2, ) vector
        std_inv: standard deviation $Sigma^{-1/2}$
            can be a single number, a vector of dimension 2, or a 2x2 matrix
        size: size of the kernel
            pair of integer for width and height
            or single number will be used for both width and height
    Returns:
        A gaussian kernel of shape size.
    """
    if type(mean) is torch.Tensor:
        device = mean.device
    elif type(std_inv) is torch.Tensor:
        device = std_inv.device
    else:
        device = "cpu"

    # repeat the size for width, height if single number
    if isinstance(size, numbers.Number):
        width = height = size
    else:
        width, height = size

    # expand std to (2, 2) matrix
    if isinstance(std_inv, numbers.Number):
        std_inv = torch.tensor([[std_inv, 0], [0, std_inv]], device=device)
    elif std_inv.dim() == 0:
        std_inv = torch.diag(std_inv.repeat(2))
    elif std_inv.dim() == 1:
        assert len(std_inv) == 2
        std_inv = torch.diag(std_inv)

    # Enforce PSD of covariance matrix
    covariance_inv = std_inv.transpose(0, 1) @ std_inv
    covariance_inv = covariance_inv.float()

    # make a grid (width, height, 2)
    X = torch.cat(
        [
            t.unsqueeze(-1)
            for t in reversed(
                torch.meshgrid(
                    [torch.arange(s, device=device) for s in [width, height]]
                )
            )
        ],
        dim=-1,
    )
    X = X.float()

    # center the gaussian in (0, 0) and then shift to mean
    X -= torch.tensor([(width - 1) / 2, (height - 1) / 2], device=device).float()
    X -= mean.float()

    # does not use the normalize constant of gaussian distribution
    Y = torch.exp((-1 / 2) * torch.einsum("xyi,ij,xyj->xy", [X, covariance_inv, X]))

    # normalize
    # TODO could compute the correct normalization (1/2pi det ...)
    # and send warning if there is a significant diff
    # -> part of the gaussian is outside the kernel
    Z = Y / Y.sum()
    return Z


class BertSelfAttention(nn.Module):

	def __init__(self, config):
		
		super(BertSelfAttention, self).__init__()
		self.model_dim = config['model_dim']
		self.heads = config['mha_heads']
		self.pre_norm = config['pre_norm']
		self.query = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
		self.key = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
		self.value = nn.Linear(self.heads*self.model_dim, self.model_dim, bias=False)
		self.norm = nn.LayerNorm(self.model_dim)


	def forward(self, x, return_attn_scores=False):
		''' Input has shape (bs, h, w, in_dim) '''
		
		bs, h, w, in_dim = x.size()
		sqrt_normalizer = math.sqrt(self.model_dim)

		if self.pre_norm:
			x = self.norm(x)

		Q = self.query(x).view(bs, h, w, self.heads, -1)							# Q -> (bs, h, w, heads, model_dim)
		K = self.key(x).view(bs, h, w, self.heads, -1)								# K -> (bs, h, w, heads, model_dim)

		correlation = torch.einsum('bijhd,bklhd->bijhkl', Q, K)						# (bs, h, w, heads, h, w)
		corr_size = correlation.size()		  					
		correlation = correlation.view(*corr_size[:-2], -1)/sqrt_normalizer		  	# (bs, h, w, heads, h*w)
		attn_probs = F.softmax(correlation, dim=-1).view(corr_size)					# (bs, h, w, heads, h, w)		
		
		logits = torch.einsum('bijhkl,bkld->bijhd', attn_probs, x)					# logits -> (bs, h, w, heads, model_dim)
		logits = logits.contiguous().view(bs, h, w, -1)								# logits -> (bs, h, w, heads*model_dim)
		logits = self.value(logits)													# logits -> (bs, h, w, model_dim)

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
			self.query = nn.Linear(self.model_dim, self.model_dim*self.heads, bias=False)

		if self.use_attention_data:
			self.key = nn.Linear(self.model_dim, self.model_dim*self.heads, bias=False)

		self.dropout = nn.Dropout(config['attention_dropout_prob'])
		self.value = nn.Linear(self.model_dim*self.heads, self.model_dim, bias=False)

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
		output_value = self.value(input_values)							# (bs, h, w, model_dim)

		if return_attn_scores:
			attention_scores_per_type['attention_scores'] = attention_scores
			attention_scores_per_type['attention_probs'] = attention_probs
			return output_value, attention_probs
		else:
			return output_value


class GaussianSelfAttention(nn.Module):

	def __init__(self, config):
		
		super(GaussianSelfAttention, self).__init__()
		self.attention_blur_trick = config['attention_gaussian_blur_trick']			# They're G-blurring the final tensor for some reason
		self.isotropic_gaussian = config['attention_isotropic_gaussian']			# Circular spread of attention PDF
		self.mu_init_noise = config['mu_init_noise']								# Means initialization
		self.sigma_init_noise = config['sigma_init_noise']							# Std dev initialization
		self.config = config 

		self.heads = config['mha_heads']
		self.model_dim = config['model_dim']

		# Initialize attention centers noisily around 0 and covariances
		self.attention_centers = nn.Parameter(torch.zeros(self.heads, 2).normal_(0, self.mu_init_noise))

		if self.isotropic_gaussian:
			# Covariance is a c * I type diagonal matrix, so we maintain only one scalar
			attention_spreads = 1 + torch.zeros(self.heads).normal_(0, self.sigma_init_noise)
		
		else:
			# Why is it inverse covariance? The original code says so
			attention_spreads = torch.eye(2).unsqueeze(0).repeat(self.heads, 1, 1)							# (n_heads, 2, 2)
			attention_spreads += torch.zeros_like(attention_spreads).normal_(0, self.sigma_init_noise)		# Added noise

		self.attention_spreads = nn.Parameter(attention_spreads)
		self.value = nn.Linear(self.heads*self.model_dim, self.model_dim, bias=False)

		# Gaussian blur trick. Not sure what benefit this gives
		# relative encoding grid (delta_x, delta_y, delta_x**2, delta_y**2, delta_x * delta_y)
		if self.attention_blur_trick:
			MAX_WIDTH_HEIGHT = 50
			range_ = torch.arange(MAX_WIDTH_HEIGHT)
			grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid([range_, range_])], dim=-1)			# (MWH, MWH, 2)
			rel_idx = grid.unsqueeze(0).unsqueeze(0) - grid.unsqueeze(-2).unsqueeze(-2)						# (MWH, MWH, MWH, MWH, 2)
			R = torch.cat([rel_idx, rel_idx**2, (rel_idx[..., 0] * rel_idx[..., 1]).unsqueeze(-1)], dim=-1)	# (MWH, MWH, MWH, MWH, MWH)
			self.register_buffer('R', R.float())
			self.dropout = nn.Dropout(config['attention_dropout_prob'])


	def get_heads_target_vectors(self):

		# This part extracts the elements of the covariance of the attention gaussians
		# Key: Covariance -> [[a, b]
		#                     [b, c]]

		if self.isotropic_gaussian:
			a = c = self.attention_spreads ** 2
			b = torch.zeros_like(self.attention_spreads)
		else:
			inv_covariance = torch.einsum('hij,hkj->hik', [self.attention_spreads, self.attention_spreads])
			a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]

		mu_1, mu_2 = self.attention_centers[:, 0], self.attention_centers[:, 1]

		# I don't know what this is
		target = -0.5 * torch.stack([
			-2 * (a*mu_1 + b*mu_2),
			-2 * (c*mu_2 + b*mu_1),
			a,
			c,
			2 * b
		], dim=-1)

		return target


	def get_attention_probs(self, height, width):
		''' 
		Compute positional attention for image of size (height, width) 
		Returns tensor of probs (h, w, n_heads, h, w)
		'''
		u = self.get_heads_target_vectors()

		attention_scores = torch.einsum('ijkld,hd->ijhkl', [self.R[:height, :width, :height, :width, :], u])
		attention_probs = F.softmax(attention_scores.view(height, width, self.heads, -1), dim=-1)
		attention_probs = attention_probs.view(height, width, self.heads, height, width)

		return attention_probs


	def reset_heads(self):

		device = self.attention_spreads.data.device
		reset_heads_mask = torch.zeros(self.n_heads, device=device, dtype=torch.bool)
		for head in heads:
			reset_heads_mask[head] = 1

		# Reinitialize mu and sigma of heads
		self.attention_centers.data[reset_heads_mask].zero_().normal_(0.0, self.mu_init_noise)
		
		if self.isotropic_gaussian:
			self.attention_spreads.ones_().normal_(0, self.sigma_init_noise)
		else:
			self.attention_spreads.zero_().normal_(0, self.sigma_init_noise)
			self.attention_spreads[:, 0, 0] += 1
			self.attention_spreads[:, 1, 1] += 1

		# Reinitialize value matrix for all heads
		mask = torch.zeros(self.n_heads, self.model_dim*self.heads, dtype=torch.bool)
		for head in heads:
			mask[head] = 1

		mask = mask.view(-1).contiguous()
		self.value.weight.data[:, mask].normal_(0.0, self.config['initializer_range'])


	def blurred_attention(self, X):
		"""
		Compute the weighted average according to gaussian attention without
        computing explicitly the attention coefficients.
        Args:
            X (tensor): shape (batch, width, height, dim)
        Output:
            shape (batch, width, height, dim x num_heads)
        """
		b, h, w, d_total = X.size()
		Y = X.permute(0, 3, 1, 2).contiguous()

		kernels = []
		kernel_width = kernel_height = 7
		
		for mean, std in zip(self.attention_centers, self.attention_spreads):
			conv_weights = gaussian_kernel_2d(mean, std, size=(kernel_height, kernel_width))
			conv_weights = conv_weights.view(1, 1, kernel_height, kernel_width).repeat(d_total, 1, 1, 1)
			kernels.append(conv_weights)

		weights = torch.cat(kernels)
		padding_width = (kernel_width - 1)//2
		padding_height = (kernel_height - 1)//2 
		out = F.conv2d(Y, weights, groups=d_total, padding=(padding_height, padding_width))

		# Renormalize for padding
		all_one_input = torch.ones(1, d_total, h, w, device=X.device)
		normalizer = F.conv2d(all_one_input, weights, groups=d_total, padding=(padding_height, padding_width))
		out /= normalize

		return out.permute(0, 2, 3, 1).contiguous()


	def forward(self, hidden_state, return_attn_scores=True):

		assert len(hidden_state.size()) == 4, f'Bad hidden state shape {hidden_state.size()}'
		b, h, w, c = hidden_state.size()

		if self.attention_blur_trick:
			attention_probs = self.get_attention_probs(h, w)
			attention_probs = self.dropout(attention_probs)

			input_values = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_state)
			input_values = input_values.contiguous().view(b, h, w, -1)
		else:
			input_values = self.blurred_attention(hidden_state)

		output_value = self.value(input_values)

		if return_attn_scores:
			return output_value, attention_probs
		else:
			return output_value


class Feedforward(nn.Module):

	def __init__(self, config):
		
		super(Feedforward, self).__init__()
		model_dim = config['model_dim']
		hidden_dim = config['ff_dim']
		self.pre_norm = config['pre_norm']
		self.fc1 = nn.Linear(model_dim, hidden_dim)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(hidden_dim, model_dim)
		self.norm = nn.LayerNorm(model_dim)


	def forward(self, x):
		''' Input has shape (bs, n, model_dim) '''

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
		elif name == 'gaussian':
			self.attention = GaussianSelfAttention(config)
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
		self.config = config

		# In channels to feature upscaler
		if self.config['pool_with_resnet']:
			name = config.get('name', None)	
			assert name in list(MODEL_HELPER.keys()), f'name should be one of {list(MODEL_HELPER.keys())}'
			
			base_model = MODEL_HELPER[name]['net'](pretrained=config['pretrained'])
			channels = MODEL_HELPER[name]['features']
			res_layers = list(base_model.children())[0:4+config['block']]
			self.bottom = nn.Sequential(*res_layers)

			a = torch.rand((1, 3, 32, 32))
			with torch.no_grad():
				out = self.bottom(a)
			in_features = out.size(1)

		elif self.config['pool_concatenate_size'] > 1:
			in_features = 3 * self.config['pool_concatenate_size'] ** 2
			
		else:
			in_features = 3

		self.feature_upscale = nn.Linear(in_features, config['model_dim'])


	def downsample_concatenate(self, x, kernel):
		''' To be used if pooling is not with resnet '''

		assert (self.config['pool_concatenate_size'] > 1) & (not self.config['pool_with_resnet']), "Something's wrong, I can feel it"
		b, h, w, c = x.size()
		y = x.contiguous().view(b, h, w//kernel, c*kernel)
		y = y.permute(0, 2, 1, 3).contiguous()
		y = y.view(b, w//kernel, h//kernel, kernel*kernel*c).contiguous() 
		y = y.permute(0, 2, 1, 3).contiguous() 
		return y


	def forward(self, x):
		''' x has size (bs, c, h, w) '''

		if self.config['pool_with_resnet']:
			features = self.bottom(x)						# For resnet50 and CIFAR10, output has size (bs, 256, 8, 8)
			features = features.permute(0, 2, 3, 1)			# Convert to NHWC; required for attention mechanism

		elif self.config['pool_concatenate_size'] > 1:
			x = x.permute(0, 2, 3, 1)															# Convert to NHWC
			features = self.downsample_concatenate(x, self.config['pool_concatenate_size'])		# (bs, h//cs, w//cs, cs*cs*c)

		else:
			features = x.permute(0, 2, 3, 1)

		return self.feature_upscale(features)				# (bs, h, w, model_dim)


class ClassificationHead(nn.Module):

	def __init__(self, config):
		
		super(ClassificationHead, self).__init__()
		self.fc = nn.Linear(config['model_dim'], config['n_classes'])


	def forward(self, x):
		''' Size of input: (bs, h, w, c) '''

		bs, h, w, c = x.size()
		x = x.view(bs, -1, c).mean(dim=1)
		return F.log_softmax(self.fc(x), dim=-1)
