
""" 
Attention mechanisms.

Authors: Nishant Prabhu, Mukund Varma T
"""

import math
import numbers 
import torch
import torch.nn as nn
import torch.nn.functional as F 


# ===================================================================================================================================
#  Helper functions
# 
#    - gaussian_kernel_2d: Generates a 2d gaussian kernel
# ===================================================================================================================================


def guassian_kernel_2d(mean, std_inv, kernel_size=(3, 3)):
    """ 
    Generates a 2D gaussian kernel. 

    Args:
        mean: Center of gaussian filter (shift from origin) 
                (2,) vector
        std_inv: Standard deviation $Sigma^{-1/2}$ 
                single number, (2,) vector or (2, 2) tensor 
        kernel_size: Tuple (width, height) of the kernel's size 
                If integer is passed, square kernel is returned  

    Returns:
        Gaussian kernel of size (width, height) 
    """
    if type(mean) is torch.Tensor:
        device = mean.device 
    elif type(std_inv) is torch.Tensor:
        device = std_inv.device 
    else:
        device = 'cpu'

    if isinstance(size, number.Number):
        width = height = int(size)
    else:
        width, height = size  

    # Expand std to (2, 2) matrix and enforce PSD 
    if isinstance(std_inv, number.Number):
        std_inv = torch.Tensor([[std_inv, 0], [0, std_inv]], device=device)
    elif std_inv.dim() == 0:
        std_inv = torch.diag(std_inv.repeat(2))
    elif std_inv.dim() == 1:
        assert len(std_inv) == 2
        std_inv = torch.diag(std_inv)

    covariance_inv = std_inv.transpose(0, 1) @ std_inv
    covariance_inv = covariance_inv.float()

    # Make a grid of size (width, height, 2)
    X = torch.cat([t.unsqueeze(-1) for t in reversed(torch.meshgrid([torch.arange(s, device=device) for s in [width, height]]))], dim=-1)
    X = X.float()

    # Center the gaussian to (0, 0) and shift to mean 
    X -= torch.Tensor([(width-1)/2, (height-1)/2], device=device).float()
    X -= mean.float()

    # Compute the normal distribution logits
    z = torch.exp(-0.5 * torch.einsum('xyi,ij,xyj->xy', [X, covariance_inv, X]))
    z /= math.sqrt(2*math.pi) * torch.pow(torch.det(covariance_inv), mean.size()/2)

    return z


# ===================================================================================================================================
#  Attention mechanisms
#
#    - BertSelfAttention: Usual BERT's self attention mechanism
#    - Learned2dRelativeSelfAttention: Self attention based on relative shift among pixels and (optinally) content information
#    - GaussianAttention: Positional self attention with Gaussian priors
# =================================================================================================================================== 


class BertSelfAttention(nn.Module):
    ''' 
    Standard self attention mechanism for BERT 
    '''

    def __init__(self, config):
        
        super().__init__()
        self.model_dim = config['model_dim']
        self.heads = config['num_heads']
        self.pre_norm = config['pre_norm']
        self.hierarchical = config['hierarchical_weight_sharing']

        # Layers  
        self.query = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
        self.key = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
        self.value = nn.Linear(self.heads*self.model_dim, self.model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(config.get('attention_dropout_prob', 0.1))

        # [TEMP] For hierarchical attention, to keep key tensor fixed
        self.K = None

    
    def forward(self, x):
        ''' Input has size (bs, h, w, model_dim) '''

        if len(x.size()) == 4:
            bs, h, w, _ = x.size() 
            x = x.view(bs, -1, self.model_dim)                                                      # (bs, n, model_dim)

        bs = x.size(0)
        sqrt_normalizer = math.sqrt(self.model_dim)
        if self.pre_norm:
            x = self.layer_norm(x)

        Q = self.query(x).view(bs, self.heads, -1, self.model_dim)                                  # (bs, heads, n, model_dim)
        if self.hierarchical:
            if self.K is None:
                self.K = self.key(x).view(bs, self.heads, -1, self.model_dim)                       # (bs, heads, n, model_dim)
        else:
            self.K = self.key(x).view(bs, self.heads, -1, self.model_dim)                           # (bs, heads, n, model_dim)

        attention_scores = torch.einsum('bhid,bhjd->bhij', [Q, self.K])                             # (bs, heads, n, n)
        attention_probs = F.softmax(attention_scores/sqrt_normalizer, dim=-1)                       # (bs, heads, n, n) softmaxed
        attention_probs = self.dropout(attention_probs)

        out = torch.einsum('bhij,bjd->bhid', [attention_probs, x])                                  # (bs, heads, n, model_dim)
        out = out.permute(0, 2, 1, 3).contiguous()                                                  # (bs, n, heads, model_dim)
        out = self.value(out.view(bs, -1, self.heads*self.model_dim))                               # (bs, n, model_dim)

        # Residual connection
        out = out + x
        if not self.pre_norm:
            out = self.layer_norm(out)

        return out, attention_probs


class Learned2dRelativeSelfAttention(nn.Module):
    '''
    Self attention based on relative shifts among pixels and (optionally) content information.
    The following options are available for this attention mechanism.

    * Positional attention only: [use_attention_data=False, query_positional_score=False]
        Neither the query's positional embedding nor image features are used.
        w_q^T r 

    * Query positional score: [use_attention_data=False, query_positional_score=True]
        Uses positional attention of query pixel with other pixels.
        X * W_Q * r

    * Positional and content based: [use_attention_data=True, query_positional_score=True]
        Uses positional attention and adds content based attention to it.
        (X * W_Q * r) + (X * W_Q * W_K^T * X^T) 
    '''
    
    def __init__(self, config):

        super().__init__()
        self.use_attention_data = config.get('use_attention_data', False)
        self.query_positional_score = config.get('query_positional_score', False)
        self.heads = config['num_heads']
        self.model_dim = config['model_dim']
        self.pre_norm = config['pre_norm']
        self.hierarchical = config['hierarchical_weight_sharing']

        max_position_embeddings = config.get('max_position_embeddings', 16)         # Used in defining relative pixel indices
        position_embedding_size = self.model_dim                                    # Dimension of positional embedding

        if self.query_positional_score:
            # If query embeddings are being used, we assume half the features ...
            # ... carry row embeddings and the other half carry column embeddings
            position_embedding_size = self.model_dim // 2  

        if config.get('position_embedding_size', -1) != -1:
            position_embedding_size = config['position_embedding_size']

        # Embeddings
        self.row_embeddings = nn.Embedding(2*max_position_embeddings-1, position_embedding_size)
        self.col_embeddings = nn.Embedding(2*max_position_embeddings-1, position_embedding_size)

        if not self.query_positional_score:
            # The positional embeddings must be transformed into a space ...
            # ... with as many features as attention heads
            self.head_keys_row = nn.Linear(position_embedding_size, self.heads)
            self.head_keys_col = nn.Linear(position_embedding_size, self.heads)

        # Linear transforms for key and query (to use content attention)
        if self.use_attention_data or self.query_positional_score:
            self.query = nn.Linear(self.model_dim, self.model_dim*self.heads, bias=False)

        if self.use_attention_data:
            self.key = nn.Linear(self.model_dim, self.model_dim*self.heads, bias=False)

        self.dropout = nn.Dropout(config.get('attention_dropout_prob', 0.1))
        self.value = nn.Linear(self.model_dim*self.heads, self.model_dim)
        self.layer_norm = nn.LayerNorm(self.model_dim)

        # Relative positional indices
        deltas = torch.arange(max_position_embeddings).view(1, -1) - torch.arange(max_position_embeddings).view(-1, 1)
        relative_indices = deltas + max_position_embeddings - 1
        self.register_buffer('relative_indices', relative_indices)

        # [TEMP] For hierarchical attention, to keep key tensor fixed
        self.k = None


    def compute_attention_scores(self, hidden_state):
        ''' 
        hidden_state has size (bs, h, w, model_dim)
        Output attention has size (bs, w, h, num_heads, w, h)
        '''
        bs, h, w, _ = hidden_state.size()
        sqrt_normalizer = math.sqrt(self.model_dim)

        # Compute query data if needed
        if self.query_positional_score or self.use_attention_data:
            q = self.query(hidden_state).view(bs, w, h, self.heads, self.model_dim)                 # (bs, w, h, heads, model_dim)

        if self.use_attention_data:
            if self.hierarchical:
                if self.k is None:
                    self.k = self.key(hidden_state).view(bs, w, h, self.heads, self.model_dim)      # (bs, w, h, heads, model_dim)
            else:
                self.k = self.key(hidden_state).view(bs, w, h, self.heads, self.model_dim)          # (bs, w, h, heads, model_dim)

        # Compute row and column embeddings based on position
        rel_idx = self.relative_indices[:w, :w].reshape(-1,)
        row_embeds = self.row_embeddings(rel_idx)                                                   # (w^2, position_embedding_size)

        rel_idx = self.relative_indices[:h, :h].reshape(-1,)
        col_embeds = self.col_embeddings(rel_idx)                                                   # (h^2, position_embedding_size)

        # Compute attention scores
        if not self.query_positional_score:
            # No query positional data or attention data used here
            row_scores = self.head_keys_row(row_embeds).view(1, w, 1, w, self.heads)                # (1, w, 1, w, heads)
            col_scores = self.head_keys_col(col_embeds).view(h, 1, h, 1, self.heads)                # (h, 1, h, 1, heads)

            attention_scores = row_scores + col_scores                                              # (h, w, h, w, heads)
            attention_scores = attention_scores.permute(0, 1, 4, 2, 3)                              # (h, w, heads, h, w)
            attention_scores = attention_scores.unsqueeze(0)                                        # (1, h, w, heads, h, w)

        else:
            # Query positional scores used, half features encode row embeddings, other half encode column embeddings
            q_row = q[:, :, :, :, :self.model_dim//2]
            q_col = q[:, :, :, :, self.model_dim//2:]
            row_scores = torch.einsum('bijhd,ikd->bijhk', q_row, row_embeds.view(w, w, -1))     # (bs, w, h, heads, w)
            col_scores = torch.einsum('bijhd,jld->bijhl', q_col, col_embeds.view(h, h, -1))     # (bs, w, h, heads, h)

            attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)                  # (bs, w, h, heads, w, h)
            attention_scores = attention_scores / sqrt_normalizer           
            attention_scores = attention_scores.permute(0, 2, 1, 3, 5, 4)                           # (bs, h, w, heads, h, w)

        if self.use_attention_data:
            # Compute content based attention scores
            att_content_scores = torch.einsum('bijhd,bklhd->bijhkl', q, self.k)                     # (bs, w, h, heads, w, h)
            att_content_scores = att_content_scores/sqrt_normalizer
            att_content_scores = att_content_scores.permute(0, 2, 1, 3, 5, 4)                       # (bs, h, w, heads, h, w)
            attention_scores = attention_scores + att_content_scores                                # (bs, h, w, heads, h, w)

        return attention_scores


    def forward(self, hidden_state):

        bs, h, w, _ = hidden_state.size()
        if self.pre_norm:
            hidden_state = self.layer_norm(hidden_state)

        attention_scores = self.compute_attention_scores(hidden_state)                              # (bs, h, w, heads, h, w)
        attn_size = attention_scores.size() 
        attention_scores = attention_scores.view(*attn_size[:-2], -1).contiguous()
        attention_probs = F.softmax(attention_scores, dim=-1)                                       # (bs, h, w, heads, n)
        attention_probs = attention_probs.view(attn_size)                                           # (bs, h, w, heads, h, w)

        if attn_size[0] != bs:
            attention_probs = attention_probs.expand(bs, *attn_size[1:])

        attention_probs = self.dropout(attention_probs)                                             # (bs, h, w, heads, h, w)
        context = torch.einsum('bijhkl,bkld->bijhd', attention_probs, hidden_state)                 # (bs, h, w, heads, model_dim)
        context = context.view(bs, h, w, -1).contiguous()                                           # (bs, h, w, heads*model_dim)
        output = self.value(context)                                                                # (bs, h, w, model_dim)

        # Residual connection
        output = output + hidden_state
        if not self.pre_norm:
            otuput = self.layer_norm(output)

        return output, attention_probs


class GaussianAttention(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.attention_gaussian_blur_trick = config.get('attention_gaussian_blur_trick', False)
        self.attention_isotropic_gaussian = config.get('attention_isotropic_gaussian', False)
        self.gaussian_mu_init_noise = config.get('gaussian_mu_init_noise', 2.0)
        self.gaussian_sigma_init_noise = config.get('gaussian_sigma_init_noise', 0.01)
        self.heads = config['num_heads']
        self.model_dim = config['model_dim']
        self.pre_norm = config['pre_norm']

        # Attention head means init
        self.attention_centers = nn.Parameter(torch.zeros(self.heads, 2).normal_(0.0, self.gaussian_mu_init_noise))

        # Attention heads covariance init
        if self.attention_isotropic_gaussian:
            attention_spreads = 1.0 + torch.zeros(self.heads).normal_(0.0, self.gaussian_sigma_init_noise)
        else:
            # Non-isotropic covariance, initialized to noisy identity matrix
            attention_spreads = torch.eye(2).unsqueeze(0).repeat(self.heads, 1, 1)
            attention_spreads += torch.zeros_like(attention_spreads).normal_(0.0, self.gaussian_sigma_init_noise)
        
        self.attention_spreads = nn.Parameter(attention_spreads)

        # Other layers
        self.value = nn.Linear(self.model_dim*self.heads, self.model_dim)
        self.layer_norm = nn.LayerNorm(self.model_dim)

        # If not using gaussian blur trick, define quadratic positional encoding
        if not self.attention_gaussian_blur_trick:
            
            MAX_WIDTH_HEIGHT = 50
            range_ = torch.arange(MAX_WIDTH_HEIGHT)
            grid = torch.cat([t.unsqueeze(-1) for t in torch.meshgrid([range_, range_])], dim=-1)       # (50, 50, 2)
            rel_idx = grid.unsqueeze(0).unsqueeze(0) - grid.unsqueeze(-2).unsqueeze(-2)                 # (50, 50, 50, 50, 2)
            R = torch.cat([
                rel_idx, rel_idx ** 2, (rel_idx[..., 0] * rel_idx[..., 1]).unsqueeze(-1)
            ], dim=-1)
            
            self.register_buffer('R', R.float())
            self.dropout = nn.Dropout(config.get('attention_dropout_prob', 0.1))


    def get_heads_target_vectors(self):

        if self.attention_isotropic_gaussian:
            a = c = self.attention_spreads ** 2
            b = torch.zeros_like(self.attention_spreads)
        else:
            inv_covariance = torch.einsum('hij,hkj->hik', self.attention_spreads, self.attention_spreads)
            a, b, c = inv_covariance[:, 0, 0], inv_covariance[:, 0, 1], inv_covariance[:, 1, 1]

        mu_1, mu_2 = self.attention_centers[:, 0], self.attention_centers[:, 1]                         # (num_heads,) each
        target = -0.5 * torch.stack([-2*(a*mu_1 + b*mu_2), -2*(c+mu_2 + b*mu_1), a, c, 2*b], dim=-1)
        
        return target


    def get_attention_probs(self, height, width):
        '''
        Computes positional attention for image of size (height, width)
        Returns output probabilities of size (h, w, num_heads, h, w)
        '''
        u = self.get_heads_target_vectors()

        attention_scores = torch.einsum('ijkld,hd->ijhkl', self.R[:height, :width, :height, :width, :], u)
        attention_probs = F.softmax(attention_scores.view(height, width, self.heads, -1), dim=-1)
        attention_probs = attention_probs.view(height, width, self.heads, height, width)

        return attention_probs


    def blurred_attention(self, X):
        ''' 
        Convolves the image features with a gaussian kernel to 
        directly obtain attention scores. To use it, set 
        attention_gaussian_blur_trick to True.
        
        Args:
            X: image feature of size (bs, h, w, model_dim)
        Output:
            attention_score: (bs, h, w, heads, h, w)
        '''
        bs, h, w, in_channels = X.size()
        Y = X.permute(0, 3, 1, 2).contiguous()                                                          # Reshape to NCHW

        kernels = []
        kernel_width = kernel_height = 7

        for mean, std_inv in zip(self.attention_centers, self.attention_spreads):
            conv_weights = gaussian_kernel_2d(mean, std_inv, size=(kernel_height, kernel_width))
            conv_weights = conv_weights.view(1, 1, kernel_height, kernel_width).repeat(in_channels, 1, 1, 1)
            kernels.append(conv_weights)

        weights = torch.cat(kernels)
        padding_height = (kernel_height - 1) // 2
        padding_width = (kernel_width - 1) // 2
        out = F.conv2d(Y, weights, groups=in_channels, padding=(padding_height, padding_width))

        # Normalization
        all_one_input = torch.ones(1, in_channels, height, width, device=X.device)
        normalizer = F.conv2d(all_one_input, weights, groups=in_channels, padding=(padding_height, padding_width))
        out /= normalizer

        return out.permute(0, 2, 3, 1).contiguous()


    def forward(self, hidden_state):

        bs, h, w, _ = hidden_state.size()
        if self.pre_norm:
            hidden_state = self.layer_norm(hidden_state)

        if not self.attention_gaussian_blur_trick:
            attention_probs = self.get_attention_probs(h, w)                                        # (h, w, heads, h, w)
            attention_probs = self.dropout(attention_probs)

            context = torch.einsum('ijhkl,bkld->bijhd', attention_probs, hidden_state)              # (bs, h, w, heads, model_dim)
            context = context.contiguous().view(bs, h, w, -1)                                       # (bs, h, w, heads*model_dim)
        else:
            context = self.blurred_attention(hidden_state)                                          # (bs, h, w, heads*model_dim)

        output = self.value(context)                                                                # (bs, h, w, model_dim)

        # Residual connection
        output = output + hidden_state
        if not self.pre_norm:
            output = self.layer_norm(output)

        return output, attention_probs

