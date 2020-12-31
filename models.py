
""" 
Networks and models.

Authors: Nishant Prabhu, Mukund Varma T
"""

import math
import torch
import numbers 
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 

MODEL_HELPER = {
    'resnet18': models.resnet18,
    'resnet50': models.resnet50,
    'resnet101': models.resnet101
}


def guassian_kernel_2d(mean, std_inv, kernel_size=(3, 3)):
    """ 
    [Helper function] Generates a 2D gaussian kernel. 

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

        # Layers  
        self.query = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
        self.key = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
        self.value = nn.Linear(self.heads*self.model_dim, self.model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(config.get('attention_dropout_prob', 0.1))

    
    def forward(self, x):
        ''' Input has size (bs, h, w, model_dim) '''

        bs, h, w, _ = x.size() 
        sqrt_normalizer = math.sqrt(self.model_dim)
        x = x.view(bs, -1, self.model_dim)                                      # (bs, n, model_dim)
        if self.pre_norm:
            x = self.layer_norm(x)

        Q = self.query(x).view(bs, self.heads, -1, self.model_dim)              # (bs, heads, n, model_dim)
        K = self.key(x).view(bs, self.heads, -1, self.model_dim)                # (bs, heads, n, model_dim)
        V = self.value(x).view(bs, self.heads, -1, self.model_dim)              # (bs, heads, n, model_dim)

        attention_scores = torch.einsum('bhid,bhjd->bhij', [Q, K])              # (bs, heads, n, n)
        attention_probs = F.softmax(attention_scores/sqrt_normalizer, dim=-1)   # (bs, heads, n, n) softmaxed
        attention_probs = self.dropout(attention_probs)

        out = torch.einsum('bhij,bhjd->bhid', [attention_probs, V])             # (bs, heads, n, model_dim)
        out = out.permute(0, 2, 1, 3).contiguous()                              # (bs, n, heads, model_dim)
        out = self.value(out.view(bs, -1, self.heads*self.model_dim))           # (bs, n, model_dim)

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
        relative_indices += max_position_embeddings - 1
        self.register_buffer('relative_indices', relative_indices)


    def compute_attention_scores(self, hidden_state):
        ''' 
        hidden_state has size (bs, h, w, model_dim)
        Output attention has size (bs, w, h, num_heads, w, h)
        '''
        bs, h, w, _ = hidden_state.size()
        sqrt_normalizer = math.sqrt(self.model_dim)

        # Compute query data if needed
        if self.query_positional_score or self.use_attention_data:
            q = self.query(hidden_state).view(bs, w, h, self.heads, self.model_dim)         # (bs, w, h, heads, model_dim)

        if self.use_attention_data:
            k = self.key(hidden_state).view(bs, w, h, self.heads, self.model_dim)           # (bs, w, h, heads, model_dim)

        # Compute row and column embeddings based on position
        rel_idx = self.relative_indices[:w, :w].reshape(-1,)
        row_embeds = self.row_embeddings(rel_idx)                                           # (w^2, position_embedding_size)

        rel_idx = self.relative_indices[:h, :h].reshape(-1,)
        col_embeds = self.col_embeddings(rel_idx)                                           # (h^2, position_embedding_size)

        # Compute attention scores
        if not self.query_positional_score:
            # No query positional data or attention data used here
            row_scores = self.head_keys_row(row_embeds).view(1, w, 1, w, self.heads)        # (1, w, 1, w, heads)
            col_scores = self.head_keys_col(col_embeds).view(h, 1, h, 1, self.heads)        # (h, 1, h, 1, heads)

            attention_scores = row_scores + col_scores                                      # (h, w, h, w, heads)
            attention_scores = attention_scores.permute(0, 1, 4, 2, 3)                      # (h, w, heads, h, w)
            attention_scores = attention_scores.unsqueeze(0)                                # (1, h, w, heads, h, w)

        else:
            # Query positional scores used, half features encode row embeddings, other half encode column embeddings
            q_row = q[:, :, :, :, :self.model_dim//2]
            q_col = q[:, :, :, :, self.model_dim//2:]
            row_scores = torch.einsum('bijhd,ikd->bijhk', q_row, row_embeddings.view(w, w, -1))     # (bs, w, h, heads, w)
            col_scores = torch.einsum('bijhd,jld->bijhl', q_col, col_embeddings.view(h, h, -1))     # (bs, w, h, heads, h)

            attention_scores = row_scores.unsqueeze(-1) + col_scores.unsqueeze(-2)          # (bs, w, h, heads, w, h)
            attention_scores = attention_scores / sqrt_normalizer           
            attention_scores = attention_scores.permute(0, 2, 1, 3, 5, 4)                   # (bs, h, w, heads, h, w)

        if self.use_attention_data:
            # Compute content based attention scores
            att_content_scores = torch.einsum('bijhd,bklhd->bijhkl', q, k)                  # (bs, w, h, heads, w, h)
            att_content_scores = att_content_scores/sqrt_normalizer
            att_content_scores = att_content_scores.permute(0, 2, 1, 3, 5, 4)               # (bs, h, w, heads, h, w)
            attention_scores = attention_scores + att_content_scores                        # (bs, h, w, heads, h, w)

        return attention_scores


    def forward(self, hidden_state):

        bs, h, w, _ = hidden_state.size()
        if self.pre_norm:
            hidden_state = self.layer_norm(hidden_state)

        attention_scores = self.compute_attention_scores(hidden_state)                      # (bs, h, w, heads, h, w)
        attn_size = attention_scores.size() 
        attention_probs = F.softmax(attention_scores.view(*attn_size[:-2], -1), dim=-1)     # (bs, h, w, heads, n)
        attention_probs = attention_probs.view(attn_size)                                   # (bs, h, w, heads, h, w)

        if attn_size[0] != bs:
            attention_probs = attention_probs.expand(bs, *attn_size[1:])

        attention_probs = self.dropout(attention_probs)                                     # (bs, h, w, heads, h, w)
        context = torch.einsum('bijhkl,bkld->bijhd', attention_probs, hidden_state)         # (bs, h, w, heads, model_dim)
        context = context.view(bs, h, w, -1).contiguous()                                   # (bs, h, w, heads*model_dim)
        output = self.value(context)                                                        # (bs, h, w, model_dim)

        # Residual connection
        output = output + hidden_state
        if not self.pre_norm:
            otuput = self.layer_norm(output)

        return output, attention_probs


class GaussianAttention(nn.Module):

    def __init__(self, config):

        