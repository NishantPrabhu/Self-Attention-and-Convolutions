
""" 
Architectures and networks.

Authors: Nishant Prabhu, Mukund Varma T
"""

import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F


def gelu(x):
    ''' GeLU activation function for feedforward layer '''
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model_dim = config['model_dim']
        self.heads = config['num_heads']
        self.pre_norm = config.get('pre_norm', True)

        # Layers 
        self.query = nn.Linear(self.model_dim, self.model_dim*self.heads, bias=False)
        self.key = nn.Linear(self.model_dim, self.model_dim*self.heads, bias=False)
        self.value = nn.Linear(self.model_dim*self.heads, self.model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(config['attention_dropout_prob'])

    def forward(self, x):
        ''' Input will have size (bs, num_patches, model_dim) '''
        if len(x.size()) != 3:
            raise ValueError(f'Expected 3d tensor as input, got size {x.size()}')

        bs, n, _ = x.size()                                                                             # (bs, n, model_dim)
        sqrt_normalizer = math.sqrt(self.model_dim)
        if self.pre_norm:
            x = self.layer_norm(x)          

        q = self.query(x).view(bs, n, self.heads, self.model_dim).permute(0, 2, 1, 3)                   # (bs, heads, n, model_dim)
        k = self.key(x).view(bs, n, self.heads, self.model_dim).permute(0, 2, 1, 3)                     # (bs, heads, n, model_dim)

        attention_score = torch.einsum('bhid,bhjd->bhij', [q, k]) / sqrt_normalizer                     # (bs, heads, n, n)
        attention_probs = self.dropout(F.softmax(attention_score, dim=-1))                              # (bs, heads, n, n)
        context = torch.einsum('bhij,bjd->bhid', [attention_probs, x])                                  # (bs, heads, n, model_dim)
        context = context.permute(0, 2, 1, 3).contiguous().view(bs, n, -1)                              # (bs, n, heads * model_dim)
        
        out = self.value(context) + x                                                                   # (bs, n, model_dim)
        if not self.pre_norm:
            out = self.layer_norm(out)

        return out, attention_probs


class Feedforward(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.model_dim = config['model_dim']
        self.ff_dim = config['ff_dim']
        self.pre_norm = config.get('pre_norm', True)

        # Layers
        self.fc_1 = nn.Linear(self.model_dim, self.ff_dim)
        self.fc_2 = nn.Linear(self.ff_dim, self.model_dim)
        self.act = gelu
        self.layer_norm = nn.LayerNorm(self.model_dim)

    def forward(self, x):
        if self.pre_norm:
            x = self.layer_norm(x)
        out = self.fc_2(self.act(self.fc_1(x))) + x
        if not self.pre_norm:
            out = self.layer_norm(out)

        return out


class PatchExtraction(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.patch_size = config['patch_size']
        self.h, self.w = config['image_size']
        self.model_dim = config['model_dim']
        self.num_patches = int((self.h * self.w) / pow(self.patch_size, 2))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Layers
        self.unfold = nn.Unfold(kernel_size=self.patch_size, stride=self.patch_size)
        self.feature_upscale = nn.Linear(3 * pow(self.patch_size, 2), self.model_dim)
        self.position_embedding = nn.Embedding(self.num_patches+1, self.model_dim)
    
    def forward(self, x):
        ''' 
        Input:                      (bs, channels, h, w)
        Patch collection size:      (bs, k^2 * c, num_patches+1)
        Positional encoding:        (bs, k^2 * c, num_patches+1)
        '''
        patches = self.unfold(x)                                                                    # (bs, k^2 * c, n)
        bs, d, _ = patches.size()
        cls_patch_prepend = torch.Tensor(bs, d, 1).zero_().to(self.device)
        patches = torch.cat([cls_patch_prepend, patches], dim=-1)                                   # (bs, k^2 * c, n+1)
        patches = self.feature_upscale(patches.permute(0, 2, 1).contiguous())                       # (bs, n+1, model_dim)
        pos_embeds = self.position_embedding(torch.arange(self.num_patches+1).to(self.device))      # (bs, n+1, model_dim)
        out = patches + pos_embeds                                                                  # (bs, n+1, model_dim)
        return out


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = SelfAttention(config)
        self.feedfwd = Feedforward(config)

    def forward(self, x):
        out, attn_probs = self.attention(x)
        out = self.feedfwd(out)
        return out, attn_probs


class Encoder(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.num_blocks = config['num_encoder_blocks']
        self.heads = config['num_heads']
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(self.num_blocks)])

    def forward(self, x, return_attn=False):
        attn_scores = {}                                            # Layerwise attention scores collector
        for i in range(self.num_blocks):
            x, att = self.blocks[i](x)
            attn_scores[f'layer_{i}'] = att

        if return_attn:
            return x, attn_scores
        else:
            return x


class ClassificationHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.fc = nn.Linear(config['model_dim'], config['num_classes'])

    def forward(self, x):
        ''' Input has size (batch_size, n, model_dim) '''
        return self.fc(x.mean(dim=1))