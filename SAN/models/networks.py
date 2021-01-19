
""" 
Attention mechanisms and networks.

Authors: Nishant Prabhu, Mukund Varma T
"""


import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F


def position(H, W, nchw=True):
    '''
    Positional embedding for SAN
    '''
    if torch.cuda.is_available():
        loc_w = torch.linspace(-1, 1, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1, 1, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1, 1, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1, 1, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], dim=0).unsqueeze(0)                   # (1, 2, h, w)
    
    if not nchw:
        return loc.permute(0, 2, 3, 1).contiguous()                                                 # (1, h, w, 2)
    else:
        return loc                                                                                  # (1, 2, h, w)    
        
        
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ChannelAttention(nn.Module):
    ''' 
    Channel wise self attention. I'm not sure if this is what the authors have also done
    Refer https://arxiv.org/abs/2004.13621
    '''

    def __init__(self, config):
        
        super().__init__()
        self.model_dim = config['model_dim']
        self.heads = config['num_heads']
        self.pre_norm = config['pre_norm']

        # Layers  
        self.query = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
        self.key = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
        self.value = nn.Linear(self.model_dim, self.heads*self.model_dim, bias=False)
        self.out = nn.Linear(self.heads*self.model_dim, self.model_dim, bias=False)
        self.layer_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(config.get('attention_dropout_prob', 0.1))


    def normalize_(self, x):
        ''' Input has shape (bs, c, h, w) '''
        x = self.layer_norm(x.permute(0, 2, 3, 1).contiguous())
        return x.permute(0, 3, 1, 2).contiguous()

    
    def forward(self, x, k=None):
        ''' Input has size (bs, h, w, model_dim) '''

        if len(x.size()) != 4:
            raise ValueError(f'Input tensor should have dim == 4, got size {x.size()}')

        bs, h, w, c = x.size()                                                                      # (bs, h, w, model_dim)
        sqrt_normalizer = math.sqrt(self.model_dim)
        if self.pre_norm:
            x = self.layer_norm(x)

        q = self.query(x).view(bs, -1, self.heads, self.model_dim).permute(0, 3, 1, 2)              # (bs, model_dim, n, heads)
        k = self.key(x).view(bs, -1, self.heads, self.model_dim).permute(0, 3, 1, 2)                # (bs, model_dim, n, heads)
        v = self.value(x).view(bs, -1, self.heads, self.model_dim).permute(0, 3, 1, 2)              # (bs, model_dim, n, heads)

        attention_scores = torch.einsum('bdih,bdjh->bdij', [q, k])                                  # (bs, model_dim, n, n)
        attention_probs = self.dropout(F.softmax(attention_scores/sqrt_normalizer, dim=-1))         # (bs, model_dim, n, n)
        out = torch.einsum('bdij,bdjh->bdih', [attention_probs, v])                                 # (bs, model_dim, n, heads)
        out = out.permute(0, 2, 3, 1).contiguous().view(bs, h, w, -1)                               # (bs, h, w, model_dim * heads)
        out = self.out(out)                                                                         # (bs, h, w, model_dim)

        # Residual connection
        out = out + x
        if not self.pre_norm:
            out = self.layer_norm(out)                                                              # (bs, h, w, model_dim)

        return out, attention_probs


class Feedforward(nn.Module):

    def __init__(self, config):

        super().__init__()
        model_dim = config['model_dim']
        hidden_dim = config['ff_dim']
        self.pre_norm = config['pre_norm']
        self.relu = nn.ReLU()

        # Layers
        self.dense_1 = nn.Linear(model_dim, hidden_dim, bias=True)
        self.dense_2 = nn.Linear(hidden_dim, model_dim, bias=True)
        self.layer_norm = nn.LayerNorm(model_dim)


    def normalize_(self, x):
        ''' Input has shape (bs, h, w, c) '''
        x = self.layer_norm(x.permute(0, 2, 3, 1).contiguous())                         
        return x.permute(0, 3, 1, 2).contiguous()


    def forward(self, x):
        ''' Input has shape (bs, h, w, model_dim) '''

        if self.pre_norm:
            x = self.layer_norm(x)

        out = self.dense_2(self.relu(self.dense_1(x)))
        out = out + x

        if not self.pre_norm:
            out = self.layer_norm(out)

        return out


class EncoderBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.attention = ChannelAttention(config)
        self.feedfwd = Feedforward(config)

    def forward(self, x, k=None):
        ''' 
        In case of hierarchical attention, subsequent blocks
        receive key tensor from first block
        '''
        x, att_scores = self.attention(x)
        x = self.feedfwd(x)
        return x, att_scores


class FeaturePooling(nn.Module):
    ''' 
    Either a resnet base (until end of first BasicBlock)
    or invertible 2x2 downsampling
    '''
    
    def __init__(self, config):
        
        super().__init__()
        self.config = config 

        # Choice of downscaling method 
        if config['pool_with_resnet']:
            name = config.get('resnet', None)
            assert name in list(MODEL_HELPER.keys()), f'Invalid resnet {name}'

            base_model = MODEL_HELPER[name](pretrained=config['pretrained'])
            res_layers = list(base_model.children())[0:4+config['block']]
            self.bottom = nn.Sequential(*res_layers)

            a = torch.rand((1, 3, 32, 32))
            with torch.no_grad():
                out = self.bottom(a)
            in_features = out.size(1)

        elif self.config['pool_downsample_size'] >= 1:
            in_features = 3 * self.config['pool_downsample_size'] ** 2


    def downsample_pooling(self, x, kernel):
        ''' 
        Used if not pooling with ResNet 
        Takes in x (bs, h, w, c) and returns (bs, h//kernel, w//kernel, c*kernel*kernel)
        '''
        assert (not self.config['pool_with_resnet']) & (self.config['pool_downsample_size'] >= 1), "Something's wrong, I can feel it"
        b, h, w, c = x.size()
        y = x.contiguous().view(b, h, w//kernel, c*kernel)
        y = y.permute(0, 2, 1, 3).contiguous()
        y = y.view(b, w//kernel, h//kernel, c*kernel*kernel)
        y = y.permute(0, 2, 1, 3).contiguous()
        return y                                                    # Reference (bs, 3, 16, 16)


    def forward(self, x):
        ''' Input has size (bs, c, h, w) '''

        if self.config['pool_with_resnet']:
            features = self.bottom(x)                               # For resnet50 and CIFAR10, it has size (bs, 256, 8, 8)

        elif self.config['pool_downsample_size'] >= 1:
            x = x.permute(0, 2, 3, 1).contiguous()
            features = self.downsample_pooling(x, self.config['pool_downsample_size'])

        else:
            features = x.permute(0, 2, 3, 1).contiguous()

        return features


class SAN(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.num_blocks = config['num_encoder_blocks']
        self.model_dim = config['model_dim']

        # Layers
        self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(self.num_blocks)])
        self.positional_features = nn.Linear(2, 2, bias=True)
        self.feature_upscale = nn.Linear(14, self.model_dim, bias=True)

    def forward(self, x, return_attn=False):
        bs, h, w, c = x.size() 
        p_enc = self.positional_features(position(h, w, False))                                     # Positional encoding
        x = torch.cat([x, p_enc.repeat(x.size(0), 1, 1, 1)], dim=-1)                                # (bs, h, w, 14)
        x = self.feature_upscale(x)                                                                 # (bs, h, w, model_dim)

        attn_scores = {}                                                                            # Layerwise attention scores
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
        ''' Input has size (bs, h, w, model_dim) '''

        b, h, w, c = x.size()
        x = x.view(b, -1, c).contiguous().mean(dim=1)                                               # (bs, model_dim)
        out = self.fc(x)                                                                            # (bs, n_classes)
        return out