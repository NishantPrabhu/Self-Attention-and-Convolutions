
"""
Networks and model classes.

Authors: Nishant Prabhu, Mukund Varma T
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models 
from . import attention


MODEL_HELPER = {
    'resnet18': {'net': models.resnet18, 'out': 512},
    'resnet50': {'net': models.resnet50, 'out': 2048},
    'resnet101': {'net': models.resnet101, 'out': 2048}
}


# ===================================================================================================================================
#  Activation functions for feedforward layer
# 
#     - GeLU: Gaussian error linear units 
#     - Swish: Swish function
# ===================================================================================================================================


def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

# Activation function map
ACT_FN = {'gelu': gelu, 'relu': F.relu, 'swish': swish}


# ===================================================================================================================================
#  Networks
# 
#     - Feedforward: Feedforward module of BERT encoder unit 
#     - EncoderBlock: Single BERT encoder unit consisting of MHA + Feedforward
#     - BertEncoder: Stack of EncoderBlocks based on provided configuration
#     - FeaturePooling: Extracts low level features from input image. Either a resnet bottom or 2x2 invertible downsampling
#     - ClassificationHead: Takes in BERT encoder's features, average-pools them and outputs class probabilities
# ===================================================================================================================================


class Feedforward(nn.Module):

    def __init__(self, config):

        super().__init__()
        model_dim = config['model_dim']
        hidden_dim = config['ff_dim']
        act_fn = config['ff_activation']
        self.pre_norm = config['pre_norm']

        # Layers
        self.fc1 = nn.Linear(model_dim, hidden_dim)
        self.act = ACT_FN[act_fn]
        self.fc2 = nn.Linear(hidden_dim, model_dim)
        self.layer_norm = nn.LayerNorm(model_dim)


    def forward(self, x):
        ''' Input has shape (bs, ..., model_dim) '''

        if self.pre_norm:
            x = self.layer_norm(x)

        out = self.fc2(self.act(self.fc1(x)))
        out = out + x

        if not self.pre_norm:
            out = self.layer_norm(out)

        return out


class EncoderBlock(nn.Module):

    def __init__(self, config):

        super().__init__()
        attn_name = config.get('attention', 'bert')
        
        if attn_name == 'bert':
            self.attention = attention.BertSelfAttention(config)
        elif attn_name == 'learned_2d':
            self.attention = attention.Learned2dRelativeSelfAttention(config)
        elif attn_name == 'gaussian':
            self.attention = attention.GaussianAttention(config)
        else:
            raise ValueError(f'Invalid attention type {attn_name}')

        self.feedfwd = Feedforward(config)


    def forward(self, x, k=None):
        ''' 
        In case of hierarchical attention, subsequent blocks
        receive key tensor from first block
        '''
        x, att_scores, k = self.attention(x, k)
        x = self.feedfwd(x)
        return x, att_scores, k


class BertEncoder(nn.Module):

    def __init__(self, config):

        super().__init__()
        self.num_blocks = config['num_encoder_blocks']
        self.hierarchical = config['hierarchical_weight_sharing']
        if not self.hierarchical:
            self.blocks = nn.ModuleList([EncoderBlock(config) for _ in range(self.num_blocks)])
        else:
            self.block = EncoderBlock(config)


    def forward(self, x, return_attn=False):

        attn_scores = {}                                            # Layerwise attention scores collector
        k = None                                                    # Initialize to None, it will change depending on hierarchical

        if not self.hierarchical: 
            for i in range(self.num_blocks):
                x, att, k = self.blocks[i](x, k)
                attn_scores[f'layer_{i}'] = att
        else:
            for i in range(self.num_blocks):
                x, att, k = self.block(x, k)
                attn_scores[f'layer_{i}'] = att

        if return_attn:
            return x, attn_scores
        else:
            return x


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

        # Feature upscaling
        self.feature_upscale = nn.Linear(in_features, config['model_dim'])


    def downsample_pooling(self, x, kernel):
        ''' 
        Used if not pooling with ResNet 
        Takes in x (bs, h, w, c) and returns (bs, h//kernel, w//kernel, c*kernel*kernel)
        '''

        assert (not self.config['pool_with_resnet']) & (self.config['pool_downsample_size'] > 1), "Something's wrong, I can feel it"
        b, h, w, c = x.size()
        y = x.contiguous().view(b, h, w//kernel, c*kernel)
        y = y.permute(0, 2, 1, 3).contiguous()
        y = y.view(b, w//kernel, h//kernel, c*kernel*kernel)
        y = y.permute(0, 2, 1, 3).contiguous()
        
        return y


    def forward(self, x):
        ''' Input has size (bs, c, h, w) '''

        if self.config['pool_with_resnet']:
            features = self.bottom(x)                               # For resnet50 and CIFAR10, it has size (bs, 256, 8, 8)
            features = features.permute(0, 2, 3, 1).contiguous()    # Convert to NHWC

        elif self.config['pool_downsample_size'] > 1:
            x = x.permute(0, 2, 3, 1).contiguous()
            features = self.downsample_pooling(x, self.config['pool_downsample_size'])

        else:
            features = x.permute(0, 2, 3, 1).contiguous()

        return self.feature_upscale(features)


class ClassificationHead(nn.Module):
    ''' 
    Transforms encoder features into class probabilities
    '''

    def __init__(self, config):

        super().__init__()
        self.fc = nn.Linear(config['model_dim'], config['num_classes'])


    def forward(self, x):
        ''' Input has size (bs, h, w, c) '''

        bs, c = x.size(0), x.size(-1)
        x = x.view(bs, -1, c).contiguous().mean(dim=1)
        out = self.fc(x)
        return F.log_softmax(out, dim=-1)


class ResnetClassifier(nn.Module):
    ''' 
    Normal resnet for classification
    '''

    def __init__(self, config):
        super().__init__()
        name = config.get('name', None)
        assert name in list(MODEL_HELPER.keys()), f'name should be one of {list(MODEL_HELPER.keys())}'

        # Initial layers adjusted for CIFAR10
        conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        bn1 = nn.BatchNorm2d(64)
        relu1 = nn.ReLU()

        # Extract layers from resnet18 and add own
        resnet = MODEL_HELPER[name]['net'](pretrained=False)
        out_channels = MODEL_HELPER[name]['out']
        layers = list(resnet.children())
        self.backbone = nn.Sequential(conv0, bn1, relu1, *layers[4:len(layers)-1])
        self.flatten = nn.Flatten()
        self.fc_out = nn.Linear(out_channels, 10, bias=True)

        # Weight initialization
        for m in self.backbone.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Reference: https://arxiv.org/abs/1706.02677
        if config['zero_init_residual']:
            for m in self.backbone.modules():
                if isinstance(m, models.resnet.Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, models.resnet.BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)


    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        return F.log_softmax(self.fc_out(x), dim=-1)
