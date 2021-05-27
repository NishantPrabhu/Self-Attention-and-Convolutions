
""" 
Attention mechanisms and networks.

Acknowledgement:
    Most of the code here has been adapted with very few changes from 
    https://github.com/hszhao/SAN/blob/master/model/san.py
"""

import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .modules import Subtraction, Subtraction2, Aggregation


def position(H, W, nchw=True):
    '''
    Generates positional encoding for input tensor
    Two planes, one with row positions equispaced between (-1, 1)
    other with column positions equispaced between (-1, 1)
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
        return loc  

def conv1x1(in_planes, out_planes, stride=1):
    '''
    Convolution with 1x1 kernel and no bias
    '''
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Attention(nn.Module):
    ''' 
    Self attention layer for pairwise and patchwise attention.
    '''

    def __init__(self, sa_type, in_planes, rel_planes, out_planes, share_planes, kernel_size=3, stride=1, dilation=1):
        super().__init__()
        self.sa_type = sa_type
        self.stride = stride
        self.kernel_size = kernel_size
        self.in_planes = in_planes
        self.rel_planes = rel_planes
        self.hierarchical = 'hier' in self.sa_type

        # Layers
        self.conv1 = nn.Conv2d(in_planes, rel_planes, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_planes, rel_planes, kernel_size=1, bias=False)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)
        self.aggregation = Aggregation(kernel_size, stride, (dilation * (kernel_size-1) + 1)//2, dilation, pad_mode=1)

        if 'pair' in self.sa_type:
            self.conv_w = nn.Sequential(
                nn.BatchNorm2d(rel_planes+2), nn.ReLU(inplace=True),
                nn.Conv2d(rel_planes+2, rel_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(rel_planes), nn.ReLU(inplace=True),
                nn.Conv2d(rel_planes, out_planes//share_planes, kernel_size=1))
            self.conv_p = nn.Conv2d(2, 2, kernel_size=1)
            self.subtraction = Subtraction(kernel_size, stride, (dilation*(kernel_size-1)+1)//2, dilation, pad_mode=1)
            self.subtraction2 = Subtraction2(kernel_size, stride, (dilation*(kernel_size-1)+1)//2, dilation, pad_mode=1)
            self.softmax = nn.Softmax(dim=-2)
        
        elif 'patch' in self.sa_type:
            self.conv_w = nn.Sequential(
                nn.BatchNorm2d(rel_planes * (pow(kernel_size, 2) + 1)), nn.ReLU(inplace=True),
                nn.Conv2d(rel_planes * (pow(kernel_size, 2) + 1), out_planes//share_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes//share_planes), nn.ReLU(inplace=True),
                nn.Conv2d(out_planes//share_planes, pow(kernel_size, 2) * out_planes // share_planes, kernel_size=1))
            self.unfold_i = nn.Unfold(kernel_size=1, dilation=dilation, padding=0, stride=stride)
            self.unfold_j = nn.Unfold(kernel_size=kernel_size, dilation=dilation, padding=0, stride=stride)
            self.pad = nn.ReflectionPad2d(kernel_size//2)
        
        else:
            raise ValueError(f'Unrecognized attention type {self.sa_type}')
        
    def forward(self, x, k=None):
        q = self.conv1(x) 
        v = self.conv3(x)
        if k is None:
            k = self.conv2(x)

        if 'pair' in self.sa_type:
            p = self.conv_p(position(x.size(2), x.size(3), nchw=True))
            w = self.softmax(self.conv_w(torch.cat([self.subtraction2(q, k), self.subtraction(p).repeat(x.size(0), 1, 1, 1)], 1)))
        
        elif 'patch' in self.sa_type:
            if self.stride != 1: 
                q = self.unfold_i(q)
            q = q.view(x.size(0), -1, 1, x.size(2)*x.size(3))
            k_ = self.unfold_j(self.pad(k)).view(x.size(0), -1, 1, q.size(-1))
            w = self.conv_w(torch.cat([q, k_], dim=1)).view(x.size(0), -1, pow(self.kernel_size, 2), q.size(-1))
        
        x = self.aggregation(v, w)
        
        if self.hierarchical:
            return x, w, k
        else:
            return x, w, None


class Bottleneck(nn.Module):
    ''' 
    Building block for the encoder network
    '''

    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=3, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.attention = Attention(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, inp):
        x, _, k = inp
        identity = x
        out = self.relu(self.bn1(x))
        out, w, k = self.attention(x, k)
        out = self.relu(self.bn2(out))
        out = self.conv(out)
        out += identity
        return out, w, k

    
class Encoder(nn.Module):
    '''
    Module containing the attention layers. The code written here
    is to be used with hierarchical self-attention only.
    '''

    def __init__(self, config):
        super().__init__()
        sa_type = config['sa_type']
        layers = config['layers']
        kernels = config['kernels']
        num_classes = config['num_classes']
        self.hier = 'hier' in config['sa_type']
        self.layers = layers

        c = 256
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.enc_layer = self._make_layer(sa_type, Bottleneck, c, layers[0], kernels[0])
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.conv1, self.bn1 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.conv2, self.bn2 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.conv3, self.bn3 = conv1x1(c, c), nn.BatchNorm2d(c)

        self.layer0 = self._make_layer(sa_type, Bottleneck, c, layers[0], kernels[0])
        self.layer1 = self._make_layer(sa_type, Bottleneck, c, layers[1], kernels[1])
        self.layer2 = self._make_layer(sa_type, Bottleneck, c, layers[2], kernels[2])
        self.layer3 = self._make_layer(sa_type, Bottleneck, c, layers[3], kernels[3])

        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(c, num_classes)


    def _make_layer(self, sa_type, block, planes, blocks, kernel_size=3, stride=1):
        layers = []
        for _ in range(blocks):
            layers.append(block(sa_type, planes, planes//16, planes//4, planes, 8, kernel_size, stride))
        return nn.Sequential(*layers)


    def forward(self, x, return_attn=False):
        k = None
        x = self.relu(self.bn_in(self.conv_in(x)))

        if self.hier:
            x = self.conv0(x)
            x, w1, k = self.enc_layer([x, None, k])
            x = self.relu(self.bn0(x))
            
            x = self.conv1(x)
            x, w2, k = self.enc_layer([x, None, k])
            x = self.relu(self.bn1(x))

            x = self.conv2(x)
            x, w3, k = self.enc_layer([x, None, k])
            x = self.relu(self.bn2(x))

            x = self.conv3(x)
            x, w4, k = self.enc_layer([x, None, k])
            x = self.relu(self.bn3(x))

            attn_scores = {'pass_1': w1, 'pass_2': w2, 'pass_3': w3, 'pass_4': w4}
        
        else:
            # This is different from the original implementation.
            # We did NOT use this; we used the original code from 
            # the official repository with small changes

            x, _, k = self.layer0([self.conv0(self.pool(x)), None, k])
            x = self.relu(self.bn0(x))
            x, _, k = self.layer1([self.conv1(self.pool(x)), None, k])
            x = self.relu(self.bn1(x))
            x, _, k = self.layer2([self.conv2(self.pool(x)), None, k])
            x = self.relu(self.bn2(x))
            x, _, k = self.layer3([self.conv3(self.pool(x)), None, k])
            x = self.relu(self.bn3(x))

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        if return_attn:
            return x, attn_scores
        else:
            return x