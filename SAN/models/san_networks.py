
""" 
Attention mechanisms and networks from SAN

Authors: Whoever wrote SAN
"""

import math
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from .modules import Subtraction, Subtraction2, Aggregation


def position(H, W, nchw=True):
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
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Attention(nn.Module):

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
            return x, k
        else:
            return x, None


class Bottleneck(nn.Module):

    def __init__(self, sa_type, in_planes, rel_planes, mid_planes, out_planes, share_planes=8, kernel_size=3, stride=1):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.attention = Attention(sa_type, in_planes, rel_planes, mid_planes, share_planes, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv = nn.Conv2d(mid_planes, out_planes, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride

    def forward(self, inp):
        x, k = inp
        identity = x
        out = self.relu(self.bn1(x))
        out, k = self.attention(out, k)
        out = self.relu(self.bn2(out))
        out = self.conv(out)
        out += identity
        return out, k

    
class Encoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        sa_type = config['sa_type']
        layers = config['layers']
        kernels = config['kernels']
        num_classes = config['num_classes']
        self.hier = 'hier' in config['sa_type']

        c = 64
        self.conv_in, self.bn_in = conv1x1(3, c), nn.BatchNorm2d(c)
        self.conv0, self.bn0 = conv1x1(c, c), nn.BatchNorm2d(c)
        self.layer0 = self._make_layer(sa_type, Bottleneck, c, layers[0], kernels[0])

        c *= 4
        self.conv1, self.bn1 = conv1x1(c//4, c), nn.BatchNorm2d(c)
        self.layer1 = self._make_layer(sa_type, Bottleneck, c, layers[1], kernels[1])

        c *= 2
        self.conv2, self.bn2 = conv1x1(c//2, c), nn.BatchNorm2d(c)
        self.layer2 = self._make_layer(sa_type, Bottleneck, c, layers[2], kernels[2])

        c *= 2
        self.conv3, self.bn3 = conv1x1(c//2, c), nn.BatchNorm2d(c)
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

    def _resize_key(self, x, kernel_size=2):
        ''' Input size (bs, c, h, w) '''
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()                                          # (bs, h, w, c)
        x = x.view(b, h, w//kernel_size, c*kernel_size)                                 # (bs, h, w//2, c*2)
        x = x.permute(0, 2, 1, 3).contiguous()                                          # (bs, w//2, h, c*2)
        x = x.view(b, w//kernel_size, h//kernel_size, c*kernel_size*kernel_size)        # (bs, w//2, h//2, c*4)
        return x.permute(0, 3, 2, 1).contiguous()

    def forward(self, x):
        k = None
        x = self.relu(self.bn_in(self.conv_in(x)))

        x, k = self.layer0([self.conv0(self.pool(x)), k])
        x = self.relu(self.bn0(x))
        x, k = self.layer1([self.conv1(self.pool(x)), k])
        x = self.relu(self.bn1(x))
        x, k = self.layer2([self.conv2(self.pool(x)), k])
        x = self.relu(self.bn2(x))
        x, k = self.layer3([self.conv3(self.pool(x)), k])
        x = self.relu(self.bn3(x))      
    
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x