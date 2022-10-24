"""
Refer to the VOLO implementations https://github.com/sail-sg/volo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import numpy as np

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class OutlookAttention(nn.Module):
    """
    Implementation of outlook attention
    --dim: hidden dim
    --num_heads: number of heads
    --kernel_size: kernel size in each window for outlook attention
    return: token features after outlook attention
    """

    #if bert-base: kernel_size is (3,768), stride is (2,768)
    #if bert-large: kernel_size is (3,1024), stride is (2,1024)
    #if global2fine wich conv, the kernel_size will changed based on the number of convolution filters
    #to (*,300) or (*,384) the 384 is 128*3
    def __init__(self, dim, num_heads, kernel_size=(3,300), padding=1, stride=(2,300),
                 qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        head_dim = dim // num_heads
        self.emb_dim = 300 
        self.num_heads = num_heads
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.scale = qk_scale or head_dim**-0.5

        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        #self.attn = nn.Linear(dim, kernel_size**4 * num_heads)     byzfy
        self.attn = nn.Linear(dim, kernel_size[0]*kernel_size[1]*kernel_size[0]* num_heads)


        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.unfold = nn.Unfold(kernel_size=kernel_size, padding=1, stride=stride)
        #self.pool = nn.AvgPool2d(kernel_size=stride, stride=stride, ceil_mode=True)
        #F.max_pool1d(conved_2, conved_2.shape[2]).squeeze(2)
        self.pool = nn.AvgPool2d(kernel_size=stride, padding=0, stride=stride, ceil_mode=True)

    def forward(self, x):
        B, H, W, C = x.shape

        v = self.v(x).permute(0, 3, 1, 2)  # B, C, H, W
        h, w = math.ceil(H / self.stride[0]), math.ceil(W / self.stride[1])
        v = self.unfold(v)

        #size of v: (B,1728,196):
        #1728=C*kernal_size*kernal_size
        #196=((H+2*padding-(kernel_size-1))/stride)*(W+2*padding-(kernel_size-1))/stride)
        '''
        v = v.reshape(B, self.num_heads, C // self.num_heads,
                                   self.kernel_size * self.kernel_size,
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
        '''
        v = v.reshape(B, self.num_heads, C // self.num_heads,
                                  self.kernel_size[0]*self.kernel_size[1],
                                   h * w).permute(0, 1, 4, 3, 2)  # B,H,N,kxk,C/H
        attn = self.pool(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        #attn = self.attn(attn).reshape(
        #    B, h * w, self.num_heads, self.kernel_size * self.kernel_size,
        #    self.kernel_size * self.kernel_size).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = self.attn(attn)
        attn = attn.reshape(
            B, h * w, self.num_heads, self.kernel_size[0],
            self.kernel_size[0]*self.kernel_size[1]).permute(0, 2, 1, 3, 4)  # B,H,N,kxk,kxk
        attn = attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        #attn@v:(1,6,196,9,32)
        #x = (attn @ v).permute(0, 1, 4, 3, 2).reshape(
        #    B, C * self.kernel_size * self.kernel_size, h * w)
        y=attn@v
        x = (attn @ v).permute(0, 1, 4, 3, 2).repeat(1,1,self.emb_dim,1,1)
        x = x.reshape(
            B, C * self.kernel_size[0]*self.kernel_size[1], h * w)
        x = F.fold(x, output_size=(H, W), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride)
        x = self.proj(x.permute(0, 2, 3, 1))
        x = self.proj_drop(x)

        return x


class Outlooker(nn.Module):
    """
    Implementation of outlooker layer: which includes outlook attention + MLP
    Outlooker is the first stage in our VOLO
    --dim: hidden dim
    --num_heads: number of heads
    --mlp_ratio: mlp ratio
    --kernel_size: kernel size in each window for outlook attention
    return: outlooker layer
    """
    def __init__(self, dim, kernel_size=(3,300), padding=1, stride=(2,300),
            num_heads=1, fine_ratio=1., mlp_ratio=3., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm, qkv_bias=False,
                 qk_scale=None):
        super().__init__()
        self.fine_ratio = fine_ratio
        self.norm1 = norm_layer(dim)
        self.attn = OutlookAttention(dim, num_heads, kernel_size=kernel_size,
                                     padding=padding, stride=stride,
                                     qkv_bias=qkv_bias, qk_scale=qk_scale,
                                     attn_drop=attn_drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer)

    def forward(self, x):
        x = x + self.fine_ratio*self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
