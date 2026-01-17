import os
import time
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch_dct

from WTConv import wtconv
from WTConv.wtconvnext import wtconvnext
from WTConv.wtconv import WTConv2d

DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class to_channels_first(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 3, 1, 2).contiguous()


class to_channels_last(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute(0, 2, 3, 1).contiguous()
    
    
def build_norm_layer(dim,
                     norm_layer,
                     in_format='channels_last',
                     out_format='channels_last',
                     eps=1e-6):
    layers = []
    if norm_layer == 'BN':
        if in_format == 'channels_last':
            layers.append(to_channels_first())
        layers.append(nn.BatchNorm2d(dim))
        if out_format == 'channels_last':
            layers.append(to_channels_last())
    elif norm_layer == 'LN':
        if in_format == 'channels_first':
            layers.append(to_channels_last())
        layers.append(nn.LayerNorm(dim, eps=eps))
        if out_format == 'channels_first':
            layers.append(to_channels_first())
    else:
        raise NotImplementedError(
            f'build_norm_layer does not support {norm_layer}')
    return nn.Sequential(*layers)


def build_act_layer(act_layer):
    if act_layer == 'ReLU':
        return nn.ReLU(inplace=True)
    elif act_layer == 'SiLU':
        return nn.SiLU(inplace=True)
    elif act_layer == 'GELU':
        return nn.GELU()

    raise NotImplementedError(f'build_act_layer does not support {act_layer}')
    
    
class StemLayer(nn.Module):
    r""" Stem layer of InternImage
    Args:
        in_chans (int): number of input channels
        out_chans (int): number of output channels
        act_layer (str): activation layer
        norm_layer (str): normalization layer
    """

    def __init__(self,
                 in_chans=3,
                 out_chans=96,
                 act_layer='GELU',
                 norm_layer='BN'):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans,
                               out_chans // 2,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm1 = build_norm_layer(out_chans // 2, norm_layer,
                                      'channels_first', 'channels_first')
        self.act = build_act_layer(act_layer)
        self.conv2 = nn.Conv2d(out_chans // 2,
                               out_chans,
                               kernel_size=3,
                               stride=2,
                               padding=1)
        self.norm2 = build_norm_layer(out_chans, norm_layer, 'channels_first',
                                      'channels_first')

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm2(x)
        return x
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        Linear = partial(nn.Conv2d, kernel_size=1, padding=0) if channels_first else nn.Linear
        self.fc1 = Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Wave2D(nn.Module):
    """
    Wave equation operator:
    d2u/dt2 - c2(d2u/dx2 + d2u/dy2) + αdu/dt = 0;
    du/dx_{x=0, x=a} = 0
    du/dy_{y=0, y=b} = 0
    =>
    A_{n, m} = C(a, b, n==0, m==0) * sum_{0}^{a}{ sum_{0}^{b}{\phi(x, y)cos(n\pi/ax)cos(m\pi/by)dxdy }}
    core = cos(n\pi/ax)cos(m\pi/by) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
    u_{x, y, t} = sum_{0}^{\infinite}{ sum_{0}^{\infinite}{ core } }
    
    assume a = N, b = M; x in [0, N], y in [0, M]; n in [0, N], m in [0, M]; with some slight change
    => 
    (\phi(x, y) = linear(dwconv(input(x, y))))
    A(n, m) = DCT2D(\phi(x, y))
    u(x, y, t) = IDCT2D(A(n, m) * (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt))
    """    
    def __init__(self, infer_mode=False, res=14, dim=96, hidden_dim=96, **kwargs):
        super().__init__()
        self.res = res
        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_linear = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.infer_mode = infer_mode
        self.to_k = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.GELU(),
        )
        self.c = nn.Parameter(torch.ones(1) * 1.0)
        self.alpha = nn.Parameter(torch.ones(1) * 0.1)
        self.save_attention = False
        
    def infer_init_wave2d(self, freq):
        weight_exp = self.get_decay_map((self.res, self.res), device=freq.device)
        self.k_exp = nn.Parameter(torch.pow(weight_exp[:, :, None], self.to_k(freq)), requires_grad=False)
        del self.to_k

    @staticmethod
    def get_cos_map(N=224, device=torch.device("cpu"), dtype=torch.float):
        # cos((x + 0.5) / N * n * \pi) which is also the form of DCT and IDCT
        # DCT: F(n) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * f(x) )
        # IDCT: f(x) = sum( (sqrt(2/N) if n > 0 else sqrt(1/N)) * cos((x + 0.5) / N * n * \pi) * F(n) )
        # returns: (Res_n, Res_x)
        weight_x = (torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(1, -1) + 0.5) / N
        weight_n = torch.linspace(0, N - 1, N, device=device, dtype=dtype).view(-1, 1)
        weight = torch.cos(weight_n * weight_x * torch.pi) * math.sqrt(2 / N)
        weight[0, :] = weight[0, :] / math.sqrt(2)
        return weight

    @staticmethod
    def get_decay_map(resolution=(224, 224), device=torch.device("cpu"), dtype=torch.float):
        # (1 - [(n\pi/a)^2 + (m\pi/b)^2]c2t2) * e^(-αt)
        # returns: (Res_h, Res_w)
        resh, resw = resolution
        weight_n = torch.linspace(0, torch.pi, resh + 1, device=device, dtype=dtype)[:resh].view(-1, 1)
        weight_m = torch.linspace(0, torch.pi, resw + 1, device=device, dtype=dtype)[:resw].view(1, -1)
        # Quadratic term for wave equation
        weight = torch.pow(weight_n, 2) + torch.pow(weight_m, 2)
        weight = torch.exp(-weight)
        return weight

    def forward(self, x: torch.Tensor, freq_embed=None, test_index=None):
        B, C, H, W = x.shape
        x = self.dwconv(x)
        x = self.linear(x.permute(0, 2, 3, 1).contiguous())
        x, z = x.chunk(chunks=2, dim=-1)

        cached_weight_cosn = getattr(self, "__WEIGHT_COSN__", None)
        if ((H, W) == getattr(self, "__RES__", (0, 0))) and cached_weight_cosn is not None and (cached_weight_cosn.device == x.device):
            weight_cosn = cached_weight_cosn
            weight_cosm = getattr(self, "__WEIGHT_COSM__", None)
            weight_exp = getattr(self, "__WEIGHT_EXP__", None)
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()
            weight_exp = self.get_decay_map((H, W), device=x.device).detach_()
            setattr(self, "__RES__", (H, W))
            setattr(self, "__WEIGHT_COSN__", weight_cosn)
            setattr(self, "__WEIGHT_COSM__", weight_cosm)
            setattr(self, "__WEIGHT_EXP__", weight_exp)

        N, M = weight_cosn.shape[0], weight_cosm.shape[0]
        weight_cosn_kernel = weight_cosn.view(H, 1, H)
        weight_cosm_kernel = weight_cosm.view(W, 1, W)
        x_perm = x.permute(0, 3, 2, 1).contiguous() # [B, C, W, H]
        x_flat_H = x_perm.view(-1, 1, H)            # [B*C*W, 1, H]
        x_u0 = F.conv1d(x_flat_H, weight_cosn_kernel).squeeze(-1) # [B*C*W, H]
        x_u0 = x_u0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous() # [B, H, W, C]
        x_perm = x_u0.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        x_flat_W = x_perm.view(-1, 1, W)               # [B*C*H, 1, W]
        x_u0 = F.conv1d(x_flat_W, weight_cosm_kernel).squeeze(-1) # [B*C*H, W]
        x_u0 = x_u0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()
        x_perm = x.permute(0, 3, 2, 1).contiguous() # [B, C, W, H]
        x_flat_H = x_perm.view(-1, 1, H)            # [B*C*W, 1, H]
        x_v0 = F.conv1d(x_flat_H, weight_cosn_kernel).squeeze(-1) # [B*C*W, H]
        x_v0 = x_v0.view(B, C, W, H).permute(0, 3, 2, 1).contiguous() # [B, H, W, C]
        x_perm = x_v0.permute(0, 3, 1, 2).contiguous() # [B, C, H, W]
        x_flat_W = x_perm.view(-1, 1, W)               # [B*C*H, 1, W]
        x_v0 = F.conv1d(x_flat_W, weight_cosm_kernel).squeeze(-1) # [B*C*H, W]
        x_v0 = x_v0.view(B, C, H, W).permute(0, 2, 3, 1).contiguous()
        if freq_embed is None:
            freq_embed = torch.zeros(B, H, W, self.hidden_dim, device=x.device, dtype=x.dtype)
        t = self.to_k(freq_embed)
        c_t = self.c * t
        cos_term = torch.cos(c_t)
        eps = 1e-8
        sin_term = torch.sin(c_t) / (self.c + eps)
        wave_term = cos_term * x_u0
        velocity_term = sin_term * (x_v0 + (self.alpha / 2) * x_u0)
        final_term = wave_term + velocity_term
        cached_weight_cosn_idct = getattr(self, "__WEIGHT_COSN_IDCT__", None)
        cached_weight_cosm_idct = getattr(self, "__WEIGHT_COSM_IDCT__", None)
        
        if ((H, W) == getattr(self, "__RES_IDCT__", (0, 0))) and \
           cached_weight_cosn_idct is not None and \
           cached_weight_cosn_idct.device == x.device:
            weight_cosn = cached_weight_cosn_idct
            weight_cosm = cached_weight_cosm_idct
        else:
            weight_cosn = self.get_cos_map(H, device=x.device).detach_()  # (H, H)
            weight_cosm = self.get_cos_map(W, device=x.device).detach_()  # (W, W)
            setattr(self, "__RES_IDCT__", (H, W))
            setattr(self, "__WEIGHT_COSN_IDCT__", weight_cosn)
            setattr(self, "__WEIGHT_COSM_IDCT__", weight_cosm)
        x_w = final_term.permute(0, 1, 3, 2).contiguous().view(B * H * C, 1, W)  # (B*H*C, 1, W)
        weight_cosm_kernel_t = weight_cosm.t().contiguous().view(W, 1, W)  # (W,1,W)
        x_w = F.conv1d(x_w, weight_cosm_kernel_t).squeeze(-1)  # (B*H*C, W)
        x_w = x_w.view(B, H, C, W).permute(0, 1, 3, 2).contiguous()  # (B,H,W,C)
        x_h = x_w.permute(0, 2, 3, 1).contiguous().view(B * W * C, 1, H)  # (B*W*C,1,H)
        weight_cosn_kernel_t = weight_cosn.t().contiguous().view(H, 1, H)  # (H,1,H)
        x_h = F.conv1d(x_h, weight_cosn_kernel_t).squeeze(-1)  # (B*W*C,H)
        x_final = x_h.view(B, W, C, H).permute(0, 3, 1, 2).contiguous()
        x = self.out_norm(x_final)
        gate = nn.functional.silu(z)
        x_gated = x * gate
        x = self.out_linear(x_gated)
        x = x.permute(0, 3, 1, 2).contiguous()
        if test_index is not None and hasattr(self, 'save_attention') and self.save_attention:
            center_h, center_w = H // 2, W // 2
            attention_map = (x_final * x_final[:, center_h:center_h+1, center_w:center_w+1, :]).sum(-1)
            
            import matplotlib.pyplot as plt

            # Ensure the save directory exists
            save_dir = "./save/attention_map"
            os.makedirs(save_dir, exist_ok=True)

            # Move attention_map to cpu and convert to numpy
            att_map_np = attention_map.detach().cpu().numpy()

            # Normalize for visualization
            att_map_norm = (att_map_np - att_map_np.min()) / (att_map_np.max() - att_map_np.min() + 1e-8)

            # Save each sample in the batch
            for i in range(att_map_norm.shape[0]):
                filename = os.path.join(save_dir, f"attention_map_{test_index}_{i}.png")
                plt.imsave(filename, att_map_norm[i], cmap='viridis')
        
        return x




class WaveBlock(nn.Module):
    def __init__(
        self,
        res: int = 14,
        infer_mode = False,
        hidden_dim: int = 0,
        drop_path: float = 0,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
        use_checkpoint: bool = False,
        drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        mlp_ratio: float = 4.0,
        post_norm = True,
        layer_scale = None,
        **kwargs,
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        self.norm1 = norm_layer(hidden_dim)
        self.op = Wave2D(res=res, dim=hidden_dim, hidden_dim=hidden_dim, infer_mode=infer_mode)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = norm_layer(hidden_dim)
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = Mlp(in_features=hidden_dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, channels_first=True)
        self.post_norm = post_norm
        self.layer_scale = layer_scale is not None
        
        self.infer_mode = infer_mode
        
        if self.layer_scale:
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(hidden_dim),
                                       requires_grad=True)

    def _forward(self, x: torch.Tensor, freq_embed, test_index=None):
        if not self.layer_scale:
            if self.post_norm:
                x = x + self.drop_path(self.norm1(self.op(x, freq_embed)))
                if self.mlp_branch:
                    x = x + self.drop_path(self.norm2(self.mlp(x))) # FFN
            else:
                x = x + self.drop_path(self.op(self.norm1(x), freq_embed))
                if self.mlp_branch:
                    x = x + self.drop_path(self.mlp(self.norm2(x))) # FFN
            return x
        if self.post_norm:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.norm1(self.op(x, freq_embed)))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.norm2(self.mlp(x))) # FFN
        else:
            x = x + self.drop_path(self.gamma1[:, None, None] * self.op(self.norm1(x), freq_embed))
            if self.mlp_branch:
                x = x + self.drop_path(self.gamma2[:, None, None] * self.mlp(self.norm2(x))) # FFN
        return x
    
    def forward(self, input: torch.Tensor, freq_embed=None):
        if self.use_checkpoint:
            return checkpoint.checkpoint(self._forward, input, freq_embed)
        else:
            return self._forward(input, freq_embed)


class AdditionalInputSequential(nn.Sequential):
    def forward(self, x, *args, **kwargs):
        for module in self[:-1]:
            if isinstance(module, nn.Module):
                x = module(x, *args, **kwargs)
            else:
                x = module(x)
        x = self[-1](x)
        return x


class WaveFormer(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.2, patch_norm=True, post_norm=True,
                 layer_scale=None, use_checkpoint=False, mlp_ratio=4.0, img_size=224,
                 act_layer='GELU', infer_mode=False, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]
        self.embed_dim = dims[0]
        self.num_features = dims[-1]
        self.dims = dims
        
        self.depths = depths
        
        self.patch_embed = StemLayer(in_chans=in_chans,
                                     out_chans=self.embed_dim,
                                     act_layer='GELU',
                                     norm_layer='LN')
        
        res0 = img_size/patch_size
        self.res = [int(res0), int(res0//2), int(res0//4), int(res0//8)]
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        
        self.infer_mode = infer_mode
        
        self.freq_embed = nn.ParameterList()
        for i in range(self.num_layers):
            freq_param = nn.Parameter(torch.zeros(self.res[i], self.res[i], self.dims[i]), requires_grad=True)
            trunc_normal_(freq_param, std=.015)
            self.freq_embed.append(freq_param)
        
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.layers.append(self.make_layer(
                res = self.res[i_layer],
                dim = self.dims[i_layer],
                depth = depths[i_layer],
                drop_path = dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                use_checkpoint=use_checkpoint,
                norm_layer=LayerNorm2d,
                post_norm=post_norm,
                layer_scale=layer_scale,
                downsample=self.make_downsample(
                    self.dims[i_layer], 
                    self.dims[i_layer + 1], 
                    norm_layer=LayerNorm2d,
                ) if (i_layer < self.num_layers - 1) else nn.Identity(),
                mlp_ratio=mlp_ratio,
                infer_mode=infer_mode,
            ))
            
        self.classifier = nn.Sequential(
            LayerNorm2d(self.num_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(self.num_features, num_classes),
        )

        self.apply(self._init_weights)

    @staticmethod
    def make_downsample(dim=96, out_dim=192, norm_layer=LayerNorm2d):
        return nn.Sequential(
            #norm_layer(dim),
            #nn.Conv2d(dim, out_dim, kernel_size=2, stride=2)
            nn.Conv2d(dim, out_dim, kernel_size=3, stride=2, padding=1, bias=False),
            norm_layer(out_dim)
        )

    @staticmethod
    def make_layer(
        res=14,
        dim=96, 
        depth=2,
        drop_path=[0.1, 0.1], 
        use_checkpoint=False, 
        norm_layer=LayerNorm2d,
        post_norm=True,
        layer_scale=None,
        downsample=nn.Identity(), 
        mlp_ratio=4.0,
        infer_mode=False,
        **kwargs,
    ):
        assert depth == len(drop_path)
        blocks = []
        for d in range(depth):
            blocks.append(WaveBlock(
                res=res,
                hidden_dim=dim, 
                drop_path=drop_path[d],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                post_norm=post_norm,
                layer_scale=layer_scale,
                infer_mode=infer_mode,
            ))
        
        return AdditionalInputSequential(
            *blocks, 
            downsample,
        )
 
    def _init_weights(self, m: nn.Module):
        """
        Initialize weights for Linear, Conv2d, and LayerNorm layers
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.Conv2d, nn.Conv1d)):
            # Initialize Conv2d and Conv1d layers using kaiming initialization
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Parameter):
            # Initialize parameters with small values
            if m.data.numel() > 1:
                trunc_normal_(m.data, std=.02)
            else:
                nn.init.constant_(m.data, 0)
    
    def infer_init(self):
        for i, layer in enumerate(self.layers):
            for block in layer[:-1]:
                block.op.infer_init_wave2d(self.freq_embed[i])
        del self.freq_embed
    
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.infer_mode:
            for layer in self.layers:
                x = layer(x)
        else:
            for i, layer in enumerate(self.layers):
                x = layer(x, self.freq_embed[i]) # (B, C, H, W)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    pass



