from mmdet.models.backbones.swin import BaseModule, MODELS
from waveformer.waveformer import WaveFormer
from torch import nn
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import checkpoint
from functools import partial


# ---------------------------------------------------------------------------
# monkey patch: mmengine 分布式评估 collect_results_cpu 在 16 卡 NCCL 下
# broadcast tmpdir 路径时偶发产生 null byte，导致 open() 报 "embedded null byte"。
# 用 split(b'\x00')[0] 截断 null byte 及之后内容，兼容正常/异常两种情况。
# ---------------------------------------------------------------------------
import mmengine
import tempfile
import pickle
import shutil
import os.path as osp
from itertools import zip_longest, chain
import mmengine.dist.dist as _dist


def _collect_results_cpu_patched(result_part, size, tmpdir=None):
    from mmengine.dist import get_dist_info, broadcast, barrier
    rank, world_size = get_dist_info()
    if world_size == 1:
        return result_part[:size]
    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8)
        if rank == 0:
            mmengine.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8)
            dir_tensor[:len(tmpdir)] = tmpdir
        broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.numpy().tobytes().split(b'\x00')[0].decode().rstrip()
    else:
        mmengine.mkdir_or_exist(tmpdir)
    with open(osp.join(tmpdir, f'part_{rank}.pkl'), 'wb') as f:
        pickle.dump(result_part, f, protocol=2)
    barrier()
    if rank != 0:
        return None
    part_list = []
    for i in range(world_size):
        path = osp.join(tmpdir, f'part_{i}.pkl')
        if not osp.exists(path):
            raise FileNotFoundError(
                f'{tmpdir} is not a shared directory for rank {i}')
        with open(path, 'rb') as f:
            part_list.append(pickle.load(f))
    zipped_results = zip_longest(*part_list)
    ordered_results = [
        i for i in chain.from_iterable(zipped_results) if i is not None
    ]
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


_dist.collect_results_cpu = _collect_results_cpu_patched


@MODELS.register_module()
class MMDET_WAVEFORMER(BaseModule, WaveFormer):
    def __init__(self, patch_size=4, in_chans=3, num_classes=1000, depths=[2, 2, 9, 2], 
                 dims=[96, 192, 384, 768], drop_path_rate=0.1,
                 patch_norm=True, post_norm=True, layer_scale=None,
                 use_checkpoint=False, out_indices=(0, 1, 2, 3), pretrained=None, img_size=512, 
                 num_heads=1, **kwargs,
        ):
        BaseModule.__init__(self)
        WaveFormer.__init__(self, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, depths=depths, 
                 dims=dims, drop_path_rate=drop_path_rate,
                 patch_norm=patch_norm, post_norm=post_norm, layer_scale=layer_scale, img_size=img_size, 
                 use_checkpoint=use_checkpoint, num_heads=num_heads, **kwargs)
        
        # add norm ===========================
        self.out_indices = out_indices
        for i in out_indices:
            layer = nn.LayerNorm(self.dims[i])
            layer_name = f'outnorm{i}'
            self.add_module(layer_name, layer)
        
        # modify layer ========================
        def layer_forward(self: nn.Sequential, x, *args, **kwargs):
            for blk in self[:-1]:
                if isinstance(blk, nn.Module):
                    if blk.use_checkpoint:
                        x = checkpoint.checkpoint(blk, x, *args, **kwargs)
                    else:
                        x = blk(x, *args, **kwargs)
                else:
                    if blk.use_checkpoint:
                        x = checkpoint.checkpoint(blk, x)
                    else:
                        x = blk(x)
            # y = None
            # if self.downsample is not None:
            y = self[-1](x)

            return x, y

        for l in self.layers:
            l.forward = partial(layer_forward, l)

        # delete head ===-======================
#         del self.head
#         del self.avgpool
#         del self.norm
        del self.classifier

        # load pretrained ======================
        if pretrained is not None:
            assert os.path.exists(pretrained)
            self.load_pretrained(pretrained)

    def load_pretrained(self, ckpt=""):
        _ckpt = torch.load(open(ckpt, "rb"), map_location=torch.device("cpu"), weights_only=False)
        print(f"Successfully load ckpt {ckpt}")
        state_dict = _ckpt['model']
        # 分类在 img_size=224 训练，alpha_embed 分辨率与检测的 512 不同。
        # 按归一化频率网格做 bicubic 插值对齐（与 forward 里的多尺度逻辑一致）。
        for i in range(len(self.alpha_embed)):
            key = f'alpha_embed.{i}'
            if key in state_dict and state_dict[key].shape != self.alpha_embed[i].shape:
                src = state_dict[key]  # (res_pt, res_pt//2+1, K)
                tg_h, tg_w = self.alpha_embed[i].shape[0], self.alpha_embed[i].shape[1]
                tmp = src.permute(2, 0, 1).unsqueeze(0).float()
                tmp = F.interpolate(tmp, size=(tg_h, tg_w), mode='bicubic').squeeze(0).permute(1, 2, 0)
                state_dict[key] = tmp.to(src.dtype)
        incompatibleKeys = self.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        y = x
        for i, layer in enumerate(self.layers):
        
            alpha = self.alpha_embed[i] # (res, res//2+1, K)
            if alpha.shape[:2] != (y.shape[2], y.shape[3] // 2 + 1):
                tmp = alpha.permute(2, 0, 1).contiguous().unsqueeze(0)
                tmp = F.interpolate(tmp, size=(y.shape[2], y.shape[3] // 2 + 1), mode='bicubic').squeeze(0).permute(1, 2, 0).contiguous()
                alpha = tmp
            x, y = layer(y, alpha) # (B, C, H, W)
            if i in self.out_indices:
                norm_layer: nn.LayerNorm = getattr(self, f'outnorm{i}')
                out = norm_layer(x.permute(0, 2, 3, 1))
                out = out.permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return outs
