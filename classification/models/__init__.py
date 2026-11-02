import os
import torch
import torch.nn.functional as F

from functools import partial

from .waveformer import WaveFormer


def build_waveformer_model(config, is_pretrain=False):
    model_type = config.MODEL.TYPE
    
    if model_type in ["WaveFormer"]:
        model = WaveFormer(
            in_chans=config.MODEL.WAVE.IN_CHANS, 
            patch_size=config.MODEL.WAVE.PATCH_SIZE, 
            num_classes=config.MODEL.NUM_CLASSES, 
            depths=config.MODEL.WAVE.DEPTHS, 
            dims=config.MODEL.WAVE.EMBED_DIM, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            mlp_ratio=config.MODEL.WAVE.MLP_RATIO,
            post_norm=config.MODEL.WAVE.POST_NORM,
            layer_scale=config.MODEL.WAVE.LAYER_SCALE,
            img_size=config.DATA.IMG_SIZE,
            infer_mode=config.EVAL_MODE or config.THROUGHPUT_MODE,
            num_heads=config.MODEL.WAVE.NUM_HEADS,
        )
        if config.THROUGHPUT_MODE:
            model.infer_init()
        return model
    
    
def build_model(config, is_pretrain=False):
    model = build_waveformer_model(config, is_pretrain)
    return model
