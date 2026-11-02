_base_ = [
    '../_base_/models/mask-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py',
]

# =============================================================================
# Mask R-CNN + FPN with a WaveFormer-B backbone, COCO, 1x multi-scale training.
# =============================================================================
model = dict(
    type='MaskRCNN',
    backbone=dict(
        _delete_=True,
        type='MMDET_WAVEFORMER',
        depths=(2, 2, 18, 2),
        dims=128,
        num_heads=1,
        post_norm=True,
        layer_scale=1.e-5,
        drop_path_rate=0.3,
        img_size=512,
        out_indices=(0, 1, 2, 3),
        # ImageNet-1K classification checkpoint; run_detection.sh sets it for training.
        # Leave it None for testing a trained detector.
        pretrained=None,
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
)

# Multi-scale + random crop augmentation (large jitter, from DETR/Sparse RCNN).
# Test pipeline stays single-scale (1333, 800).
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            dict(
                type='RandomChoiceResize',
                scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                        (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                        (736, 1333), (768, 1333), (800, 1333)],
                keep_ratio=True)
        ],
                    [
                        dict(
                            type='RandomChoiceResize',
                            scales=[(400, 1333), (500, 1333), (600, 1333)],
                            keep_ratio=True),
                        dict(
                            type='RandomCrop',
                            crop_type='absolute_range',
                            crop_size=(384, 600),
                            allow_negative_crop=True),
                        dict(
                            type='RandomChoiceResize',
                            scales=[(480, 1333), (512, 1333), (544, 1333),
                                    (576, 1333), (608, 1333), (640, 1333),
                                    (672, 1333), (704, 1333), (736, 1333),
                                    (768, 1333), (800, 1333)],
                            keep_ratio=True)
                    ]]),
    dict(type='PackDetInputs')
]
train_dataloader = dict(batch_size=2, dataset=dict(pipeline=train_pipeline))
val_dataloader = dict(batch_size=2)

# 1x schedule: 12 epochs, lr decays at epochs 10 and 11.
max_epochs = 12
train_cfg = dict(max_epochs=max_epochs)
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0,
        end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[10, 11],
        gamma=0.1)
]

# AdamW, lr scaled for 16 GPUs (batch 32 = 2x the standard 8-GPU batch 16):
# 1e-4 -> 2e-4, keeping the effective per-sample lr of the 8-GPU recipe.
optim_wrapper = dict(
    type='OptimWrapper',
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }),
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=2e-4,
        betas=(0.9, 0.999),
        weight_decay=0.05))

custom_hooks = [dict(type='EMAHook', momentum=0.0002)]
