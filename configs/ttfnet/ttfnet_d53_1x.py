# model settings
model = dict(
    type='TTFNet',
    backbone=dict(
        type='DarknetV3',
        layers=[1, 2, 8, 8, 4],
        inplanes=[3, 32, 64, 128, 256, 512],
        planes=[32, 64, 128, 256, 512, 1024],
        norm_cfg=dict(type='BN'),
        out_indices=(1, 2, 3, 4),
        frozen_stages=1,
        norm_eval=False,
        init_cfg=dict(
            type='Xavier',
        )
    ),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(128, 256, 512, 1024),
        head_conv=128,
        wh_conv=64,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=81,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.),
    train_cfg=dict(
        vis_every_n_iters=100,
        debug=False
    ),
    test_cfg=dict(
        score_thr=0.01,
        max_per_img=100
    )
)

cudnn_benchmark = True
# training and testing settings

# dataset settings
dataset_type = 'CocoDataset'
data_root = r'D:/Datasets/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    imgs_per_gpu=12,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        test_mode=True,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        test_mode=True,
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
# optimizer
optimizer = dict(type='SGD', lr=0.015, momentum=0.9, weight_decay=0.0004,
                 paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 5,
    step=[9, 11])
checkpoint_config = dict(interval=4)
log_config = dict(
    interval=20,
    hooks=[
        dict(type='TextLoggerHook'),
    ])
# yapf:enable
# runtime settings
total_epochs = 12
device_ids = range(8)
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = 'work_dirs/ttfnet53_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
