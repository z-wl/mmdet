# model settings
model = dict(
    type='YOLOAF',
    use_fft=True,
    backbone=dict(
        type='ShuffleNetV2',
        stage_out_channels=[-1, 24, 48, 96, 192],
        load_param=False,
        all_output=True,
        data_channel=4,
        yolo_af=True
    ),
    neck=None,
    bbox_head=dict(
        type='TTFHead',
        inplanes=(24, 48, 96, 192),
        head_conv=64,
        wh_conv=32,
        hm_head_conv_num=2,
        wh_head_conv_num=2,
        num_classes=7,
        wh_offset_base=16,
        wh_agnostic=True,
        wh_gaussian=True,
        shortcut_cfg=(1, 2, 3),
        norm_cfg=dict(type='BN'),
        alpha=0.54,
        hm_weight=1.,
        wh_weight=5.)
)
cudnn_benchmark = True
# training and testing settings
train_cfg = dict(
    vis_every_n_iters=100,
    debug=False)
test_cfg = dict(
    score_thr=0.01,
    max_per_img=100)
# dataset settings
dataset_type = 'LaSOT'
data_root = 'G:/Datasets/LaSOT/LaSOTBenchmark/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=['filename','ori_shape',
                            'img_shape', 'img_norm_cfg', 'pad_shape']),
]
test_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'], meta_keys=['filename','ori_shape',
                            'img_shape', 'img_norm_cfg', 'pad_shape', 'scale_factor']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    shuffle=True,
    train=dict(
        type=dataset_type,
        root_dir=data_root,
        split_file='lasot_sub2.json',
        subset='val',
        batch_size=48,
        test_mode=False,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        root_dir=data_root,
        split_file='lasot_sub2.json',
        subset='val',
        batch_size=1,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        split_file='lasot_sub2.json',
        root_dir=data_root,
        subset='val',
        batch_size=1,
        test_mode=True,
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
work_dir = 'work_dirs/yoloaf_1x'
load_from = None
resume_from = None
workflow = [('train', 1)]
