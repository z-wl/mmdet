from mmdet.datasets import RepeatDataset
from mmdet.datasets.builder import build_dataset, build_dataloader


if __name__ == '__main__':
    img_norm_cfg = dict(mean=[0, 0, 0], std=[255., 255., 255.], to_rgb=True)
    pipeline_ = [
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='Expand',
                mean=img_norm_cfg['mean'],
                to_rgb=img_norm_cfg['to_rgb'],
                ratio_range=(1, 2)),
            dict(
                type='MinIoURandomCrop',
                min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
                min_crop_size=0.3),
            dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='PhotoMetricDistortion'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ]
    p2 = [
            dict(type='LoadImageFromFile', to_float32=True),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', img_scale=[(320, 320), (608, 608)], keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=['filename','ori_shape',
                            'img_shape', 'img_norm_cfg'])
        ]
    cfg = dict(
        type = 'CocoDataset',
        data_root = r'D:/Datasets/coco/val2017',
        ann_file = r'D:/Datasets/coco/annotations/instances_val2017.json',
        pipeline = p2
    )

    cfg2 = dict(
        type = 'LaSOT',
        root_dir = r'G:\Datasets\LaSOT\LaSOTBenchmark',
        pipeline = [
            dict(type='Resize', img_scale=(512, 512), keep_ratio=False),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=['filename','ori_shape',
                            'img_shape', 'img_norm_cfg'])
        ]
    )
    dataset = build_dataset(cfg2)

    # print(dataset[0].keys())
    # print(dataset[0])
    # print(dataset.get_ann_info(0))

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        num_gpus=1,
        dist=False,
        shuffle=False
    )

    for data in dataloader:
        print('final', data)
        # print('cpu', batch[0])print(data['img_metas'].data[0].shape)
        print(data['img'].data[0].shape)
        # print(data['gt_bboxes'].data[0][0].shape)
        # print(data['gt_labels'])



# pip install mmcv-full==1.4.1 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html