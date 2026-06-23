_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
    './retinanet_tta.py'
]

# optimizer
optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001))

# ---------------------- 自定义数据集配置 ----------------------
data_root = "./data/coco/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type="CocoDataset",
        ann_file=data_root + "annotations/instances_train2017.json",
        img_prefix=data_root + "train2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(type="LoadAnnotations", with_bbox=True),
            dict(
                type="Resize",
                img_scale=[(640,1400), (672,1400), (704,1400), (736,1400), (768,1400), (800,1400), (900,1400)],
                multiscale_mode="value"
            ),
            dict(type="RandomFlip", flip_ratio=0.5),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
    val=dict(
        type="CocoDataset",
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(800, 1333),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Normalize",** img_norm_cfg),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
    test=dict(
        type="CocoDataset",
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=[
            dict(type="LoadImageFromFile"),
            dict(
                type="MultiScaleFlipAug",
                img_scale=(800, 1333),
                flip=False,
                transforms=[
                    dict(type="Resize", keep_ratio=True),
                    dict(type="RandomFlip"),
                    dict(type="Normalize", **img_norm_cfg),
                    dict(type="Pad", size_divisor=32),
                    dict(type="ImageToTensor", keys=["img"]),
                    dict(type="Collect", keys=["img"]),
                ],
            ),
        ],
    ),
)

# ---------------------- 迭代与学习率衰减（对齐Detectron2） ----------------------
runner = dict(type="IterBasedRunner", max_iters=3000)
lr_config = dict(
    policy="step",
    warmup="linear",
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[1800, 2400],
    gamma=0.1,
)

# ---------------------- Anchor、FocalLoss参数对齐原代码 ----------------------
model = dict(
    anchor_generator=dict(
        ratios=[0.5, 1.0, 2.0],
        scales=[4, 8, 16, 32, 64],
        strides=[8, 16, 32, 64, 128],
    ),
    bbox_head=dict(
        loss_cls=dict(gamma=2.0, alpha=0.25),
    ),
)

# ---------------------- 每300轮验证一次 ----------------------
evaluation = dict(interval=300, metric="bbox")

# ---------------------- 预训练权重 & 输出目录 ----------------------
load_from = "./configs/model_final_bfca0b.pkl"
work_dir = "./output_retinanet_mmd"