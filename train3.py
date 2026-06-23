import os
import json
from mmengine.runner import set_random_seed, Runner
from mmengine.config import Config

# ========== 1. 基础配置与随机种子 ==========
set_random_seed(42, deterministic=True)

# ========== 2. 数据集路径（COCO标准目录） ==========
TRAIN_ANNO = "./data/coco/annotations/instances_train2017.json"
TRAIN_IMG_DIR = "./data/coco/train2017"
VAL_ANNO = "./data/coco/annotations/instances_val2017.json"
VAL_IMG_DIR = "./data/coco/val2017"

# 自动读取类别数量
with open(TRAIN_ANNO, "r", encoding="utf-8") as f:
    coco_json = json.load(f)
category_list = sorted(coco_json["categories"], key=lambda x: x["id"])
class_names = [cat["name"] for cat in category_list]
num_classes = len(class_names)
print(f"检测类别: {class_names}，类别总数: {num_classes}")

# ========== 3. 加载 RetinaNet R50-FPN 1x 配置 ==========
# ⚠️ 确保这个配置文件是 MMDetection 3.x 版本的
cfg = Config.fromfile("./configs/retinanet/retinanet_r50_fpn_1x_coco.py")

# 设置工作目录
cfg.work_dir = "./output_retinanet_mmd"
os.makedirs(cfg.work_dir, exist_ok=True)

# ========== 4. 数据集绑定与元数据配置 ==========

# 1. 定义 metainfo（注意：MMDetection 3.x 中键名必须是小写的 'classes' 和 'palette'）
metainfo = dict(
    classes=tuple(class_names),  # 这里会自动变成 ('un', 'a9')
    palette=[(220, 20, 60), (119, 11, 32)]  # 设置2个类别对应的颜色
)

# 2. 统一设置 data_root
cfg.train_dataloader.dataset.data_root = "./data/coco/"
cfg.val_dataloader.dataset.data_root = "./data/coco/"
cfg.test_dataloader.dataset.data_root = "./data/coco/"

# 3. 将 metainfo 注入到所有的 dataloader 中（解决 concatenate 报错的核心）
cfg.train_dataloader.dataset.metainfo = metainfo
cfg.val_dataloader.dataset.metainfo = metainfo
cfg.test_dataloader.dataset.metainfo = metainfo

# 4. 设置相对路径（相对于 data_root）
cfg.train_dataloader.dataset.ann_file = "annotations/instances_train2017.json"
cfg.train_dataloader.dataset.data_prefix.img = "train2017/"

cfg.val_dataloader.dataset.ann_file = "annotations/instances_val2017.json"
cfg.val_dataloader.dataset.data_prefix.img = "val2017/"

cfg.test_dataloader.dataset.ann_file = "annotations/instances_val2017.json"
cfg.test_dataloader.dataset.data_prefix.img = "val2017/"

# 5. 修改类别数
cfg.model.bbox_head.num_classes = num_classes

# ========== 5. 预训练权重 ==========
# 如果有本地权重，取消下面这行的注释并修改路径
# cfg.load_from = "./configs/model_final_bfca0b.pkl"

# ========== 6. 训练超参数 (复刻 Detectron2 配置) ==========
# 批次大小 (3.x 在 dataloader 中配置)
cfg.train_dataloader.batch_size = 1

# 优化器配置 (3.x 使用 optim_wrapper)
cfg.optim_wrapper.optimizer.lr = 0.0001

# 学习率调度器 (3.x 使用 param_scheduler)
cfg.param_scheduler = [
    dict(
        type='LinearLR', 
        start_factor=0.001, 
        by_epoch=False,  # 按 iter 训练时设为 False
        begin=0, 
        end=500
    ),
    dict(
        type='MultiStepLR', 
        by_epoch=False,  # 按 iter 训练时设为 False
        milestones=[1800, 2400], 
        gamma=0.1
    )
]

# 训练总轮数/迭代数 (3.x 使用 train_cfg)
cfg.train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=3000,       # 按 iter 训练
    val_interval=300      # 每300个iter验证一次
)

# ========== 7. 模型参数对齐 ==========
# Focal Loss 参数
cfg.model.bbox_head.loss_cls.gamma = 2.0
cfg.model.bbox_head.loss_cls.alpha = 0.25

# ⚠️ 【关键】：彻底注释掉所有手动修改 Anchor 的代码，防止 CUDA 维度报错！
# cfg.model.bbox_head.anchor_generator.ratios = [0.5, 1.0, 2.0]
# cfg.model.bbox_head.anchor_generator.scales = [4, 8, 16, 32, 64]
# cfg.model.bbox_head.anchor_generator.strides = [8, 16, 32, 64, 128]

# if 'octave_base_scale' in cfg.model.bbox_head.anchor_generator:
#     del cfg.model.bbox_head.anchor_generator['octave_base_scale']
# if 'scales_per_octave' in cfg.model.bbox_head.anchor_generator:
#     del cfg.model.bbox_head.anchor_generator['scales_per_octave']

# 多尺度训练 + 水平翻转 (3.x 在 train_pipeline 中配置)
for transform in cfg.train_pipeline:
    if transform.type == 'Resize':
        transform.scale = [(640, 1400), (672, 1400), (704, 1400), 
                           (736, 1400), (768, 1400), (800, 1400), (900, 1400)]
        transform.keep_ratio = True
    if transform.type == 'RandomFlip':
        transform.prob = 0.5

# 测试时的图片缩放
for transform in cfg.test_pipeline:
    if transform.type == 'Resize':
        transform.scale = (800, 1333)
        transform.keep_ratio = True

# ========== 8. 启动训练 ==========
if __name__ == "__main__":
    # 使用 Runner 统一接管训练流程
    runner = Runner.from_cfg(cfg)
    runner.train()