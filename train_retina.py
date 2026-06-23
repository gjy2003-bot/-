import os
import json
import random
import numpy as np
import torch
from mmdet.apis import train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmcv import Config

# 手动固定随机种子
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------------- 正确路径（data在mmdetection_source_bak内部） ----------------------
BASE = "/root/mmdetection_source_bak"
TRAIN_ANNO = os.path.join(BASE, "mmdetection_bak/data/coco/annotations/instances_train2017.json")
TRAIN_IMG_DIR = os.path.join(BASE, "mmdetection_bak/data/coco/train2017")
VAL_ANNO = os.path.join(BASE, "mmdetection_bak/data/coco/annotations/instances_val2017.json")
VAL_IMG_DIR = os.path.join(BASE, "mmdetection_bak/data/coco/val2017")
# ----------------------------------------------------------------------------------------

# 加载配置文件
cfg_path = os.path.join(BASE, "mmdetection_bak/configs/retinanet/retinanet_r50_fpn_1x_coco.py")
cfg = Config.fromfile(cfg_path)

# 强制覆盖配置里的数据集路径，避免配置文件内路径冲突
cfg.data.train.dataset.ann_file = TRAIN_ANNO
cfg.data.train.dataset.img_prefix = TRAIN_IMG_DIR
cfg.data.val.ann_file = VAL_ANNO
cfg.data.val.img_prefix = VAL_IMG_DIR
cfg.data.test.ann_file = VAL_ANNO
cfg.data.test.img_prefix = VAL_IMG_DIR

# 读取类别
with open(TRAIN_ANNO, "r", encoding="utf-8") as f:
    coco_json = json.load(f)
category_list = sorted(coco_json["categories"], key=lambda x: x["id"])
num_classes = len(category_list)
cfg.model.bbox_head.num_classes = num_classes
print(f"检测类别总数：{num_classes}")

if __name__ == "__main__":
    datasets = [build_dataset(cfg.data.train)]
    model = build_detector(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    train_detector(model, datasets, cfg, distributed=False, validate=True)