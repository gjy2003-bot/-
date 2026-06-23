import os
import json
import random
import shutil

# ===================== 配置区 =====================
root = "./data"
train_ratio = 0.8
random.seed(42)  # 固定随机种子，划分可复现
# ==================================================

# 输出COCO标准目录
coco_root = os.path.join(root, "coco")
train_img_dir = os.path.join(coco_root, "train2017")
val_img_dir = os.path.join(coco_root, "val2017")
ann_out_dir = os.path.join(coco_root, "annotations")
os.makedirs(train_img_dir, exist_ok=True)
os.makedirs(val_img_dir, exist_ok=True)
os.makedirs(ann_out_dir, exist_ok=True)

# 1. 收集所有图片+对应标注文件
all_samples = []
sub_dirs = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d.isdigit()]
for sub in sub_dirs:
    sub_path = os.path.join(root, sub)
    file_list = os.listdir(sub_path)
    # 匹配图片文件
    img_suffix = (".jpg", ".png", ".jpeg", ".JPG", ".PNG")
    for fname in file_list:
        if fname.endswith(img_suffix):
            img_abs = os.path.join(sub_path, fname)
            # 匹配同名标注json
            json_name = os.path.splitext(fname)[0] + ".json"
            json_abs = os.path.join(sub_path, json_name)
            if os.path.exists(json_abs):
                all_samples.append((img_abs, json_abs, fname))

print(f"总样本数量：{len(all_samples)}")

# 2. 划分训练/验证集
random.shuffle(all_samples)
split_point = int(len(all_samples) * train_ratio)
train_samples = all_samples[:split_point]
val_samples = all_samples[split_point:]
print(f"训练集：{len(train_samples)} 张，验证集：{len(val_samples)} 张")

# 3. 复制图片到对应文件夹
def copy_sample(sample_list, target_img_dir):
    for img_abs, _, fname in sample_list:
        dst = os.path.join(target_img_dir, fname)
        shutil.copy(img_abs, dst)

copy_sample(train_samples, train_img_dir)
copy_sample(val_samples, val_img_dir)

# 4. 合并单图json为标准COCO格式
def merge_coco(sample_list):
    full_coco = {
        "info": {},
        "licenses": [],
        "categories": [],
        "images": [],
        "annotations": []
    }
    img_id = 1
    ann_id = 1
    cat_set = set()

    for img_abs, json_abs, fname in sample_list:
        with open(json_abs, "r", encoding="utf-8") as f:
            single_data = json.load(f)
        # 填充图片信息
        img_info = {
            "id": img_id,
            "file_name": fname,
            "width": single_data["imageWidth"],
            "height": single_data["imageHeight"]
        }
        full_coco["images"].append(img_info)
        # 填充标注
        for shape in single_data["shapes"]:
            cat_name = shape["label"]
            if cat_name not in cat_set:
                cat_set.add(cat_name)
                full_coco["categories"].append({"id": len(cat_set), "name": cat_name})
            cat_id = list(cat_set).index(cat_name) + 1
            # 多边形转bbox（MMDetection通用）
            points = shape["points"]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            xmin, ymin = min(xs), min(ys)
            xmax, ymax = max(xs), max(ys)
            w = xmax - xmin
            h = ymax - ymin
            ann = {
                "id": ann_id,
                "image_id": img_id,
                "category_id": cat_id,
                "bbox": [xmin, ymin, w, h],
                "area": w * h,
                "iscrowd": 0,
                "segmentation": [sum(points, [])]
            }
            full_coco["annotations"].append(ann)
            ann_id += 1
        img_id += 1
    return full_coco

# 生成train、val标注json
train_coco = merge_coco(train_samples)
val_coco = merge_coco(val_samples)

# 保存文件
with open(os.path.join(ann_out_dir, "instances_train2017.json"), "w", encoding="utf-8") as f:
    json.dump(train_coco, f, indent=2)
with open(os.path.join(ann_out_dir, "instances_val2017.json"), "w", encoding="utf-8") as f:
    json.dump(val_coco, f, indent=2)

print("数据集划分完成！")
print(f"Train: {len(train_coco['images'])} 图，{len(train_coco['annotations'])} 标注")
print(f"Val: {len(val_coco['images'])} 图，{len(val_coco['annotations'])} 标注")