#!/bin/bash

# VSS CamVid 基线模型训练脚本
# 第一步：基线模型训练

set -e  # 遇到错误立即退出

echo "开始训练 CamVid 基线模型..."
echo "数据集: camvid"
echo "数据路径: datasets/camvid"
echo "模型: resnet50_seg"
echo "训练轮数: 200"
echo "批大小: 16"
echo "保存路径: ./runs/camvid_resnet50_seg"
echo "----------------------------------------"

# 检查数据集路径
if [ ! -d "datasets/camvid" ]; then
    echo "错误: 数据集路径不存在: datasets/camvid"
    echo "请确保 CamVid 数据集已正确下载和解压到 datasets/camvid 目录"
    exit 1
fi

# 创建输出目录
mkdir -p ./runs/camvid_resnet50_seg

python train.py \
    --dataset camvid \
    --data-root ../datasets/camvid \
    --model resnet50_seg \
    --epochs 200 \
    --batch-size 16 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --scheduler poly \
    --save ./runs/camvid_resnet50_seg \
    --num-workers 4

echo "----------------------------------------"
echo "基线模型训练完成!"
echo "模型保存在: ./runs/camvid_resnet50_seg/best.pt"
echo "现在可以进行剪枝实验了"
