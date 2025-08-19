## 语义分割剪枝基准框架（VSS）

本仓库提供一个专门针对**语义分割任务**的神经网络剪枝基准框架，集成了多种先进的剪枝方法，支持完整的训练→剪枝→微调→评估流程：

### 🔥 集成剪枝方法
- **Taylor pruning**: BN-γ 泰勒重要性估计（参考：[Taylor_pruning](https://github.com/NVlabs/Taylor_pruning)）
- **Network Slimming**: L1正则化稀疏训练（参考：[slimming](https://github.com/liuzhuang13/slimming)）
- **FPGM**: 几何中值滤波器剪枝（参考：[filter-pruning-geometric-median](https://github.com/he-y/filter-pruning-geometric-median)）
- **FGP**: 特征-梯度剪枝（参考：[FGP](https://github.com/FGP-code/FGP)）
- **SIRFP**: 空间感知信息冗余剪枝（参考：[SIRFP](https://github.com/dywu98/SIRFP)）
- **DMCP**: 可微分马尔可夫通道剪枝（参考：[dmcp](https://github.com/zx55/dmcp)）
- **Norm-based**: L1/L2范数剪枝 + Random基线

### 🎯 专业分割优化
- **数据集**: Cityscapes（19类）、CamVid（12类）
- **模型**: DeepLabV3、FCN、ResNet-Seg、SegNet、U-Net
- **指标**: mIoU、像素准确率、速度、FLOPs、参数量等
- **物理剪枝**: 基于torch-pruning，确保真实速度收益

## ✨ 主要功能

### 🚀 完整工作流程
1. **模型训练**: 支持多种分割模型的训练
2. **智能剪枝**: 集成7种先进剪枝算法
3. **批量微调**: 自动对剪枝模型进行微调恢复性能
4. **全面评估**: 多维度性能指标评估
5. **批量基准**: 一键对比多种方法的性能

### 📊 评测指标
- **分割专用指标**：
  - **mIoU**（平均交并比）：语义分割标准评测指标
  - **像素准确率**：正确分类像素的比例
  - **类别准确率**：每个类别的平均准确率
  - **频率加权 IoU**：考虑类别频率的 IoU
- **通用指标**：
  - 剪枝比（按通道统计）
  - 推理速度提升倍数（延迟、FPS、像素/秒）
  - 通道冗余识别精度（与 Oracle 的 Top-K 重合率）
  - 模型重构误差（剪枝前后输出 MSE）
  - 模型大小/参数量/FLOPs 变化
- 物理裁剪：集成 `torch-pruning` 自动处理依赖，确保真实速度收益。

### 依赖安装

```
pip install -r requirements.txt
```

主要依赖：
- torch, torchvision
- torch-pruning（通道依赖解析与物理裁剪）
- thop（FLOPs/Params 估计）
- rich（日志美化，可选）
- pyyaml, tqdm, numpy

## 🚀 快速开始

### 1️⃣ 训练基线模型
```bash
# CamVid数据集 + ResNet50-Seg
python train.py \
  --dataset camvid --data-root ./datasets/camvid \
  --model resnet50_seg --epochs 300 --batch-size 16 \
  --save ./runs/camvid_resnet50_seg
```

### 2️⃣ 单个剪枝实验
```bash
# 使用Taylor方法进行10%剪枝
python prune.py \
  --dataset camvid --data-root ./datasets/camvid \
  --model resnet50_seg \
  --ckpt ./runs/camvid_resnet50_seg/best.pt \
  --method taylor --global-ratio 0.1 \
  --save ./runs/taylor_pruned \
  --input-size 360,480
```

### 3️⃣ 批量基准测试
```bash
# 对比多种剪枝方法
python benchmark.py \
  --dataset camvid --data-root ./datasets/camvid \
  --model resnet50_seg \
  --baseline ./runs/camvid_resnet50_seg/best.pt \
  --methods taylor,fpgm,fgp,sirfp \
  --ratios 0.1 \
  --out-dir ./runs/camvid_benchmark
```

### 4️⃣ 批量微调（新功能🆕）
```bash
# 自动对所有剪枝模型进行微调
python finetune_batch.py \
  --benchmark-dir ./runs/camvid_benchmark \
  --dataset camvid --data-root ./datasets/camvid \
  --model resnet50_seg \
  --baseline ./runs/camvid_resnet50_seg/best.pt \
  --finetune-epochs 20
```

### 5️⃣ 单独评估
```bash
# 评估特定模型
python eval.py \
  --dataset camvid --data-root ./datasets/camvid \
  --model resnet50_seg \
  --baseline ./runs/camvid_resnet50_seg/best.pt \
  --candidate ./runs/taylor_pruned/pruned_model.pt \
  --report ./runs/taylor_pruned/report.json
```

### 数据集准备

#### Cityscapes 数据集
1. 从 [Cityscapes官网](https://www.cityscapes-dataset.com/) 下载数据集
2. 解压到如下结构：
```
cityscapes/
├── leftImg8bit/
│   ├── train/
│   ├── val/
│   └── test/
└── gtFine/
    ├── train/
    ├── val/
    └── test/
```

#### CamVid 数据集
1. 从相关渠道下载 CamVid 数据集
2. 解压到如下结构：
```
camvid/
├── train/
├── val/
├── test/
├── trainannot/
├── valannot/
└── testannot/
```

## 📁 项目结构

```
VSS/
├── README.md                    # 项目说明文档
├── requirements.txt             # 依赖包列表
├── train.py                     # 模型训练脚本
├── prune.py                     # 模型剪枝脚本
├── eval.py                      # 模型评估脚本
├── benchmark.py                 # 批量基准测试脚本
├── finetune_batch.py           # 批量微调脚本 🆕
├── datasets/                    # 数据集目录
│   ├── camvid/                 # CamVid数据集
│   └── cityscapes/             # Cityscapes数据集
├── runs/                       # 实验结果目录
│   ├── camvid_resnet50_seg/    # 基线模型
│   └── camvid_benchmark/       # 基准测试结果
├── bench/                      # 核心代码库
│   ├── data.py                 # 数据加载器
│   ├── metrics.py              # 评估指标
│   ├── train_eval.py           # 训练评估工具
│   ├── models/
│   │   ├── __init__.py
│   │   └── segmentation.py     # 分割模型定义
│   └── pruners/                # 剪枝方法实现
│       ├── __init__.py
│       ├── base.py             # 剪枝基类
│       ├── taylor/             # Taylor剪枝
│       ├── slimming/           # Network Slimming
│       ├── fpgm/               # FPGM剪枝
│       ├── norm/               # 范数剪枝
│       ├── fgp/                # FGP剪枝
│       ├── sirfp/              # SIRFP剪枝
│       └── dmcp/               # DMCP剪枝
└── configs/                    # 配置文件
    └── training_configs.yaml   # 训练配置
```
