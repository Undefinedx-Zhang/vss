## 语义分割剪枝基准框架（VSS）

本仓库提供一个专门针对**语义分割任务**的神经网络剪枝基准框架，集成了多种先进的剪枝方法：

- **DMCP**: Differentiable Markov Channel Pruning（参考：[dmcp](https://github.com/zx55/dmcp)）
- **FGP**: Feature-Gradient-Prune（参考：[FGP](https://github.com/FGP-code/FGP)）
- **FPGM**: Filter Pruning via Geometric Median（参考：[filter-pruning-geometric-median](https://github.com/he-y/filter-pruning-geometric-median)）
- **Taylor pruning**: Importance Estimation for Neural Network Pruning（参考：[Taylor_pruning](https://github.com/NVlabs/Taylor_pruning)）
- **Network Slimming**（参考：[slimming](https://github.com/liuzhuang13/slimming)）
- **SIRFP**: Structural Pruning via Spatial-aware Information Redundancy for Semantic Segmentation（参考：[SIRFP](https://github.com/dywu98/SIRFP)）

本框架专为语义分割任务优化，支持 Cityscapes 和 CamVid 数据集，提供 mIoU 等分割专用评测指标。

### 主要功能

#### 语义分割任务
- 数据集：**Cityscapes**（19类）、**CamVid**（12类）
- 模型：**DeepLabV3-ResNet50/101**、**FCN-ResNet50/101**、**ResNet50/101-Seg**、**Simple SegNet**、**U-Net**
- 剪枝方法：
  - Taylor（BN-γ 一阶泰勒重要性，参考 [Taylor_pruning](https://github.com/NVlabs/Taylor_pruning)）
  - Network Slimming（训练时 L1 正则，依据 BN-γ 阈剪，参考 [slimming](https://github.com/liuzhuang13/slimming)）
  - FPGM（几何中值剪枝，参考 [filter-pruning-geometric-median](https://github.com/he-y/filter-pruning-geometric-median)）
  - Norm（L1/L2）、Random 基线
  - **DMCP**（可微分马尔可夫通道剪枝，参考 [dmcp](https://github.com/zx55/dmcp)）
  - **FGP**（特征梯度剪枝，参考 [FGP](https://github.com/FGP-code/FGP)）
  - **SIRFP**（空间感知信息冗余结构化剪枝，参考 [SIRFP](https://github.com/dywu98/SIRFP)）
#### 评测指标
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

### 快速开始

#### 🚀 一键快速开始
```bash
# 检查环境和数据集
python quick_start.py check --data-root /path/to/cityscapes --dataset cityscapes

# 完整流程（训练+剪枝+评测）
python quick_start.py full \
  --dataset cityscapes --data-root /path/to/cityscapes \
  --model deeplabv3_resnet50 --method fpgm --ratio 0.3

# 快速基准测试
python quick_start.py benchmark \
  --dataset cityscapes --data-root /path/to/cityscapes \
  --model deeplabv3_resnet50
```

#### 📖 详细使用步骤

1) 训练分割基线（以 Cityscapes + DeepLabV3 为例）

```bash
python scripts/train.py \
  --dataset cityscapes --data-root /path/to/cityscapes \
  --model deeplabv3_resnet50 --epochs 100 --batch-size 8 \
  --lr 0.01 --weight-decay 1e-4 \
  --save ./runs/seg_baseline_cityscapes
```

2) 剪枝分割模型（示例：FGP 30% 通道剪枝）

```bash
python scripts/prune.py \
  --dataset cityscapes --data-root /path/to/cityscapes \
  --model deeplabv3_resnet50 \
  --ckpt ./runs/seg_baseline_cityscapes/best.pt \
  --method fgp --global-ratio 0.3 \
  --save ./runs/seg_pruned_fgp_30 \
  --input-size 512,1024
```

3) 评测分割模型（包含 mIoU 等指标）

```bash
python scripts/eval.py \
  --dataset cityscapes --data-root /path/to/cityscapes \
  --model deeplabv3_resnet50 \
  --baseline ./runs/seg_baseline_cityscapes/best.pt \
  --candidate ./runs/seg_pruned_fgp_30/pruned_model.pt \
  --report ./runs/seg_pruned_fgp_30/seg_report.json \
  --input-size 512,1024
```

4) 批量分割基准测试

```bash
python scripts/benchmark.py \
  --dataset cityscapes --data-root /path/to/cityscapes \
  --model deeplabv3_resnet50 \
  --baseline ./runs/seg_baseline_cityscapes/best.pt \
  --methods taylor,slimming,fpgm,fgp,sirfp \
  --ratios 0.2,0.3,0.5 \
  --out-dir ./runs/seg_benchmark \
  --input-size 512,1024
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

### 评测说明

#### 分割专用指标
- **mIoU**（平均交并比）：语义分割标准评测指标
- **像素准确率**：正确分类像素的比例
- **类别准确率**：每个类别的平均准确率
- **频率加权 IoU**：考虑类别频率的 IoU
- **性能保持率**：`mIoU_pruned / mIoU_baseline`

#### 通用指标
- **剪枝比**：统计所有可剪 Conv 的输出通道移除比例
- **速度提升**：延迟、FPS、像素/秒的对比倍数
- **冗余识别精度**：与 Oracle 重要性排序的 Top-K 重合率
- **模型重构误差**：剪枝前后模型输出的均方误差（MSE）
- **模型复杂度**：参数量、FLOPs 的变化

### 目录结构

```
VSS/
  README.md
  requirements.txt
  bench/
    __init__.py
    data.py              # 分割数据集加载器（Cityscapes, CamVid）
    metrics.py           # 分割评测指标（mIoU 等）
    train_eval.py        # 分割训练评估
    models/
      __init__.py
      segmentation.py    # 分割模型（DeepLabV3, FCN, U-Net, SegNet）
    pruners/
      __init__.py
      base.py
      taylor.py
      slimming.py
      fpgm.py
      norm.py
      dmcp/              # DMCP 可微分马尔可夫通道剪枝
      fgp/               # FGP 特征梯度剪枝  
      sirfp/             # SIRFP 空间感知信息冗余剪枝
  scripts/
    train.py             # 分割模型训练
    prune.py             # 分割模型剪枝
    eval.py              # 分割模型评测
    benchmark.py         # 分割批量基准测试
```
