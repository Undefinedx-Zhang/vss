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

### 引用与参考

- Taylor pruning（CVPR 2019）：[NVlabs/Taylor_pruning](https://github.com/NVlabs/Taylor_pruning)
- Network Slimming（ICCV 2017）：[liuzhuang13/slimming](https://github.com/liuzhuang13/slimming)
- FPGM（ICCV 2019）：[he-y/filter-pruning-geometric-median](https://github.com/he-y/filter-pruning-geometric-median)
- DMCP（CVPR 2020）：`https://github.com/zx55/dmcp`
- FGP（ICLR 2021）：`https://github.com/FGP-code/FGP`
- SIRFP（AAAI 2023）：`https://github.com/dywu98/SIRFP`

### 新增集成方法说明

#### DMCP (Differentiable Markov Channel Pruning)
- 采用马尔可夫过程建模通道剪枝决策，消除独立伯努利变量的重复解
- 使用可微分门控变量进行端到端训练
- 适配为两阶段过程：正常训练 + DMCP 优化

#### FGP (Feature-Gradient-Prune) 
- 基于特征图与梯度的相关性分析通道重要性
- 多类别梯度分析，识别跨类别一致重要的通道
- 从语义分割适配到分类任务

#### SIRFP (Spatial-aware Information Redundancy)
- 空间感知的信息冗余分析
- 多尺度空间特征分析
- 基于空间信息熵和通道间相关性的重要性评估

### 使用建议

#### 方法选择
- **SIRFP**：原本就是为分割任务设计，在分割上表现优异
- **FGP**：从分割适配而来，对分割任务有很好的适应性
- **FPGM**：几何中值方法在分割任务上也有不错的表现
- **Taylor** 和 **Network Slimming**：同样适用于分割模型
- **DMCP**：端到端可微剪枝，需要较长训练时间
- **L1/L2 Norm**：快速原型验证的基础方法

#### 实用建议
- **Batch Size**：分割任务建议使用较小的 batch size（4-8）以适应 GPU 内存限制
- **输入尺寸**：Cityscapes 推荐 512×1024，CamVid 推荐 360×480
- **Network Slimming**：需要在训练阶段添加 L1 正则化（`--slimming-lambda`）
- **训练策略**：分割模型建议使用多项式学习率调度（`--scheduler poly`）

本基准仅用于学术研究与评测复现，具体方法实现细节与开源协议请参考各自原始仓库。

---

## 🔧 新增功能和改进

### 1. 扩展的模型支持
- **ResNet系列Backbone**: 新增 ResNet50/101 作为分割backbone
- **更多预训练模型**: DeepLabV3-ResNet101, FCN-ResNet101
- **轻量级模型**: ResNet-Seg 系列，适合快速实验

### 2. 重新组织的代码结构
每个剪枝方法现在都有独立的文件夹：
```
bench/pruners/
├── taylor/          # Taylor方法
├── slimming/        # Network Slimming
├── fpgm/            # FPGM方法  
├── norm/            # L1/L2/Random
├── dmcp/            # DMCP方法
├── fgp/             # FGP方法
└── sirfp/           # SIRFP方法
```
每个文件夹包含：实现代码、README文档、使用说明

### 3. 训练参数一致性保证
- **标准化配置**: `configs/training_configs.yaml` 确保公平比较
- **方法特定参数**: 针对不同剪枝方法的优化设置
- **数据集适配**: 不同数据集的推荐参数配置
- **自动配置加载**: `bench/config_loader.py` 统一管理

### 4. 完整使用文档
- **详细指南**: `USAGE_GUIDE.md` 包含完整的使用说明
- **数据集准备**: Cityscapes 和 CamVid 的详细设置指南
- **参数调优**: 不同场景下的最佳实践
- **结果分析**: 评测报告解读和可视化方法
- **常见问题**: FAQ 和故障排除

### 5. 便捷工具
- **快速开始脚本**: `quick_start.py` 一键运行工具
- **环境检查**: 自动检查依赖和数据集
- **完整流程**: 训练→剪枝→评测 一键执行
- **基准测试**: 批量对比多种方法

## 📚 使用指南

### 🚀 快速开始
```bash
# 1. 检查环境和数据集
python quick_start.py check --data-root /path/to/cityscapes --dataset cityscapes

# 2. 运行完整流程（训练+剪枝+评测）
python quick_start.py full \
  --dataset cityscapes --data-root /path/to/cityscapes \
  --model deeplabv3_resnet50 --method fpgm --ratio 0.3

# 3. 快速基准测试
python quick_start.py benchmark \
  --dataset cityscapes --data-root /path/to/cityscapes
```

### 📖 详细文档
- **[完整使用指南](USAGE_GUIDE.md)**: 数据集准备、训练配置、结果分析
- **[配置文件说明](configs/training_configs.yaml)**: 标准化训练参数
- **剪枝方法文档**: 每个 `bench/pruners/*/README.md`

### 🎯 使用建议

**新手用户**:
1. 先运行 `python quick_start.py check` 检查环境
2. 使用 `quick_start.py full` 体验完整流程  
3. 阅读 `USAGE_GUIDE.md` 了解详细用法

**研究人员**:
1. 查看 `configs/training_configs.yaml` 了解标准配置
2. 使用 `scripts/benchmark.py` 进行系统性对比
3. 参考各方法的 README 了解实现细节

**开发者**:
1. 参考现有剪枝方法的文件夹结构添加新方法
2. 使用 `bench/config_loader.py` 确保配置一致性
3. 遵循统一的代码风格和文档格式

## 🏆 基准测试公平性保证

1. **统一训练参数**: 基于原始论文推荐设置
2. **一致的评测环境**: 相同的数据预处理和评测指标
3. **标准化流程**: 训练→剪枝→微调→评测 的统一流程
4. **可复现性**: 详细的配置文件和使用文档

---


