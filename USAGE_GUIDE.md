# VSS 语义分割剪枝基准框架 - 详细使用指南

## 📋 目录

1. [环境配置](#环境配置)
2. [数据集准备](#数据集准备)
3. [模型与方法介绍](#模型与方法介绍)
4. [基础使用流程](#基础使用流程)
5. [批量基准测试](#批量基准测试)
6. [参数配置说明](#参数配置说明)
7. [结果分析](#结果分析)
8. [常见问题](#常见问题)

---

## 🔧 环境配置

### 系统要求
- Python >= 3.8
- PyTorch >= 2.1.0
- CUDA >= 11.0 (推荐)
- 内存 >= 16GB
- GPU 内存 >= 8GB (用于 Cityscapes)

### 安装依赖
```bash
cd VSS
pip install -r requirements.txt
```

### 验证安装
```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## 📁 数据集准备

### Cityscapes 数据集

#### 1. 下载数据集
访问 [Cityscapes官网](https://www.cityscapes-dataset.com/) 注册并下载：
- `leftImg8bit_trainvaltest.zip` (训练/验证/测试图像)
- `gtFine_trainvaltest.zip` (精细标注)

#### 2. 解压与组织
```bash
# 创建数据目录
mkdir -p /path/to/datasets/cityscapes

# 解压文件
unzip leftImg8bit_trainvaltest.zip -d /path/to/datasets/cityscapes/
unzip gtFine_trainvaltest.zip -d /path/to/datasets/cityscapes/

# 最终目录结构
cityscapes/
├── leftImg8bit/
│   ├── train/          # 2975 张训练图像
│   ├── val/            # 500 张验证图像  
│   └── test/           # 1525 张测试图像
└── gtFine/
    ├── train/          # 训练标注
    ├── val/            # 验证标注
    └── test/           # 测试标注
```

#### 3. 数据集信息
- **图像尺寸**: 2048×1024
- **类别数**: 19 (有效分割类别)
- **训练集**: 2975 张图像
- **验证集**: 500 张图像
- **标注格式**: PNG，像素值对应类别ID

### CamVid 数据集

#### 1. 下载数据集
CamVid 数据集可从多个来源获取，推荐结构：

```bash
# 创建数据目录
mkdir -p /path/to/datasets/camvid

# 目录结构
camvid/
├── train/              # 训练图像
├── val/                # 验证图像  
├── test/               # 测试图像
├── trainannot/         # 训练标注
├── valannot/           # 验证标注
└── testannot/          # 测试标注
```

#### 2. 数据集信息
- **图像尺寸**: 720×960 (原始)，建议调整为 360×480
- **类别数**: 12
- **训练集**: ~367 张图像
- **验证集**: ~101 张图像
- **测试集**: ~233 张图像

---

## 🏗️ 模型与方法介绍

### 支持的分割模型

| 模型名称 | 描述 | Backbone | 特点 |
|---------|------|----------|------|
| `deeplabv3_resnet50` | DeepLabV3 + ResNet50 | ResNet50 | 最先进，精度高 |
| `deeplabv3_resnet101` | DeepLabV3 + ResNet101 | ResNet101 | 更深网络，精度更高 |
| `fcn_resnet50` | FCN + ResNet50 | ResNet50 | 经典全卷积网络 |
| `fcn_resnet101` | FCN + ResNet101 | ResNet101 | 更深的FCN |
| `resnet50_seg` | ResNet50 + 简单分割头 | ResNet50 | 轻量级分割 |
| `resnet101_seg` | ResNet101 + 简单分割头 | ResNet101 | 更深的轻量级分割 |
| `simple_segnet` | 简化的SegNet | 自定义 | 快速原型 |
| `unet` | U-Net | 自定义 | 医学图像风格 |

### 支持的剪枝方法

| 方法 | 类型 | 特点 | 训练要求 | 推荐场景 |
|------|------|------|----------|----------|
| `taylor` | 梯度基础 | BN-γ泰勒重要性 | 无 | 快速基线 |
| `slimming` | 稀疏训练 | BN-γ稀疏化 | L1正则 | 端到端训练 |
| `fpgm` | 几何基础 | 几何中值距离 | 无 | 稳健剪枝 |
| `l1` | 范数基础 | L1范数重要性 | 无 | 简单快速 |
| `l2` | 范数基础 | L2范数重要性 | 无 | 简单快速 |
| `random` | 随机基线 | 随机选择 | 无 | 下界基线 |
| `dmcp` | 可微分 | 马尔可夫过程 | 长时间训练 | 高精度剪枝 |
| `fgp` | 特征梯度 | 特征-梯度相关性 | 训练数据 | 高精度剪枝 |
| `sirfp` | 空间感知 | 空间信息冗余 | 训练数据 | 分割专用 |

---

## 🚀 基础使用流程

### 第一步：训练基线模型

#### Cityscapes 训练
```bash
python scripts/train.py \
    --dataset cityscapes \
    --data-root /path/to/cityscapes \
    --model deeplabv3_resnet50 \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --scheduler poly \
    --save ./runs/cityscapes_deeplabv3_baseline \
    --num-workers 4
```

#### CamVid 训练
```bash
python scripts/train.py \
    --dataset camvid \
    --data-root /path/to/camvid \
    --model deeplabv3_resnet50 \
    --epochs 200 \
    --batch-size 16 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --scheduler poly \
    --save ./runs/camvid_deeplabv3_baseline \
    --num-workers 4
```

#### Network Slimming 训练 (需要L1正则)
```bash
python scripts/train.py \
    --dataset cityscapes \
    --data-root /path/to/cityscapes \
    --model deeplabv3_resnet50 \
    --epochs 100 \
    --batch-size 8 \
    --lr 0.01 \
    --weight-decay 1e-4 \
    --scheduler poly \
    --slimming-lambda 1e-4 \
    --save ./runs/cityscapes_slimming_baseline \
    --num-workers 4
```

### 第二步：模型剪枝

#### 基础剪枝命令
```bash
python scripts/prune.py \
    --dataset cityscapes \
    --data-root /path/to/cityscapes \
    --model deeplabv3_resnet50 \
    --ckpt ./runs/cityscapes_deeplabv3_baseline/best.pt \
    --method fpgm \
    --global-ratio 0.3 \
    --save ./runs/pruned_fpgm_30 \
    --input-size 512,1024
```

#### 不同方法的剪枝示例

**Taylor 剪枝**
```bash
python scripts/prune.py \
    --method taylor \
    --global-ratio 0.3 \
    --ckpt ./runs/baseline/best.pt \
    --save ./runs/pruned_taylor_30
```

**Network Slimming 剪枝**
```bash
python scripts/prune.py \
    --method slimming \
    --global-ratio 0.3 \
    --ckpt ./runs/slimming_baseline/best.pt \
    --save ./runs/pruned_slimming_30
```

**SIRFP 剪枝 (分割专用)**
```bash
python scripts/prune.py \
    --method sirfp \
    --global-ratio 0.3 \
    --ckpt ./runs/baseline/best.pt \
    --save ./runs/pruned_sirfp_30
```

### 第三步：微调剪枝模型

```bash
python scripts/train.py \
    --dataset cityscapes \
    --data-root /path/to/cityscapes \
    --model-from ./runs/pruned_fpgm_30/pruned_model.pt \
    --epochs 50 \
    --batch-size 8 \
    --lr 0.005 \
    --weight-decay 1e-4 \
    --scheduler poly \
    --save ./runs/finetuned_fpgm_30 \
    --num-workers 4
```

### 第四步：评测结果

```bash
python scripts/eval.py \
    --dataset cityscapes \
    --data-root /path/to/cityscapes \
    --model deeplabv3_resnet50 \
    --baseline ./runs/cityscapes_deeplabv3_baseline/best.pt \
    --candidate ./runs/finetuned_fpgm_30/best.pt \
    --report ./runs/finetuned_fpgm_30/evaluation_report.json \
    --batch-size 4 \
    --input-size 512,1024 \
    --oracle-samples 32
```

---

## 📊 批量基准测试

### 快速基准测试

```bash
python scripts/benchmark.py \
    --dataset cityscapes \
    --data-root /path/to/cityscapes \
    --model deeplabv3_resnet50 \
    --baseline ./runs/cityscapes_deeplabv3_baseline/best.pt \
    --methods taylor,fpgm,l1,l2,random \
    --ratios 0.2,0.3,0.5 \
    --out-dir ./runs/quick_benchmark \
    --batch-size 4 \
    --input-size 512,1024
```

### 完整基准测试

```bash
python scripts/benchmark.py \
    --dataset cityscapes \
    --data-root /path/to/cityscapes \
    --model deeplabv3_resnet50 \
    --baseline ./runs/cityscapes_deeplabv3_baseline/best.pt \
    --methods taylor,slimming,fpgm,l1,l2,random,fgp,sirfp \
    --ratios 0.1,0.2,0.3,0.4,0.5,0.6,0.7 \
    --out-dir ./runs/full_benchmark \
    --batch-size 4 \
    --input-size 512,1024 \
    --oracle-samples 64 \
    --skip-existing
```

### 多模型对比

```bash
# ResNet50 backbone
python scripts/benchmark.py \
    --model deeplabv3_resnet50 \
    --baseline ./runs/deeplabv3_resnet50_baseline/best.pt \
    --out-dir ./runs/benchmark_resnet50

# ResNet101 backbone  
python scripts/benchmark.py \
    --model deeplabv3_resnet101 \
    --baseline ./runs/deeplabv3_resnet101_baseline/best.pt \
    --out-dir ./runs/benchmark_resnet101
```

---

## ⚙️ 参数配置说明

### 训练参数

| 参数 | 默认值 | 说明 | 推荐设置 |
|------|--------|------|----------|
| `--epochs` | 100 | 训练轮数 | Cityscapes: 100, CamVid: 200 |
| `--batch-size` | 8 | 批大小 | 根据GPU内存调整 (4-16) |
| `--lr` | 0.01 | 学习率 | DeepLab: 0.01, 轻量模型: 0.1 |
| `--weight-decay` | 1e-4 | 权重衰减 | 1e-4 (标准设置) |
| `--scheduler` | poly | 学习率调度 | poly (分割标准) |
| `--slimming-lambda` | 0.0 | BN L1正则权重 | Slimming: 1e-4 |

### 剪枝参数

| 参数 | 默认值 | 说明 | 推荐设置 |
|------|--------|------|----------|
| `--global-ratio` | 0.3 | 全局剪枝比例 | 0.2-0.5 (保守), 0.5-0.7 (激进) |
| `--input-size` | - | 输入尺寸 H,W | Cityscapes: 512,1024; CamVid: 360,480 |

### 评测参数

| 参数 | 默认值 | 说明 | 推荐设置 |
|------|--------|------|----------|
| `--batch-size` | 4 | 评测批大小 | 4-8 (内存允许下尽量大) |
| `--oracle-samples` | 16 | Oracle分析样本数 | 16-64 (更多=更准确但更慢) |

---

## 📈 结果分析

### 评测报告解读

评测完成后会生成JSON格式的详细报告，包含以下关键指标：

#### 分割专用指标
```json
{
  "candidate_results": {
    "mIoU": 0.724,                    // 平均交并比 (主要指标)
    "Pixel_Accuracy": 0.891,          // 像素准确率
    "Mean_Accuracy": 0.798,           // 类别平均准确率
    "Frequency_Weighted_IoU": 0.832   // 频率加权IoU
  }
}
```

#### 性能对比
```json
{
  "baseline_results": {"mIoU": 0.756},
  "candidate_results": {"mIoU": 0.724},
  "miou_retention": 0.958,            // 性能保持率 (95.8%)
  "pixel_accuracy_retention": 0.962
}
```

#### 效率提升
```json
{
  "baseline_speed": {"fps": 12.3, "avg_ms_per_iter": 81.3},
  "candidate_speed": {"fps": 18.7, "avg_ms_per_iter": 53.5},
  "speedup_latency": 1.52,           // 延迟提升1.52倍
  "speedup_fps": 1.52                // FPS提升1.52倍
}
```

#### 模型复杂度
```json
{
  "baseline_flops_params": {"FLOPs": "182.3G", "Params": "39.6M"},
  "candidate_flops_params": {"FLOPs": "127.8G", "Params": "28.1M"},
  "pruning_ratios": {
    "channel_prune_ratio": 0.31,     // 实际通道剪枝比例
    "channels_before": 2048,
    "channels_after": 1413
  }
}
```

### 基准测试结果分析

批量基准测试会生成汇总报告 `benchmark_summary.json`：

```json
{
  "summary_stats": {
    "total_experiments": 35,
    "completed": 34,
    "failed": 1,
    "success_rate": 0.971
  },
  "results": [
    {
      "method": "fpgm",
      "ratio": 0.3,
      "miou_retention": 0.958,
      "speedup_latency": 1.52,
      "channel_reduction": 0.31
    }
  ]
}
```

### 结果可视化

可以使用Python脚本分析结果：

```python
import json
import matplotlib.pyplot as plt
import pandas as pd

# 读取基准测试结果
with open('./runs/benchmark/benchmark_summary.json', 'r') as f:
    data = json.load(f)

results = pd.DataFrame(data['results'])

# 绘制性能保持率 vs 剪枝比例
plt.figure(figsize=(10, 6))
for method in results['method'].unique():
    method_data = results[results['method'] == method]
    plt.plot(method_data['ratio'], method_data['miou_retention'], 
             marker='o', label=method)

plt.xlabel('Pruning Ratio')
plt.ylabel('mIoU Retention')
plt.title('Pruning Performance Comparison')
plt.legend()
plt.grid(True)
plt.show()
```

---

## ❓ 常见问题

### Q1: 训练时出现OOM (Out of Memory) 错误
**A**: 
- 减小batch size: `--batch-size 4` 或 `--batch-size 2`
- 减小输入尺寸: `--input-size 256,512`
- 使用gradient checkpointing (需要修改模型代码)

### Q2: Cityscapes数据集加载失败
**A**:
- 检查数据集路径是否正确
- 确保标注文件为 `*_gtFine_labelTrainIds.png` 格式
- 验证目录结构是否符合要求

### Q3: Network Slimming 效果不佳
**A**:
- 确保训练时使用了L1正则: `--slimming-lambda 1e-4`
- 尝试不同的lambda值: `1e-5` 到 `1e-3`
- 训练更多轮次让BN参数充分稀疏化

### Q4: 剪枝后精度下降过多
**A**:
- 降低剪枝比例: `--global-ratio 0.2`
- 增加微调轮次: `--epochs 100`
- 使用更保守的方法: `fpgm` 而非 `random`

### Q5: FGP/SIRFP 运行很慢
**A**:
- 减少分析样本: `--oracle-samples 16`
- 使用更小的batch size
- 这些方法需要训练数据分析，本身较慢

### Q6: 基准测试中断后如何继续
**A**:
使用 `--skip-existing` 参数：
```bash
python scripts/benchmark.py --skip-existing --out-dir ./runs/benchmark
```

### Q7: 如何确保公平比较
**A**:
- 使用相同的基线模型
- 统一输入尺寸和batch size
- 相同的微调策略和轮次
- 相同的评测设置

### Q8: 模型加载错误
**A**:
- 检查checkpoint路径是否正确
- 确保模型架构匹配
- 使用 `strict=False` 参数 (已在代码中设置)

---

## 🎯 最佳实践

### 1. 训练策略
- **数据增强**: 已内置基础增强，可根据需要扩展
- **学习率调度**: 分割任务推荐使用 `poly` 调度器
- **权重衰减**: 标准设置 `1e-4`，可根据模型大小调整

### 2. 剪枝策略
- **渐进式剪枝**: 从小比例开始 (0.1 → 0.2 → 0.3)
- **方法选择**: SIRFP > FGP > FPGM > Taylor > L1/L2 > Random
- **微调重要性**: 剪枝后必须充分微调

### 3. 评测策略
- **多次运行**: 重要实验运行3次取平均
- **多个指标**: 不仅看mIoU，也关注像素准确率和速度
- **可视化检查**: 定期检查分割结果的视觉质量

### 4. 基准测试
- **系统性对比**: 使用统一的实验设置
- **充分消融**: 测试多个剪枝比例和方法
- **文档记录**: 详细记录实验设置和结果

---

## 📚 参考资料

### 原始论文
1. **Taylor**: "Importance Estimation for Neural Network Pruning" (CVPR 2019)
2. **Network Slimming**: "Learning Efficient Convolutional Networks through Network Slimming" (ICCV 2017)  
3. **FPGM**: "Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration" (CVPR 2019)
4. **DMCP**: "DMCP: Differentiable Markov Channel Pruning for Neural Networks" (CVPR 2020)
5. **SIRFP**: "Structural Pruning via Spatial-aware Information Redundancy for Semantic Segmentation" (AAAI 2023)

### 数据集论文
1. **Cityscapes**: "The Cityscapes Dataset for Semantic Urban Scene Understanding" (CVPR 2016)
2. **CamVid**: "Semantic Object Classes in Video: A High-Definition Ground Truth Database" (PRL 2009)

---

## 💡 技巧与建议

1. **GPU内存优化**: 使用混合精度训练可以减少内存使用
2. **数据并行**: 多GPU训练可以显著加速
3. **检查点保存**: 定期保存中间结果，避免重复计算
4. **日志监控**: 使用TensorBoard或wandb监控训练过程
5. **结果备份**: 重要实验结果及时备份

通过本指南，您应该能够顺利使用VSS框架进行语义分割模型的剪枝实验。如有问题，请参考代码注释或提交Issue。
