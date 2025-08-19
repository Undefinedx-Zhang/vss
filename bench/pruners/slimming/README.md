# Network Slimming

Channel pruning method using BN gamma magnitude as importance indicator.

## Method Description

Network Slimming uses BatchNorm scaling factors (gamma) to identify important channels:
- Channels with larger |gamma| values are considered more important
- Requires L1 regularization on BN gamma during training for sparsity
- Prunes channels with gamma values below a threshold

## Key Features

- **Training-time regularization**: Requires L1 penalty on BN gamma during training
- **Magnitude-based**: Uses absolute values of BN scaling factors
- **Threshold pruning**: Can use either fixed threshold or global ratio
- **Interpretable**: Clear physical meaning of importance scores

## Training Requirements

**Important**: For best results, add L1 regularization during training:
```bash
python scripts/train.py --slimming-lambda 1e-4  # Add L1 penalty
```

## Original Paper

"Learning Efficient Convolutional Networks through Network Slimming" (ICCV 2017)
- Authors: Zhuang Liu, Jianguo Li, Zhiqiang Shen, Gao Huang, Shoumeng Yan, Changshui Zhang
- Repository: https://github.com/liuzhuang13/slimming

## Usage Parameters

- `slimming_threshold`: Fixed threshold for pruning (optional)
- `slimming_lambda`: L1 regularization weight during training (e.g., 1e-4)
- Works best when training includes BN regularization
