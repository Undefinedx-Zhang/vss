# DMCP Integration

Differentiable Markov Channel Pruning for Neural Networks (CVPR 2020 Oral)

Original repository: https://github.com/zx55/dmcp

## Key Features
- Differentiable channel pruning using Markov process modeling
- Eliminates duplicated solutions from independent Bernoulli variables
- Two-stage training: stage1 (normal training) + stage2 (DMCP process)
- Supports MobileNetV2 and ResNet architectures

## Integration Notes
- Adapted to work with our unified pruning interface
- Simplified for CIFAR-10/ResNet compatibility
- Core DMCP logic preserved from original implementation
