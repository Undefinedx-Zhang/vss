# FPGM (Filter Pruning via Geometric Median)

Geometric median-based filter pruning for convolutional neural networks.

## Method Description

FPGM identifies redundant filters by computing their distance to the geometric median:
- Computes pairwise distances between all filters in a layer
- Approximates geometric median using pairwise distance sums
- Filters closer to the median are considered more redundant
- Prunes filters with smaller distances (more redundant)

## Key Features

- **Geometric approach**: Uses geometric median as redundancy indicator
- **No training required**: Works with pre-trained models directly
- **Robust to outliers**: Geometric median is more robust than arithmetic mean
- **Layer-wise analysis**: Analyzes each convolutional layer independently

## Algorithm

1. For each convolutional layer, flatten filters to vectors
2. Compute pairwise L2 distances between all filter vectors
3. Sum distances for each filter (proxy for distance to geometric median)
4. Rank filters by distance sum (lower = more redundant)
5. Prune filters with lowest distance sums

## Original Paper

"Filter Pruning via Geometric Median for Deep Convolutional Neural Networks Acceleration" (CVPR 2019)
- Authors: Yang He, Ping Liu, Ziwei Wang, Zhilan Hu, Yi Yang
- Repository: https://github.com/he-y/filter-pruning-geometric-median

## Usage Parameters

- No special parameters required
- Works well across different architectures
- Good balance between simplicity and effectiveness
