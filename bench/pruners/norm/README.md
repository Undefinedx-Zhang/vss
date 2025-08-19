# Norm-based Pruning Methods

Collection of norm-based and baseline pruning methods.

## Included Methods

### L1 Norm Pruning
- **Description**: Uses L1 norm of filter weights as importance measure
- **Formula**: importance(filter) = ||filter||_1
- **Characteristics**: Tends to select sparse, interpretable filters

### L2 Norm Pruning  
- **Description**: Uses L2 norm of filter weights as importance measure
- **Formula**: importance(filter) = ||filter||_2
- **Characteristics**: Considers overall magnitude, less sensitive to individual large weights

### Random Pruning
- **Description**: Random baseline for comparison
- **Formula**: importance(filter) = random()
- **Purpose**: Provides lower bound for pruning method performance

## Key Features

- **Simple and fast**: Minimal computational overhead
- **No training required**: Works with pre-trained models
- **Baseline methods**: Good for comparison and quick prototyping
- **Interpretable**: Clear mathematical meaning

## Usage

```python
# L1 norm pruning
pruner = NormPruner(model, config, norm_type="l1")

# L2 norm pruning  
pruner = NormPruner(model, config, norm_type="l2")

# Random pruning (baseline)
pruner = RandomPruner(model, config)
```

## When to Use

- **L1/L2 Norm**: Quick baseline comparisons, simple pruning needs
- **Random**: Lower bound baseline, debugging, sanity checks
- **Prototyping**: Fast iteration during method development
- **Comparison**: Reference point for more sophisticated methods
