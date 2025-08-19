# FGP Integration

Feature-Gradient-Prune for Efficient Convolutional Layer Pruning

Original repository: https://github.com/FGP-code/FGP

## Key Features
- Uses gradient information and feature maps for channel importance
- Focuses on semantic segmentation but adaptable to classification
- Combines feature analysis with gradient-based importance scoring
- Effective for identifying redundant channels across different classes

## Integration Notes
- Adapted from semantic segmentation to work with classification tasks
- Core FGP logic preserved: gradient-weighted feature importance
- Simplified for CIFAR-10/ResNet compatibility
- Maintains the gradient-feature correlation analysis approach
