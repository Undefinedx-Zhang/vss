# Taylor BN Pruning

Taylor-FO importance estimation using BN gamma and gradient signal.

## Method Description

This method computes channel importance using first-order Taylor expansion:
- importance(channel) ≈ |gamma_c * dL/d(gamma_c)|
- Aggregates importance over multiple samples
- Uses BatchNorm gamma parameters and their gradients

## Key Features

- **BN-based**: Uses BatchNorm gamma parameters as importance indicators
- **Gradient-aware**: Incorporates gradient information for more accurate estimation
- **Sample aggregation**: Averages importance across multiple input samples
- **Fast computation**: Efficient importance scoring without extensive retraining

## Original Paper

"Importance Estimation for Neural Network Pruning" (CVPR 2019)
- Authors: Pavlo Molchanov, Arun Mallya, Stephen Tyree, Iuri Frosio, Jan Kautz
- Repository: https://github.com/NVlabs/Taylor_pruning

## Usage Parameters

- `taylor_samples`: Number of samples for importance estimation (default: 64)
- Works well with both classification and segmentation tasks
- Recommended for quick prototyping and baseline comparisons
