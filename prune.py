"""
Pruning script for semantic segmentation models
"""
import os
import argparse
import torch

from bench.data import get_dataset_info
from bench.models import build_model
from bench.pruners import build_pruner, PruningConfig


def load_checkpoint(model, ckpt_path, strict=True):
    """Load model checkpoint"""
    state = torch.load(ckpt_path, map_location="cpu")
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=strict)


def parse_args():
    p = argparse.ArgumentParser(description="Prune segmentation model")
    p.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "camvid"])
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--model", type=str, default="deeplabv3_resnet50",
                   choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101",
                           "resnet50_seg", "resnet101_seg", "simple_segnet", "unet"])
    p.add_argument("--ckpt", type=str, required=True, help="Path to trained model checkpoint")
    p.add_argument("--method", type=str, default="fpgm", 
                   choices=["taylor", "slimming", "fpgm", "l1", "l2", "random", "dmcp", "fgp", "sirfp"])
    p.add_argument("--global-ratio", type=float, default=0.3, help="Global pruning ratio")
    p.add_argument("--save", type=str, default="./runs/seg_pruned", help="Save directory")
    p.add_argument("--input-size", type=str, default="512,1024", help="Input size as H,W")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Parse input size
    h, w = map(int, args.input_size.split(','))
    input_size = (1, 3, h, w)
    print(f"Input size: {input_size}")
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info["num_classes"]
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print(f"Pruning method: {args.method}")
    print(f"Global pruning ratio: {args.global_ratio}")
    
    # Build and load model
    print(f"Building model: {args.model}")
    model = build_model(args.model, num_classes).to(device)
    
    print(f"Loading checkpoint: {args.ckpt}")
    load_checkpoint(model, args.ckpt)
    
    # Count original parameters
    original_params = sum(p.numel() for p in model.parameters())
    original_conv_channels = sum(m.out_channels for m in model.modules() if isinstance(m, torch.nn.Conv2d))
    
    print(f"Original parameters: {original_params:,}")
    print(f"Original conv channels: {original_conv_channels}")
    
    # Create pruning config
    config = PruningConfig(
        global_ratio=args.global_ratio,
        device=device,
        input_size=input_size,
        fgp_samples=64 if args.method == "fgp" else 32,
        sirfp_samples=32 if args.method == "sirfp" else 16,
        sirfp_corr_threshold=0.8
    )
    
    # Build pruner
    print("Building pruner...")
    pruner = build_pruner(args.method, model, config)
    
    # Create example input
    example_inputs = torch.randn(*input_size).to(device)
    
    # Perform pruning
    print("Performing pruning...")
    try:
        pruned_model = pruner.prune(example_inputs)
        print("Pruning completed successfully!")
    except Exception as e:
        print(f"Pruning failed: {e}")
        print("This might be due to model complexity or pruning method compatibility.")
        print("Try using a different pruning method or reducing the pruning ratio.")
        return
    
    # Count pruned parameters
    pruned_params = sum(p.numel() for p in pruned_model.parameters())
    pruned_conv_channels = sum(m.out_channels for m in pruned_model.modules() if isinstance(m, torch.nn.Conv2d))
    
    param_reduction = (original_params - pruned_params) / original_params
    channel_reduction = (original_conv_channels - pruned_conv_channels) / original_conv_channels
    
    print(f"\nPruning Results:")
    print(f"Parameters: {original_params:,} → {pruned_params:,} ({param_reduction:.1%} reduction)")
    print(f"Conv channels: {original_conv_channels} → {pruned_conv_channels} ({channel_reduction:.1%} reduction)")
    
    # Save pruned model
    os.makedirs(args.save, exist_ok=True)
    save_path = os.path.join(args.save, "pruned_model.pt")
    
    # Save as model object to preserve architecture changes
    torch.save({"model_obj": pruned_model}, save_path)
    print(f"Pruned model saved to: {save_path}")
    
    # Save pruning info
    pruning_info = {
        "original_checkpoint": args.ckpt,
        "pruning_method": args.method,
        "global_ratio": args.global_ratio,
        "dataset": args.dataset,
        "model_type": args.model,
        "input_size": input_size,
        "original_params": original_params,
        "pruned_params": pruned_params,
        "param_reduction": param_reduction,
        "original_conv_channels": original_conv_channels,
        "pruned_conv_channels": pruned_conv_channels,
        "channel_reduction": channel_reduction
    }
    
    import json
    info_path = os.path.join(args.save, "pruning_info.json")
    with open(info_path, "w") as f:
        json.dump(pruning_info, f, indent=2)
    print(f"Pruning info saved to: {info_path}")


if __name__ == "__main__":
    main()
