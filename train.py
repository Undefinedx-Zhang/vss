"""
Training script for segmentation models
"""
import os
import argparse
import torch
import sys

# 设置可见的CUDA设备编号为4、5
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
from bench.data import get_loaders, get_dataset_info
from bench.models import build_model
from bench.train_eval import fit_seg, compute_class_weights

# 设置PyTorch缓存目录到当前脚本路径
current_dir = os.path.dirname(os.path.abspath(__file__))
cache_dir = os.path.join(current_dir, 'model_cache')
os.makedirs(cache_dir, exist_ok=True)
os.environ['TORCH_HOME'] = cache_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train segmentation model")
    p.add_argument("--dataset", type=str, default="camvid", choices=["cityscapes", "camvid"])
    p.add_argument("--data-root", type=str, default="./datasets/camvid", help="Path to dataset root")
    p.add_argument("--model", type=str, default="resnet101_seg", 
                   choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101", 
                           "resnet50_seg", "resnet101_seg", "simple_segnet", "unet"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="poly", choices=["poly", "multistep"])
    p.add_argument("--save", type=str, default="./runs/camvid_resnet101_benchmark")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--compute-class-weights", action="store_true", help="Compute class weights for balanced training")
    p.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info["num_classes"]
    ignore_index = dataset_info["ignore_index"]
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print(f"Ignore index: {ignore_index}")
    
    # Load data
    print("Loading data...")
    train_loader, val_loader = get_loaders(
        args.dataset, args.data_root, args.batch_size, args.num_workers
    )
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Build model
    print(f"Building model: {args.model}")
    model = build_model(args.model, num_classes).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
    
    # Compute class weights if requested
    class_weights = None
    if args.compute_class_weights:
        class_weights = compute_class_weights(train_loader, num_classes, ignore_index)
        class_weights = class_weights.to(device)
        print("Using computed class weights for balanced training")
    
    # Train model
    print("Starting training...")
    best_miou, train_history, val_history = fit_seg(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        save_dir=args.save,
        num_classes=num_classes,
        ignore_index=ignore_index,
        scheduler_type=args.scheduler
    )
    
    print(f"\nTraining completed!")
    print(f"Best mIoU: {best_miou:.4f}")
    print(f"Model saved to: {args.save}")
    
    # Save training configuration
    config = {
        "dataset": args.dataset,
        "model": args.model,
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "scheduler": args.scheduler,
        "best_miou": best_miou,
        "total_params": total_params,
        "trainable_params": trainable_params
    }
    
    import json
    with open(os.path.join(args.save, "config.json"), "w") as f:
        json.dump(config, f, indent=2)


if __name__ == "__main__":
    main()
