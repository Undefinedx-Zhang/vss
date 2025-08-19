"""
Evaluation script for segmentation models
"""
import os
import json
import argparse
import torch

from bench.data import get_loaders, get_dataset_info
from bench.models import build_model
from bench.metrics import (
    evaluate_segmentation, speed_benchmark_seg, flops_params_seg,
    oracle_channel_importance_seg, reconstruction_error_seg,
    compute_seg_prune_ratio, save_seg_report
)
os.environ['CUDA_VISIBLE_DEVICES'] = '4'

def load_state_or_object(model, path, num_classes):
    """Load model state or object"""
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model_obj" in state:
        return state["model_obj"]
    elif isinstance(state, dict) and "model" in state:
        model.load_state_dict(state["model"], strict=False)
        return model
    else:
        model.load_state_dict(state, strict=False)
        return model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate segmentation models")
    p.add_argument("--dataset", type=str, default="cityscapes", choices=["cityscapes", "camvid"])
    p.add_argument("--data-root", type=str, required=True)
    p.add_argument("--model", type=str, default="deeplabv3_resnet50",
                   choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101",
                           "resnet50_seg", "resnet101_seg", "simple_segnet", "unet"])
    p.add_argument("--baseline", type=str, required=True, help="Baseline model checkpoint")
    p.add_argument("--candidate", type=str, required=True, help="Candidate (pruned) model checkpoint")
    p.add_argument("--report", type=str, default="./seg_eval_report.json")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-size", type=str, default="512,1024", help="Input size as H,W")
    p.add_argument("--oracle-samples", type=int, default=16, help="Samples for oracle analysis")
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
    ignore_index = dataset_info["ignore_index"]
    class_names = dataset_info["classes"]
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    
    # Load data
    print("Loading validation data...")
    _, val_loader = get_loaders(args.dataset, args.data_root, args.batch_size, args.num_workers)
    print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Load models
    print("Loading baseline model...")
    baseline_model = build_model(args.model, num_classes)
    baseline_model = load_state_or_object(baseline_model, args.baseline, num_classes).to(device)
    
    print("Loading candidate model...")
    candidate_model = build_model(args.model, num_classes)
    candidate_model = load_state_or_object(candidate_model, args.candidate, num_classes).to(device)
    
    print("Models loaded successfully!")
    
    # Evaluate baseline model
    print("\nEvaluating baseline model...")
    baseline_results = evaluate_segmentation(baseline_model, val_loader, device, num_classes, ignore_index)
    baseline_miou = baseline_results['mIoU']
    print(f"Baseline mIoU: {baseline_miou:.4f}")
    print(f"Baseline Pixel Accuracy: {baseline_results['Pixel_Accuracy']:.4f}")
    
    # Evaluate candidate model
    print("\nEvaluating candidate model...")
    candidate_results = evaluate_segmentation(candidate_model, val_loader, device, num_classes, ignore_index)
    candidate_miou = candidate_results['mIoU']
    print(f"Candidate mIoU: {candidate_miou:.4f}")
    print(f"Candidate Pixel Accuracy: {candidate_results['Pixel_Accuracy']:.4f}")
    
    # Compute performance retention
    miou_retention = candidate_miou / max(1e-8, baseline_miou)
    pixel_acc_retention = candidate_results['Pixel_Accuracy'] / max(1e-8, baseline_results['Pixel_Accuracy'])
    
    print(f"\nPerformance Retention:")
    print(f"mIoU retention: {miou_retention:.4f} ({miou_retention*100:.1f}%)")
    print(f"Pixel Acc retention: {pixel_acc_retention:.4f} ({pixel_acc_retention*100:.1f}%)")
    
    # Speed benchmark
    print("\nRunning speed benchmark...")
    baseline_speed = speed_benchmark_seg(baseline_model, input_size, device)
    candidate_speed = speed_benchmark_seg(candidate_model, input_size, device)
    
    speedup_ms = baseline_speed["avg_ms_per_iter"] / candidate_speed["avg_ms_per_iter"]
    speedup_fps = candidate_speed["fps"] / baseline_speed["fps"]
    
    print(f"Baseline speed: {baseline_speed['avg_ms_per_iter']:.2f} ms/iter, {baseline_speed['fps']:.2f} FPS")
    print(f"Candidate speed: {candidate_speed['avg_ms_per_iter']:.2f} ms/iter, {candidate_speed['fps']:.2f} FPS")
    print(f"Speedup: {speedup_ms:.2f}x (latency), {speedup_fps:.2f}x (FPS)")
    
    # FLOPs and parameters
    print("\nComputing FLOPs and parameters...")
    baseline_fp = flops_params_seg(baseline_model, input_size)
    candidate_fp = flops_params_seg(candidate_model, input_size)
    
    print(f"Baseline - FLOPs: {baseline_fp['FLOPs']}, Params: {baseline_fp['Params']}")
    print(f"Candidate - FLOPs: {candidate_fp['FLOPs']}, Params: {candidate_fp['Params']}")
    
    # Pruning ratios
    print("\nComputing pruning ratios...")
    prune_ratios = compute_seg_prune_ratio(baseline_model, candidate_model)
    print(f"Channel pruning ratio: {prune_ratios['channel_prune_ratio']:.4f} ({prune_ratios['channel_prune_ratio']*100:.1f}%)")
    print(f"Channels: {prune_ratios['channels_before']} → {prune_ratios['channels_after']}")
    
    # Oracle channel importance analysis (on a subset)
    print("\nComputing oracle channel importance...")
    oracle_results = {}
    try:
        # Find a representative Conv2d layer for analysis
        target_layer = None
        for name, module in candidate_model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.out_channels > 1:
                target_layer = module
                layer_name = name
                break
        
        if target_layer is not None:
            print(f"Analyzing layer: {layer_name} ({target_layer.out_channels} channels)")
            oracle_scores = oracle_channel_importance_seg(
                candidate_model, target_layer, val_loader, device, 
                num_classes, ignore_index, args.oracle_samples
            )
            
            # Compute redundancy identification accuracy (simplified)
            # Use L1 norm as method importance scores
            method_scores = torch.norm(target_layer.weight.detach().view(target_layer.out_channels, -1), p=1, dim=1)
            
            # Top-K overlap analysis
            k = max(1, target_layer.out_channels // 5)  # Top 20%
            method_topk = torch.topk(method_scores, k, largest=False)[1]  # Least important by method
            oracle_topk = torch.topk(oracle_scores, k, largest=False)[1]  # Least important by oracle
            
            overlap = len(set(method_topk.tolist()).intersection(set(oracle_topk.tolist())))
            redundancy_acc = overlap / k
            
            oracle_results = {
                "analyzed_layer": layer_name,
                "layer_channels": target_layer.out_channels,
                "redundancy_identification_accuracy": redundancy_acc,
                "oracle_samples_used": args.oracle_samples
            }
            print(f"Redundancy identification accuracy: {redundancy_acc:.4f}")
        else:
            print("No suitable Conv2d layer found for oracle analysis")
    except Exception as e:
        print(f"Oracle analysis failed: {e}")
    
    # Reconstruction error
    print("\nComputing reconstruction error...")
    try:
        reconstruction_mse = reconstruction_error_seg(baseline_model, candidate_model, val_loader, device, max_batches=3)
        print(f"Reconstruction MSE: {reconstruction_mse:.6f}")
    except Exception as e:
        print(f"Reconstruction error computation failed: {e}")
        reconstruction_mse = float('nan')
    
    # Compile comprehensive report
    report = {
        "dataset": args.dataset,
        "model_type": args.model,
        "input_size": input_size,
        "num_classes": num_classes,
        "ignore_index": ignore_index,
        
        # Performance metrics
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "miou_retention": miou_retention,
        "pixel_accuracy_retention": pixel_acc_retention,
        
        # Speed metrics
        "baseline_speed": baseline_speed,
        "candidate_speed": candidate_speed,
        "speedup_latency": speedup_ms,
        "speedup_fps": speedup_fps,
        
        # Model complexity
        "baseline_flops_params": baseline_fp,
        "candidate_flops_params": candidate_fp,
        "pruning_ratios": prune_ratios,
        
        # Advanced metrics
        "oracle_analysis": oracle_results,
        "reconstruction_mse": reconstruction_mse,
        
        # Configuration
        "evaluation_config": {
            "batch_size": args.batch_size,
            "oracle_samples": args.oracle_samples,
            "baseline_checkpoint": args.baseline,
            "candidate_checkpoint": args.candidate
        }
    }
    
    # Print summary
    print("\n" + "="*60)
    print("SEGMENTATION EVALUATION SUMMARY")
    print("="*60)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model}")
    print(f"mIoU: {baseline_miou:.4f} → {candidate_miou:.4f} (retention: {miou_retention:.1%})")
    print(f"Pixel Acc: {baseline_results['Pixel_Accuracy']:.4f} → {candidate_results['Pixel_Accuracy']:.4f} (retention: {pixel_acc_retention:.1%})")
    print(f"Speedup: {speedup_ms:.2f}x (latency), {speedup_fps:.2f}x (FPS)")
    print(f"Channel reduction: {prune_ratios['channel_prune_ratio']:.1%}")
    print(f"FLOPs: {baseline_fp['FLOPs']} → {candidate_fp['FLOPs']}")
    print(f"Params: {baseline_fp['Params']} → {candidate_fp['Params']}")
    if oracle_results:
        print(f"Redundancy ID accuracy: {oracle_results['redundancy_identification_accuracy']:.3f}")
    print(f"Reconstruction MSE: {reconstruction_mse:.6f}")
    print("="*60)
    
    # Save report
    save_seg_report(args.report, report)
    print(f"\nDetailed report saved to: {args.report}")


if __name__ == "__main__":
    main()
