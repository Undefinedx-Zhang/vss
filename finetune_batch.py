"""
Batch fine-tuning script for pruned segmentation models
Automatically fine-tunes all pruned models in a benchmark directory and re-evaluates them
"""
import os
import json
import argparse
import glob
import torch
from pathlib import Path

from bench.data import get_loaders, get_dataset_info
from bench.models import build_model
from bench.train_eval import fit_seg
from bench.metrics import (
    evaluate_segmentation, speed_benchmark_seg, flops_params_seg,
    oracle_channel_importance_seg, reconstruction_error_seg,
    compute_seg_prune_ratio, save_seg_report
)


def load_pruned_model(model_path, model_type, num_classes, device):
    """Load pruned model from checkpoint"""
    print(f"Loading pruned model from: {model_path}")
    state = torch.load(model_path, map_location="cpu")
    
    if isinstance(state, dict) and "model_obj" in state:
        # Direct model object (preserves pruned architecture)
        model = state["model_obj"].to(device)
        print("Loaded model object with pruned architecture")
    elif isinstance(state, dict) and "model" in state:
        # State dict - need to build base model first
        model = build_model(model_type, num_classes).to(device)
        model.load_state_dict(state["model"], strict=False)
        print("Loaded state dict into base model")
    else:
        # Direct state dict
        model = build_model(model_type, num_classes).to(device)
        model.load_state_dict(state, strict=False)
        print("Loaded direct state dict")
    
    return model


def finetune_model(model, train_loader, val_loader, epochs, lr, weight_decay, 
                   device, save_dir, num_classes, ignore_index, scheduler_type="poly"):
    """Fine-tune a pruned model"""
    print(f"Fine-tuning for {epochs} epochs...")
    
    best_miou, train_history, val_history = fit_seg(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        device=device,
        save_dir=save_dir,
        num_classes=num_classes,
        ignore_index=ignore_index,
        scheduler_type=scheduler_type
    )
    
    print(f"Fine-tuning completed! Best mIoU: {best_miou:.4f}")
    return best_miou, train_history, val_history


def evaluate_model(model, baseline_model, val_loader, device, num_classes, ignore_index, 
                   input_size, oracle_samples, report_path):
    """Evaluate a fine-tuned model"""
    print("Evaluating fine-tuned model...")
    
    # Evaluate baseline model
    baseline_results = evaluate_segmentation(baseline_model, val_loader, device, num_classes, ignore_index)
    baseline_miou = baseline_results['mIoU']
    
    # Evaluate candidate model
    candidate_results = evaluate_segmentation(model, val_loader, device, num_classes, ignore_index)
    candidate_miou = candidate_results['mIoU']
    
    # Compute performance retention
    miou_retention = candidate_miou / max(1e-8, baseline_miou)
    pixel_acc_retention = candidate_results['Pixel_Accuracy'] / max(1e-8, baseline_results['Pixel_Accuracy'])
    
    print(f"Baseline mIoU: {baseline_miou:.4f}")
    print(f"Fine-tuned mIoU: {candidate_miou:.4f}")
    print(f"mIoU retention: {miou_retention:.4f} ({miou_retention*100:.1f}%)")
    
    # Speed benchmark
    baseline_speed = speed_benchmark_seg(baseline_model, input_size, device)
    candidate_speed = speed_benchmark_seg(model, input_size, device)
    
    speedup_ms = baseline_speed["avg_ms_per_iter"] / candidate_speed["avg_ms_per_iter"]
    speedup_fps = candidate_speed["fps"] / baseline_speed["fps"]
    
    # FLOPs and parameters
    baseline_fp = flops_params_seg(baseline_model, input_size)
    candidate_fp = flops_params_seg(model, input_size)
    
    # Pruning ratios
    prune_ratios = compute_seg_prune_ratio(baseline_model, model)
    
    # Oracle channel importance analysis (simplified)
    oracle_results = {}
    try:
        target_layer = None
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Conv2d) and module.out_channels > 1:
                target_layer = module
                layer_name = name
                break
        
        if target_layer is not None:
            oracle_scores = oracle_channel_importance_seg(
                model, target_layer, val_loader, device, 
                num_classes, ignore_index, oracle_samples
            )
            
            method_scores = torch.norm(target_layer.weight.detach().view(target_layer.out_channels, -1), p=1, dim=1)
            k = max(1, target_layer.out_channels // 5)
            method_topk = torch.topk(method_scores, k, largest=False)[1]
            oracle_topk = torch.topk(oracle_scores, k, largest=False)[1]
            
            overlap = len(set(method_topk.tolist()).intersection(set(oracle_topk.tolist())))
            redundancy_acc = overlap / k
            
            oracle_results = {
                "analyzed_layer": layer_name,
                "layer_channels": target_layer.out_channels,
                "redundancy_identification_accuracy": redundancy_acc,
                "oracle_samples_used": oracle_samples
            }
    except Exception as e:
        print(f"Oracle analysis failed: {e}")
    
    # Reconstruction error
    try:
        reconstruction_mse = reconstruction_error_seg(baseline_model, model, val_loader, device, max_batches=3)
    except Exception as e:
        print(f"Reconstruction error computation failed: {e}")
        reconstruction_mse = float('nan')
    
    # Compile report
    report = {
        "baseline_results": baseline_results,
        "candidate_results": candidate_results,
        "miou_retention": miou_retention,
        "pixel_accuracy_retention": pixel_acc_retention,
        
        "baseline_speed": baseline_speed,
        "candidate_speed": candidate_speed,
        "speedup_latency": speedup_ms,
        "speedup_fps": speedup_fps,
        
        "baseline_flops_params": baseline_fp,
        "candidate_flops_params": candidate_fp,
        "pruning_ratios": prune_ratios,
        
        "oracle_analysis": oracle_results,
        "reconstruction_mse": reconstruction_mse,
    }
    
    # Save report
    save_seg_report(report_path, report)
    print(f"Evaluation report saved to: {report_path}")
    
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Batch fine-tune pruned segmentation models")
    p.add_argument("--benchmark-dir", type=str, default="./runs/camvid_resnet50_benchmark", 
                   help="Directory containing pruned models (e.g., ./runs/camvid_resnet50_benchmark)")
    p.add_argument("--dataset", type=str, default="camvid", choices=["cityscapes", "camvid"])
    p.add_argument("--data-root", type=str, default="./datasets/camvid")
    p.add_argument("--model", type=str, default="resnet50_seg",
                   choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101",
                           "resnet50_seg", "resnet101_seg", "simple_segnet", "unet"])
    p.add_argument("--baseline", type=str, default="./runs/camvid_resnet50_seg/best.pt", help="Baseline model checkpoint")
    
    # Fine-tuning parameters
    p.add_argument("--finetune-epochs", type=int, default=20, help="Number of fine-tuning epochs")
    p.add_argument("--finetune-lr", type=float, default=0.001, help="Fine-tuning learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scheduler", type=str, default="poly", choices=["poly", "multistep"])
    
    # Evaluation parameters
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-size", type=str, default="360,480", help="Input size as H,W")
    p.add_argument("--oracle-samples", type=int, default=16)
    
    # Control options
    p.add_argument("--skip-existing", action="store_true", help="Skip if fine-tuned model already exists")
    p.add_argument("--methods", type=str, default="", 
                   help="Comma-separated methods to process (empty = all)")
    p.add_argument("--ratios", type=str, default="", 
                   help="Comma-separated ratios to process (empty = all)")
    
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Parse input size
    h, w = map(int, args.input_size.split(','))
    input_size = (1, 3, h, w)
    
    # Get dataset info
    dataset_info = get_dataset_info(args.dataset)
    num_classes = dataset_info["num_classes"]
    ignore_index = dataset_info["ignore_index"]
    
    print(f"Dataset: {args.dataset}")
    print(f"Number of classes: {num_classes}")
    print(f"Benchmark directory: {args.benchmark_dir}")
    
    # Load data
    train_loader, val_loader = get_loaders(args.dataset, args.data_root, args.batch_size, args.num_workers)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Load baseline model for evaluation comparison
    print("Loading baseline model...")
    baseline_model = build_model(args.model, num_classes)
    baseline_state = torch.load(args.baseline, map_location="cpu")
    if "model" in baseline_state:
        baseline_state = baseline_state["model"]
    baseline_model.load_state_dict(baseline_state, strict=False)
    baseline_model = baseline_model.to(device)
    
    # Find all experiment directories
    experiment_dirs = []
    for item in os.listdir(args.benchmark_dir):
        item_path = os.path.join(args.benchmark_dir, item)
        if os.path.isdir(item_path):
            pruned_model_path = os.path.join(item_path, "pruned_model.pt")
            if os.path.exists(pruned_model_path):
                experiment_dirs.append(item_path)
    
    print(f"Found {len(experiment_dirs)} experiment directories with pruned models")
    
    # Filter by methods and ratios if specified
    if args.methods:
        target_methods = [m.strip().lower() for m in args.methods.split(',')]
        experiment_dirs = [d for d in experiment_dirs 
                          if any(method in os.path.basename(d).lower() for method in target_methods)]
    
    if args.ratios:
        target_ratios = [r.strip() for r in args.ratios.split(',')]
        experiment_dirs = [d for d in experiment_dirs 
                          if any(ratio in os.path.basename(d) for ratio in target_ratios)]
    
    print(f"Processing {len(experiment_dirs)} experiments after filtering")
    
    # Process each experiment
    results_summary = []
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        exp_name = os.path.basename(exp_dir)
        print(f"\n{'='*80}")
        print(f"Processing {exp_name} ({i}/{len(experiment_dirs)})")
        print(f"{'='*80}")
        
        pruned_model_path = os.path.join(exp_dir, "pruned_model.pt")
        finetuned_model_path = os.path.join(exp_dir, "finetuned_model.pt")
        finetuned_report_path = os.path.join(exp_dir, "finetuned_report.json")
        
        # Skip if fine-tuned model already exists
        if args.skip_existing and os.path.exists(finetuned_model_path):
            print(f"⏭️  Skipping {exp_name} - fine-tuned model already exists")
            continue
        
        try:
            # Load pruned model
            model = load_pruned_model(pruned_model_path, args.model, num_classes, device)
            
            # Fine-tune the model
            finetune_save_dir = os.path.join(exp_dir, "finetune")
            os.makedirs(finetune_save_dir, exist_ok=True)
            
            best_miou, train_history, val_history = finetune_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.finetune_epochs,
                lr=args.finetune_lr,
                weight_decay=args.weight_decay,
                device=device,
                save_dir=finetune_save_dir,
                num_classes=num_classes,
                ignore_index=ignore_index,
                scheduler_type=args.scheduler
            )
            
            # Load the best fine-tuned model
            best_model_path = os.path.join(finetune_save_dir, "best.pt")
            if os.path.exists(best_model_path):
                best_state = torch.load(best_model_path, map_location="cpu")
                if "model" in best_state:
                    model.load_state_dict(best_state["model"])
                else:
                    model.load_state_dict(best_state)
                print("Loaded best fine-tuned model for evaluation")
            
            # Save fine-tuned model
            torch.save({"model_obj": model}, finetuned_model_path)
            print(f"Fine-tuned model saved to: {finetuned_model_path}")
            
            # Evaluate fine-tuned model
            report = evaluate_model(
                model=model,
                baseline_model=baseline_model,
                val_loader=val_loader,
                device=device,
                num_classes=num_classes,
                ignore_index=ignore_index,
                input_size=input_size,
                oracle_samples=args.oracle_samples,
                report_path=finetuned_report_path
            )
            
            # Add experiment info to report
            report.update({
                "experiment_name": exp_name,
                "finetune_epochs": args.finetune_epochs,
                "finetune_lr": args.finetune_lr,
                "best_finetune_miou": best_miou
            })
            
            results_summary.append(report)
            print(f"✅ {exp_name} completed successfully")
            
        except Exception as e:
            print(f"❌ {exp_name} failed: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comprehensive summary
    summary_path = os.path.join(args.benchmark_dir, "finetuned_summary.json")
    summary = {
        "finetune_config": {
            "dataset": args.dataset,
            "model": args.model,
            "baseline": args.baseline,
            "finetune_epochs": args.finetune_epochs,
            "finetune_lr": args.finetune_lr,
            "weight_decay": args.weight_decay,
            "scheduler": args.scheduler,
            "batch_size": args.batch_size,
            "input_size": args.input_size
        },
        "results": results_summary
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*80}")
    print("BATCH FINE-TUNING COMPLETED")
    print(f"{'='*80}")
    print(f"Total experiments: {len(experiment_dirs)}")
    print(f"Successfully processed: {len(results_summary)}")
    print(f"Failed: {len(experiment_dirs) - len(results_summary)}")
    print(f"Summary saved to: {summary_path}")
    
    # Print results table
    if results_summary:
        print(f"\n{'='*120}")
        print("FINE-TUNED RESULTS SUMMARY")
        print(f"{'='*120}")
        print(f"{'Experiment':<25} {'mIoU':<8} {'Retention':<10} {'Speedup':<8} {'Ch.Red.':<8} {'FLOPs':<15} {'Params':<15}")
        print(f"{'-'*120}")
        
        for result in results_summary:
            exp_name = result['experiment_name']
            miou = result['candidate_results']['mIoU']
            retention = result['miou_retention']
            speedup = result['speedup_latency']
            ch_red = result['pruning_ratios']['channel_prune_ratio']
            flops = result['candidate_flops_params']['FLOPs']
            params = result['candidate_flops_params']['Params']
            
            print(f"{exp_name:<25} {miou:<8.3f} {retention:<10.3f} {speedup:<8.2f} {ch_red:<8.1%} {flops:<15} {params:<15}")
        
        print(f"{'-'*120}")
    
    print(f"\nBatch fine-tuning completed! All results saved to: {args.benchmark_dir}")


if __name__ == "__main__":
    main()
