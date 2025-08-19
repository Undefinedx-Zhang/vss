"""
Comprehensive benchmark for segmentation pruning methods
"""
import os
import json
import argparse
import subprocess
import sys
from typing import List
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark segmentation pruning methods")
    p.add_argument("--dataset", type=str, default="camvid", choices=["cityscapes", "camvid"])
    p.add_argument("--data-root", type=str, default="./datasets/camvid", help="Path to dataset root")
    p.add_argument("--model", type=str, default="resnet101_seg",
                   choices=["deeplabv3_resnet50", "deeplabv3_resnet101", "fcn_resnet50", "fcn_resnet101",
                           "resnet50_seg", "resnet101_seg", "simple_segnet", "unet"])
    p.add_argument("--baseline", type=str, default="./runs/camvid_resnet101/best.pt", help="Baseline model checkpoint")
    p.add_argument("--methods", type=str, default="random, taylor, slimming, fpgm, fgp, sirfp",choices=["taylor","slimming","fpgm","random","fgp","sirfp","dmcp"],
                   help="Comma-separated pruning methods")
    p.add_argument("--ratios", type=str, default="0.01, 0.02, 0.03, 0.04, 0.05,0.06,0.07,0.08,0.09,0.1",
                   help="Comma-separated pruning ratios")
    p.add_argument("--out-dir", type=str, default="./runs/camvid_resnet101_benchmark")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--input-size", type=str, default="360,480", help="Input size as H,W")
    p.add_argument("--oracle-samples", type=int, default=64)
    p.add_argument("--skip-existing", action="store_true", help="Skip existing results")
    return p.parse_args()


def run_pruning(dataset: str, data_root: str, model: str, baseline: str, method: str, 
               ratio: float, save_dir: str, input_size: str) -> bool:
    """Run pruning for a specific method and ratio"""
    cmd = [
        "python", "-u", "prune.py",
        "--dataset", dataset,
        "--data-root", data_root,
        "--model", model,
        "--ckpt", baseline,
        "--method", method,
        "--global-ratio", str(ratio),
        "--save", save_dir,
        "--input-size", input_size
    ]
    
    # Stream logs in real-time
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        sys.stdout.flush()
    ret = proc.wait()
    if ret == 0:
        print(f"✓ Pruning {method} {ratio:.1%} completed")
        return True
    else:
        print(f"✗ Pruning {method} {ratio:.1%} failed (return code {ret})")
        return False


def run_evaluation(dataset: str, data_root: str, model: str, baseline: str, 
                  candidate: str, report_path: str, batch_size: int, 
                  num_workers: int, input_size: str, oracle_samples: int) -> bool:
    """Run evaluation for a pruned model"""
    cmd = [
        "python", "-u", "eval.py",
        "--dataset", dataset,
        "--data-root", data_root,
        "--model", model,
        "--baseline", baseline,
        "--candidate", candidate,
        "--report", report_path,
        "--batch-size", str(batch_size),
        "--num-workers", str(num_workers),
        "--input-size", input_size,
        "--oracle-samples", str(oracle_samples)
    ]
    
    # Stream logs in real-time
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        sys.stdout.flush()
    ret = proc.wait()
    if ret == 0:
        print(f"✓ Evaluation completed")
        return True
    else:
        print(f"✗ Evaluation failed (return code {ret})")
        return False


def main():
    args = parse_args()
    
    # Parse methods and ratios
    methods = [m.strip() for m in args.methods.split(',') if m.strip()]
    ratios = [float(r.strip()) for r in args.ratios.split(',')]
    
    print(f"Benchmarking {len(methods)} methods with {len(ratios)} ratios")
    print(f"Methods: {methods}")
    print(f"Ratios: {ratios}")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Baseline: {args.baseline}")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    results = []
    total_experiments = len(methods) * len(ratios)
    completed = 0
    failed = 0
    
    for method in methods:
        for ratio in ratios:
            experiment_name = f"{method}_{ratio:.2f}".replace('.', '_')
            experiment_dir = os.path.join(args.out_dir, experiment_name)
            
            print(f"\n{'='*60}")
            print(f"Experiment: {method.upper()} @ {ratio:.1%} ({completed+1}/{total_experiments})")
            print(f"{'='*60}")
            
            # Check if results already exist
            report_path = os.path.join(experiment_dir, "report.json")
            if args.skip_existing and os.path.exists(report_path):
                print(f"⏭️  Skipping existing result: {experiment_name}")
                try:
                    with open(report_path, 'r') as f:
                        result = json.load(f)
                        result['experiment_name'] = experiment_name
                        result['method'] = method
                        result['ratio'] = ratio
                        results.append(result)
                    completed += 1
                    continue
                except Exception as e:
                    print(f"Failed to load existing result: {e}")
            
            os.makedirs(experiment_dir, exist_ok=True)
            
            # Step 1: Pruning
            print(f"Step 1/2: Pruning with {method}...")
            pruning_success = run_pruning(
                args.dataset, args.data_root, args.model, args.baseline,
                method, ratio, experiment_dir, args.input_size
            )
            
            if not pruning_success:
                print(f"❌ Experiment {experiment_name} failed at pruning stage")
                failed += 1
                continue
            
            # Step 2: Evaluation
            print(f"Step 2/2: Evaluating pruned model...")
            pruned_model_path = os.path.join(experiment_dir, "pruned_model.pt")
            
            if not os.path.exists(pruned_model_path):
                print(f"❌ Pruned model not found: {pruned_model_path}")
                failed += 1
                continue
            
            eval_success = run_evaluation(
                args.dataset, args.data_root, args.model, args.baseline,
                pruned_model_path, report_path, args.batch_size,
                args.num_workers, args.input_size, args.oracle_samples
            )
            
            if not eval_success:
                print(f"❌ Experiment {experiment_name} failed at evaluation stage")
                failed += 1
                continue
            
            # Load and store results
            try:
                with open(report_path, 'r') as f:
                    result = json.load(f)
                    result['experiment_name'] = experiment_name
                    result['method'] = method
                    result['ratio'] = ratio
                    results.append(result)
                
                print(f"✅ Experiment {experiment_name} completed successfully")
                completed += 1
                
            except Exception as e:
                print(f"❌ Failed to load results for {experiment_name}: {e}")
                failed += 1
    
    print(f"\n{'='*60}")
    print(f"BENCHMARK COMPLETED")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {completed/total_experiments*100:.1f}%")
    
    # Save comprehensive summary
    summary = {
        "benchmark_config": {
            "dataset": args.dataset,
            "model": args.model,
            "baseline": args.baseline,
            "methods": methods,
            "ratios": ratios,
            "input_size": args.input_size,
            "batch_size": args.batch_size,
            "oracle_samples": args.oracle_samples
        },
        "summary_stats": {
            "total_experiments": total_experiments,
            "completed": completed,
            "failed": failed,
            "success_rate": completed/total_experiments if total_experiments > 0 else 0
        },
        "results": results
    }
    
    summary_path = os.path.join(args.out_dir, "benchmark_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    # Create results table
    if results:
        print(f"\n{'='*120}")
        print("RESULTS SUMMARY")
        print(f"{'='*120}")
        print(f"{'Method':<12} {'Ratio':<8} {'mIoU':<8} {'Retention':<10} {'Speedup':<8} {'Ch.Red.':<8} {'FLOPs':<15} {'Params':<15}")
        print(f"{'-'*120}")
        
        for result in results:
            method = result['method']
            ratio = result['ratio']
            miou = result['candidate_results']['mIoU']
            retention = result['miou_retention']
            speedup = result['speedup_latency']
            ch_red = result['pruning_ratios']['channel_prune_ratio']
            flops = result['candidate_flops_params']['FLOPs']
            params = result['candidate_flops_params']['Params']
            
            print(f"{method:<12} {ratio:<8.1%} {miou:<8.3f} {retention:<10.3f} {speedup:<8.2f} {ch_red:<8.1%} {flops:<15} {params:<15}")
        
        print(f"{'-'*120}")
    
    print(f"\nBenchmark completed! Results saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
