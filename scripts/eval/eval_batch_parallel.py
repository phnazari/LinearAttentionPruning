import os
import sys
import argparse
import json
import glob
import subprocess
import time
import ray

# Prevent Ray from killing workers that momentarily spike in memory usage
os.environ["RAY_memory_monitor_refresh_ms"] = "0"

@ray.remote(num_gpus=1)
def run_evaluation_task(model_name, checkpoint_dir, config_path, args, output_file):
    cmd = [
        sys.executable, "-m", "flame.eval_checkpoint",
        "--tasks", args.tasks,
        "--batch_size", str(args.batch_size),
        "--device", "cuda",
        "--dtype", args.dtype,
        "--output_path", output_file
    ]

    # Pass through debugging environment variables if set in the parent process
    env = os.environ.copy()
    if "CUDA_LAUNCH_BLOCKING" in os.environ:
        env["CUDA_LAUNCH_BLOCKING"] = os.environ["CUDA_LAUNCH_BLOCKING"]
    if "TORCH_USE_CUDA_DSA" in os.environ:
        env["TORCH_USE_CUDA_DSA"] = os.environ["TORCH_USE_CUDA_DSA"]

    if args.max_length:
        cmd.extend(["--max_length", str(args.max_length)])
    
    is_hf = False
    if os.path.exists(os.path.join(checkpoint_dir, "config.json")):
        if os.path.exists(os.path.join(checkpoint_dir, "pytorch_model.bin")) or \
           os.path.exists(os.path.join(checkpoint_dir, "model.safetensors")) or \
           os.path.exists(os.path.join(checkpoint_dir, "model.safetensors.index.json")):
            is_hf = True

            
    if is_hf:
        cmd.extend(["--pretrained", checkpoint_dir])
    else:
        cmd.extend([
            "--checkpoint_dir", checkpoint_dir,
            "--step", str(args.step),
            "--config", config_path,
            "--tokenizer", args.tokenizer
        ])

    start_time = time.time()
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        duration = time.time() - start_time
        return {
            "name": model_name,
            "status": "success",
            "duration_idx": f"{duration:.1f}s",
            "output": output_file
        }
    except subprocess.CalledProcessError as e:
        return {
            "name": model_name,
            "status": "failed",
            "error": e.stderr
        }

def find_models(search_dir, model_prefix, method):
    """
    Search for models inside the provided compressed_base directory.
    Uses model_prefix to construct consistent names.
    """
    models = []
    
    if not os.path.exists(search_dir):
        print(f"⚠️  Search directory not found: {search_dir}", flush=True)
        return models

    # Helper to clean up the name construction
    def get_name(path):
        # Result: prefix_directoryname (e.g., delta_net_compressed_100)
        return f"{model_prefix}_{os.path.basename(path)}"

    # 1. Find compressed_* directories
    for p in glob.glob(os.path.join(search_dir, f"{method}_compressed_*trashhh")):
        if os.path.isdir(p) and os.path.exists(os.path.join(p, "config.json")):
            models.append({
                "name": get_name(p),
                "dir": p,
                "config": os.path.join(p, "config.json")
            })

    # 2. Find finetuned_* directories
    for p in glob.glob(os.path.join(search_dir, f"{method}_finetuned_*")):
        if not os.path.isdir(p): continue
        
        name = get_name(p)
        # Modified: Add "/checkpoints" to the path
        ckpt_dir = os.path.join(p, "checkpoints")

        # Check for config in the new checkpoints path first
        if os.path.exists(os.path.join(ckpt_dir, "config.json")):
             models.append({
                 "name": name, 
                 "dir": ckpt_dir, 
                 "config": os.path.join(ckpt_dir, "config.json")
             })
        # Fallback: check root (but usually finetuned weights are in /checkpoints)
        elif os.path.exists(os.path.join(p, "config.json")):
             models.append({
                 "name": name, 
                 "dir": p, 
                 "config": os.path.join(p, "config.json")
             })

    return models

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="longbench")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--compressed_base", required=True, help="Path to search for checkpoints")
    parser.add_argument("--model_prefix", required=True, help="Prefix string for model names (e.g., delta_net)")
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--max_length", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--step", default=-1, type=int)
    parser.add_argument("--method", default="l2", help="Compression method identifier (e.g., l2, wanda, etc.)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Discover Models (passing the prefix)
    models = find_models(args.compressed_base, args.model_prefix, args.method)
    
    if not models:
        print("No models found. Exiting.", flush=True)
        sys.exit(1)

    # 2. Initialize Ray
    print("Initializing Ray...", end=" ", flush=True)
    # Use manual memory limits to avoid automatic detection issues in cluster environments
    ray.init(
        ignore_reinit_error=True,
        object_store_memory=10**9,   # 1GB object store
        _memory=2 * 10**9           # 2GB system memory for workers
    )
    print("Done.\n", flush=True)
    
    # --- Check Resources ---
    resources = ray.cluster_resources()
    print(f"⚡ Ray Cluster: {int(resources.get('GPU', 0))} GPUs, {int(resources.get('CPU', 0))} CPUs detected.", flush=True)
    # -----------------------

    # 3. Launch Tasks
    print(f"Queuing {len(models)} evaluation tasks:", flush=True)
    futures = []
    for i, m in enumerate(models):
        print(f"  {i+1}. Queuing: {m['name']}", flush=True)
        output_file = os.path.join(args.output_dir, f"{m['name']}_results.json")
        futures.append(
            run_evaluation_task.remote(m['name'], m['dir'], m['config'], args, output_file)
        )
    print("-" * 60 + "\n", flush=True)

    # 4. Progress Tracking Loop
    total = len(futures)
    completed = 0
    results_data = []

    print(f"Waiting for results...", flush=True)
    while futures:
        done, futures = ray.wait(futures)
        for result in ray.get(done):
            completed += 1
            results_data.append(result)
            if result['status'] == 'success':
                print(f"[{completed}/{total}] \033[92mSUCCESS\033[0m: {result['name']} ({result['duration_idx']})", flush=True)
            else:
                print(f"[{completed}/{total}] \033[91mFAILED \033[0m: {result['name']}", flush=True)
                if result.get('error'):
                    print(f"       Error: {result['error'].strip().splitlines()[-1]}", flush=True)

    # 5. Save Summary
    summary_path = os.path.join(args.output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "total": total,
            "successful": sum(1 for r in results_data if r['status'] == 'success'),
            "failed": sum(1 for r in results_data if r['status'] == 'failed'),
            "details": results_data
        }, f, indent=2)

    print(f"\nResults saved to {summary_path}", flush=True)
    if any(r['status'] == 'failed' for r in results_data):
        sys.exit(1)

if __name__ == "__main__":
    main()