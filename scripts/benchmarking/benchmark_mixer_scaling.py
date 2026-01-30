#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

from fla.layers.delta_net import DeltaNet

from fla.layers.delta_net import DeltaNet
from fla.layers.gated_deltanet import GatedDeltaNet

def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'

def benchmark_layer(layer, batch_size, seq_len, hidden_size, warmup_steps, steps, device, dtype):
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
    
    # Warmup
    try:
        with torch.inference_mode():
            for _ in range(warmup_steps):
                _ = layer(hidden_states)
                torch.cuda.synchronize()
    except Exception as e:
        print(f"Error during warmup: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0

    # Benchmark
    latencies = []
    total_tokens = batch_size * seq_len * steps
    
    torch.cuda.reset_peak_memory_stats()
    overall_start = time.time()
    try:
        with torch.inference_mode():
            for _ in range(steps):
                torch.cuda.synchronize()
                start = time.time()
                _ = layer(hidden_states)
                torch.cuda.synchronize()
                latencies.append(time.time() - start)
    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 0, 0, 0
    
    overall_end = time.time()
    peak_mem = torch.cuda.max_memory_allocated()
    avg_latency = sum(latencies) / len(latencies) if latencies else 1e-9
    throughput = total_tokens / (overall_end - overall_start)
    
    return throughput, avg_latency, peak_mem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark joint scaling of sequence mixer layers")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    # Fixed settings
    hidden_size = 2048
    num_heads = 16
    fixed_head_v_dim = 128
    seq_len = 4096
    
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device('cuda')

    model_types = ["delta_net", "gated_deltanet"]
    key_dims_per_head = [128, 64, 32]
    
    results = {} # (model_type, k_dim_ph) -> {tps, latency, peak_mem, params}

    print(f"Benchmarking joint scaling (seq_len={seq_len})")
    print(f"Settings: hidden_size={hidden_size}, num_heads={num_heads}, head_v_dim={fixed_head_v_dim}")
    print("-" * 80)

    for m_type in model_types:
        for k_dim_ph in key_dims_per_head:
            if m_type == "delta_net":
                expand_k = float(k_dim_ph * num_heads) / hidden_size
                expand_v = float(fixed_head_v_dim * num_heads) / hidden_size
                layer = DeltaNet(
                    hidden_size=hidden_size,
                    expand_k=expand_k,
                    expand_v=expand_v,
                    num_heads=num_heads,
                    mode='chunk',
                    use_conv=True,
                    use_beta=True,
                    use_gate=False,
                    use_output_norm=True
                )
            else:
                layer = GatedDeltaNet(
                    hidden_size=hidden_size,
                    head_dim=k_dim_ph,
                    expand_v=float(fixed_head_v_dim) / k_dim_ph,
                    num_heads=num_heads,
                    mode='chunk',
                    use_gate=False,
                    use_short_conv=True,
                )
                
            layer.to(device).to(dtype).eval()
            
            num_params = sum(p.numel() for p in layer.parameters())
            print(f"Testing {m_type:15s} | head_k_dim: {k_dim_ph:3d} | head_v_dim: {fixed_head_v_dim:3d} | Params: {sizeof_fmt(num_params)}")
            
            tps, latency, peak_mem = benchmark_layer(
                layer, args.batch_size, seq_len, hidden_size, 
                args.warmup_steps, args.steps, device, dtype
            )
            
            results[(m_type, k_dim_ph)] = {
                "tps": tps,
                "latency": latency,
                "peak_mem": peak_mem,
                "params": num_params
            }
            if tps > 0:
                print(f"-> TPS: {tps:,.2f} tokens/s | Peak Mem: {sizeof_fmt(peak_mem)}")
            else:
                print(f"-> FAILED")

    # Metrics preparation
    model_labels = {"delta_net": "DeltaNet", "gated_deltanet": "Gated DeltaNet"}
    colors = {"delta_net": "#1f77b4", "gated_deltanet": "#ff7f0e"} # Standard Blue and Orange
    markers = {"delta_net": "o", "gated_deltanet": "s"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for m_type in model_types:
        tps_ratios = []
        mem_ratios = []
        baseline_tps = results[(m_type, 128)]["tps"]
        baseline_mem = results[(m_type, 128)]["peak_mem"]
        
        for k_dim in key_dims_per_head:
            tps_ratios.append(results[(m_type, k_dim)]["tps"] / baseline_tps)
            mem_ratios.append(results[(m_type, k_dim)]["peak_mem"] / baseline_mem)

        # Plot Throughput Speedup
        ax1.plot(key_dims_per_head, tps_ratios, label=model_labels[m_type], color=colors[m_type], 
                 marker=markers[m_type], linewidth=2, markersize=8)
        
        # Plot Memory Ratio (Footprint)
        ax2.plot(key_dims_per_head, mem_ratios, label=model_labels[m_type], color=colors[m_type], 
                 marker=markers[m_type], linewidth=2, markersize=8)

    # Styling ax1 (Speedup)
    ax1.set_title("Throughput Speedup vs. Key Dimension", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel(r"Key Dimension per head ($d_k$)", fontsize=12)
    ax1.set_ylabel("Speedup Ratio (x)", fontsize=12)
    ax1.set_xticks(key_dims_per_head)
    ax1.invert_xaxis() # Show 128 -> 32 direction
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    # Styling ax2 (Memory)
    ax2.set_title("Peak Memory Footprint vs. Key Dimension", fontsize=14, fontweight='bold', pad=15)
    ax2.set_xlabel(r"Key Dimension per head ($d_k$)", fontsize=12)
    ax2.set_ylabel("Memory Ratio (Baseline = 1.0)", fontsize=12)
    ax2.set_xticks(key_dims_per_head)
    ax2.invert_xaxis()
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = "benchmark_scaling_analysis.png"
    plt.savefig(plot_path, dpi=300)
    print("-" * 80)
    print(f"Scaling analysis plot saved to: {plot_path}")
    print("-" * 80)

    # Create a 3rd plot: Pareto Frontier (Efficiency)
    plt.figure(figsize=(8, 6))
    for m_type in model_types:
        x_mems = [results[(m_type, k)]["peak_mem"] / results[(m_type, 128)]["peak_mem"] for k in key_dims_per_head]
        y_tps = [results[(m_type, k)]["tps"] / results[(m_type, 128)]["tps"] for k in key_dims_per_head]
        
        plt.plot(x_mems, y_tps, label=model_labels[m_type], color=colors[m_type], marker=markers[m_type], linewidth=2)
        for i, k in enumerate(key_dims_per_head):
            plt.annotate(f"$d_k={k}$", (x_mems[i], y_tps[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.title("Scaling Efficiency: Speedup vs. Memory Footprint", fontsize=14, fontweight='bold')
    plt.xlabel("Relative Peak Memory Usage (Lower is Better)", fontsize=12)
    plt.ylabel("Relative Throughput Speedup (Higher is Better)", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig("benchmark_pareto_frontier.png", dpi=300)
    print(f"Pareto frontier plot saved to: benchmark_pareto_frontier.png")
    print("-" * 80)

    # LaTeX Table Generation
    print("LATEX TABLE GENERATION")
    print("-" * 80)
    
    latex_lines = [
        "\\begin{table}[h]",
        "\\centering",
        "\\begin{tabular}{l|c|cc|cc}",
        "\\toprule",
        "\\textbf{Model} & \\bm{$d_k$} & \\textbf{TPS} & \\textbf{Speedup} & \\textbf{Memory} & \\textbf{Ratio} \\\\",
        "\\midrule"
    ]
    
    for m_type in model_types:
        baseline_tps = results[(m_type, 128)]["tps"]
        baseline_mem = results[(m_type, 128)]["peak_mem"]
        model_name = "DeltaNet" if m_type == "delta_net" else "Gated DeltaNet"
        
        for k_dim in key_dims_per_head:
            res = results[(m_type, k_dim)]
            tps_str = f"{res['tps']/1e6:.1f}M"
            speedup = f"{res['tps']/baseline_tps:.2f}$\\times$" if baseline_tps > 0 else "-"
            mem_str = sizeof_fmt(res["peak_mem"])
            mem_ratio = f"{res['peak_mem']/baseline_mem:.2f}" if baseline_mem > 0 else "-"
            
            # Start row with model name only for the first k_dim
            prefix = model_name if k_dim == 128 else ""
            latex_lines.append(f"{prefix} & {k_dim} & {tps_str} & {speedup} & {mem_str} & {mem_ratio} \\\\")
        
        latex_lines.append("\\midrule")
    
    # Remove last midrule and close
    latex_lines.pop()
    latex_lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Scaling performance of sequence mixers at sequence length " + str(seq_len) + ".}",
        "\\label{tab:mixer_scaling}",
        "\\end{table}"
    ])
    
    latex_output = "\n".join(latex_lines)
    print(latex_output)
    print("-" * 80)
    
    # Save to file
    with open("benchmark_results.tex", "w") as f:
        f.write(latex_output)
    print(f"LaTeX table saved to: benchmark_results.tex")
    print("-" * 80)
