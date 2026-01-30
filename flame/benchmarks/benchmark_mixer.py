#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import trange
from transformers import AutoConfig

from fla.layers.delta_net import DeltaNet

from fla.layers.delta_net import DeltaNet
from fla.layers.gated_deltanet import GatedDeltaNet

def sizeof_fmt(num, suffix='B'):
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark throughput for a single sequence mixer layer")
    parser.add_argument("--model", type=str, required=True, help="Path to model checkpoint or config")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=2048, help="Sequence length")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--steps", type=int, default=50, help="Benchmark steps")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device('cuda')

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    
    # Identify mixer type and initialize
    if config.model_type == 'delta_net':
        print(f"Initializing DeltaNet mixer for {args.model}")
        mixer = DeltaNet(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_k=config.expand_k,
            expand_v=config.expand_v,
            num_heads=config.num_heads,
            use_gate=config.use_gate,
            use_beta=config.use_beta,
            use_short_conv=config.use_short_conv,
            use_output_norm=config.use_output_norm,
            conv_size=config.conv_size,
            qk_norm=config.qk_norm,
            qk_activation=config.qk_activation,
            per_head_norm=config.per_head_norm,
            norm_eps=config.norm_eps,
        )
    elif config.model_type == 'gated_deltanet':
        print(f"Initializing GatedDeltaNet mixer for {args.model}")
        mixer = GatedDeltaNet(
            mode=config.attn_mode,
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            num_v_heads=config.num_v_heads,
            use_gate=config.use_gate,
            use_short_conv=config.use_short_conv,
            allow_neg_eigval=config.allow_neg_eigval,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
        )
    else:
        raise ValueError(f"Unsupported model type: {config.model_type}")

    mixer.to(device).to(dtype).eval()
    
    num_params = sum(p.numel() for p in mixer.parameters())
    print(f"Mixer parameters: {num_params:,} ({sizeof_fmt(num_params)})")

    # Benchmarking
    hidden_states = torch.randn(args.batch_size, args.seq_len, config.hidden_size, device=device, dtype=dtype)
    
    print(f"Warming up ({args.warmup_steps} steps)...")
    bar = trange(args.warmup_steps, desc="Warmup")
    with torch.inference_mode():
        for _ in bar:
            _ = mixer(hidden_states)
            torch.cuda.synchronize()
            bar.set_description_str(f"Warmup | Peak memory: {sizeof_fmt(torch.cuda.max_memory_allocated())}")

    print(f"Benchmarking ({args.steps} steps)...")
    torch.cuda.reset_peak_memory_stats()
    latencies = []
    total_tokens = 0
    
    bar = trange(args.steps, desc="Benchmark")
    overall_start = time.time()
    with torch.inference_mode():
        for _ in bar:
            torch.cuda.synchronize()
            start = time.time()
            _ = mixer(hidden_states)
            torch.cuda.synchronize()
            end = time.time()
            
            latencies.append(end - start)
            total_tokens += args.batch_size * args.seq_len
            current_throughput = total_tokens / (end - overall_start)
            bar.set_description_str(f"Throughput: {current_throughput:,.0f} tokens/s")

    avg_latency = sum(latencies) / len(latencies)
    throughput = total_tokens / (time.time() - overall_start)
    peak_mem = torch.cuda.max_memory_allocated()

    print(f"\nResults for {args.model}:")
    print(f"Throughput: {throughput:,.2f} tokens/s")
    print(f"Average latency: {avg_latency * 1000:.2f} ms")
    print(f"Peak memory: {sizeof_fmt(peak_mem)}")
