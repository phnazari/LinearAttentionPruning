#!/usr/bin/env python
# Copyright (c) 2023-2024, Songlin Yang, Yu Zhang.
"""
Benchmark forward-only throughput for a single model.

This script measures pure forward pass performance (no backward pass)
to isolate inference speed.

Usage:
    python benchmarks/benchmark_forward.py \
        --model /path/to/checkpoint \
        --batch_size 32 \
        --seq_len 2048 \
        --warmup_steps 10 \
        --steps 50
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from tqdm import trange
from transformers import AutoConfig, AutoModelForCausalLM

import fla  # noqa

import fla  # noqa
import custom_models.delta_net_2  # noqa - register custom models


def sizeof_fmt(num, suffix='B'):
    """Format bytes as human-readable string."""
    for unit in ('', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi'):
        if abs(num) < 1024.0:
            return f'{num:.2f}{unit}{suffix}'
        num /= 1024.0
    return f'{num:.2f}Yi{suffix}'


def prepare_inputs(
    batch_size: int,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
):
    """Prepare random input tokens for benchmarking."""
    tokens = torch.randint(high=vocab_size, size=(batch_size, seq_len), device=device)
    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark forward-only throughput for a model"
    )
    
    # Model path
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Model name for display (default: use model path)",
    )
    
    # Benchmark settings
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for benchmarking",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length",
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Number of warmup steps",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of benchmark steps",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model data type",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model with torch.compile",
    )
    
    args = parser.parse_args()
    
    # Map dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]
    
    model_name = args.name if args.name else args.model
    
    device = torch.device('cuda')
    device_name = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"
    
    print("=" * 70)
    print(f"Benchmarking: {model_name}")
    print("=" * 70)
    print(f"Model path:       {args.model}")
    print(f"Device:           {device} ({device_name})")
    print(f"Batch size:       {args.batch_size}")
    print(f"Sequence length:  {args.seq_len}")
    print(f"Warmup steps:     {args.warmup_steps}")
    print(f"Benchmark steps:  {args.steps}")
    print(f"Data type:        {args.dtype}")
    print(f"Compile:          {args.compile}")
    print("=" * 70)
    
    # Load model
    print("\nLoading model...")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).to(device)
    
    if args.compile:
        print("Compiling the model")
        model = torch.compile(model)
    
    model.eval()
    
    num_parameters = model.num_parameters()
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_parameters:,} ({sizeof_fmt(num_parameters)})")
    print(f"Number of trainable parameters: {trainable_parameters:,}")
    
    # Reset memory stats
    torch.cuda.reset_peak_memory_stats(device)
    initial_memory = torch.cuda.memory_allocated(device)
    print(f"Initial memory allocated: {sizeof_fmt(initial_memory)}")
    
    # Warmup
    print(f"\nWarming up ({args.warmup_steps} steps)...")
    bar = trange(args.warmup_steps, desc="Warmup")
    with torch.inference_mode():
        for _ in bar:
            tokens = prepare_inputs(args.batch_size, args.seq_len, config.vocab_size, device)
            _ = model(input_ids=tokens)
            torch.cuda.synchronize(device)
            bar.set_description_str(f"Warmup | Max memory: {sizeof_fmt(torch.cuda.max_memory_allocated(device))}")
    
    peak_warmup_memory = torch.cuda.max_memory_allocated(device)
    print(f"Peak memory after warmup: {sizeof_fmt(peak_warmup_memory)}")
    
    # Benchmark
    print(f"\nBenchmarking ({args.steps} steps)...")
    torch.cuda.reset_peak_memory_stats(device)
    
    total_tokens = 0
    latencies = []
    
    bar = trange(args.steps, desc="Benchmark")
    with torch.inference_mode():
        torch.cuda.synchronize(device)
        overall_start = time.time()
        
        for _ in bar:
            tokens = prepare_inputs(args.batch_size, args.seq_len, config.vocab_size, device)
            
            # Measure single step latency
            torch.cuda.synchronize(device)
            step_start = time.time()
            
            _ = model(input_ids=tokens)
            
            torch.cuda.synchronize(device)
            step_end = time.time()
            
            step_latency = step_end - step_start
            latencies.append(step_latency)
            total_tokens += args.batch_size * args.seq_len
            
            current_throughput = total_tokens / (step_end - overall_start)
            bar.set_description_str(f"Throughput: {current_throughput:,.0f} tokens/s")
        
        torch.cuda.synchronize(device)
        overall_end = time.time()
    
    # Calculate statistics
    total_time = overall_end - overall_start
    throughput = total_tokens / total_time
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)
    
    peak_benchmark_memory = torch.cuda.max_memory_allocated(device)
    
    # Print results
    print(f"\n{'=' * 70}")
    print(f"Results")
    print(f"{'=' * 70}")
    print(f"Total tokens processed:    {total_tokens:,}")
    print(f"Total time:                {total_time:.2f}s")
    print(f"Throughput:                {throughput:,.2f} tokens/s")
    print(f"Average latency per step:  {avg_latency * 1000:.2f}ms")
    print(f"Min latency per step:      {min_latency * 1000:.2f}ms")
    print(f"Max latency per step:      {max_latency * 1000:.2f}ms")
    print(f"Peak memory:               {sizeof_fmt(peak_benchmark_memory)}")
    print(f"{'=' * 70}")
    
    print("\nBenchmark complete!")
