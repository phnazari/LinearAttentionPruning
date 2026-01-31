#!/usr/bin/env python3
"""
Apply Channel Pruning to DeltaNet Models
========================================

This script applies structural pruning to DeltaNet models.
It supports two main strategies:
1. Dimension Pruning: Reduces head_k_dim within each head.
                     This reduces the rank of the attention mechanism.
2. Head Pruning: Removes entire attention heads (e.g., 16 -> 8).
                This reduces num_heads.

Importance Metrics:
1. Taylor: |Weight * Gradient|. Requires calibration data.
2. Magnitude: |Weight| (L2 Norm). Data-free.
3. Random: Random importance scores.

Implementation Details:
- Directly slices weight matrices and DeltaNet specific layers (ShortConvolution).
- Updates model configuration (num_heads, head_k_dim, expand_k, expand_v).
- Does NOT rely on external pruning libraries.
"""

import argparse
import os
import sys
import random
import torch
import math
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from einops import rearrange

import fla # noqa
import flame.custom_models.delta_net_2 # noqa
from flame.data import build_dataloader, build_dataset

# Import flash-linear-attention modules
import fla # noqa
from fla.modules import ShortConvolution
import flame.custom_models.delta_net_2 # noqa

# Import flame utilities
from flame.data import build_dataloader, build_dataset

def get_deltanet(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    if not hasattr(model, "seqlen"):
        model.seqlen = 2048
    return model

def compute_importance(model, dataloader, importance_type="taylor", num_examples=10, device="cuda"):
    """
    Compute importance scores for all pruneable layers.
    Returns a dictionary: {layer_idx: {'q_proj': scores, 'k_proj': scores}}
    """
    importance_scores = {}
    
    # Identify layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Could not find layers in model")

    # Initialize scores storage
    for i, layer in enumerate(layers):
        return importance_scores

    # 3. Taylor Importance (Weight * Gradient)
    if importance_type == "taylor":
        print("Computing Taylor importance (Weight * Gradient)...")
        # Prepare inputs
        example_prompts = []
        for i, batch in enumerate(dataloader):
            if i >= num_examples: break
            if isinstance(batch, dict):
                example_prompts.append(batch["input_ids"].to(device))
            else:
                example_prompts.append(batch[0].to(device))
        
        if not example_prompts:
            raise ValueError("No data found in dataloader for Taylor importance!")

        model.zero_grad()
        
        # Enable gradients for Q and K projections only
        for i, layer in enumerate(layers):
            if hasattr(layer, "attn"):
                layer.attn.q_proj.weight.requires_grad_(True)
                layer.attn.k_proj.weight.requires_grad_(True)
        
        total_loss = 0.0
        for inputs in tqdm(example_prompts, desc="Backward Pass"):
            # Forward + Backward
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss
            loss.backward()
            total_loss += loss.item()
        
        print(f"Average Loss: {total_loss / len(example_prompts):.4f}")
        
        # Collect scores: |W * Grad|
        for i, layer in enumerate(layers):
            if not hasattr(layer, "attn"): continue
            
            for proj_name in ["q_proj", "k_proj"]:
                if not hasattr(layer.attn, proj_name) or getattr(layer.attn, proj_name) is None:
                    continue
                    
                module = getattr(layer.attn, proj_name)
                if module.weight.grad is None:
                    print(f"WARNING: No gradient for Layer {i} {proj_name}. Fallback to random.")
                    score = torch.rand_like(module.weight).float()
                else:
                    score = (module.weight * module.weight.grad).abs().detach().float()
                
                importance_scores[i][proj_name] = score
                
                # Clean up to save memory
                module.weight.grad = None
                module.weight.requires_grad_(False)
                
        return importance_scores

    raise ValueError(f"Unknown importance type: {importance_type}")


def prune_model(model, importance_scores, pruning_ratio, pruning_strategy):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.layers

    device = layers[0].attn.q_proj.weight.device
    print(f"Pruning with strategy: {pruning_strategy}, Ratio: {pruning_ratio}")

    for i, layer in enumerate(layers):
        if i not in importance_scores: continue
        
        print(f"Pruning Layer {i}...")
        attn = layer.attn
        num_heads = attn.num_heads
        head_k_dim = attn.head_k_dim
        key_dim = attn.key_dim
        
        score_q = importance_scores[i]["q_proj"] # [key_dim, hidden]
        score_k = importance_scores[i]["k_proj"] # [key_dim, hidden]
        
        imp_q = score_q.sum(dim=1)  # [key_dim]
        imp_k = score_k.sum(dim=1)  # [key_dim]
        
        qk_imp = (imp_q + imp_k).view(num_heads, head_k_dim)
        
        # STRATEGY: Reduce Q/K Dimension (Rank) within each Head
        new_head_k_dim = math.floor(head_k_dim * (1 - pruning_ratio))
        if new_head_k_dim < 1: new_head_k_dim = 1
        _, topk_k = torch.topk(qk_imp, new_head_k_dim, dim=1) 
        topk_k, _ = torch.sort(topk_k, dim=1)
        head_offsets_k = torch.arange(num_heads, device=device).unsqueeze(1) * head_k_dim
        qk_indices = (topk_k + head_offsets_k).flatten()
        
        new_key_dim = num_heads * new_head_k_dim

        print(f"  Strategy: Dimension | New Head K Dim: {new_head_k_dim} | New Key Dim: {new_key_dim}")

        # Execute
        if len(qk_indices) > 0:
            attn.q_proj = slice_linear_layer(attn.q_proj, qk_indices, dim=0)
            attn.k_proj = slice_linear_layer(attn.k_proj, qk_indices, dim=0)
            if hasattr(attn, "q_conv1d"): 
                attn.q_conv1d = slice_conv_layer(attn.q_conv1d, qk_indices)
            if hasattr(attn, "k_conv1d"): 
                attn.k_conv1d = slice_conv_layer(attn.k_conv1d, qk_indices)

        attn.head_k_dim = new_head_k_dim
        attn.key_dim = new_key_dim
        if hasattr(attn, "conv_key_dim"): attn.conv_key_dim = new_key_dim
            
    first_attn = layers[0].attn
    
    # Robust configuration update based on model type
    if model.config.model_type == 'gated_deltanet':
        model.config.num_heads = first_attn.num_heads
        model.config.head_dim = first_attn.head_k_dim
    else:
        model.config.num_heads = first_attn.num_heads
        model.config.expand_k = float(first_attn.key_dim / model.config.hidden_size)
        if hasattr(model.config, "head_dim"):
            model.config.head_dim = first_attn.head_k_dim
            
    return model

def slice_linear_layer(layer, indices, dim=0):
    device = layer.weight.device
    dtype = layer.weight.dtype
    if dim == 0:
        new_out = len(indices)
        new_in = layer.in_features
        new_weight = layer.weight.data[indices]
        new_bias = layer.bias.data[indices] if layer.bias is not None else None
    else:
        new_out = layer.out_features
        new_in = len(indices)
        new_weight = layer.weight.data[:, indices]
        new_bias = layer.bias.data if layer.bias is not None else None
            
    new_layer = nn.Linear(new_in, new_out, bias=layer.bias is not None)
    new_layer.weight.data = new_weight
    if layer.bias is not None: new_layer.bias.data = new_bias
    return new_layer.to(device=device, dtype=dtype)

def slice_conv_layer(layer, indices, head_indices=None):
    if hasattr(layer, "shared_kernel") or hasattr(layer, "r"):
        # DeltaNet 2 SharedKernelConv1d
        device = layer.conv.bias.device if layer.conv.bias is not None else next(layer.parameters()).device
        dtype = layer.conv.bias.dtype if layer.conv.bias is not None else next(layer.parameters()).dtype
        
        new_hidden_size = len(indices)
        new_num_heads = len(head_indices) if head_indices is not None else None
        
        if head_indices is not None and getattr(layer, "per_head", False):
            # Head-level pruning for per-head kernels
            if hasattr(layer, "shared_kernel"):
                layer.shared_kernel.data = layer.shared_kernel.data[head_indices]
            elif hasattr(layer, "r"):
                layer.r.data = layer.r.data[head_indices]
        
        # SharedKernelConv1d has a resize method that handles internal ShortConvolution replacement
        layer.resize(new_hidden_size=new_hidden_size, new_num_heads=new_num_heads, device=device, dtype=dtype)
        return layer
    else:
        # Standard ShortConvolution
        device = layer.weight.device
        dtype = layer.weight.dtype
        new_hid = len(indices)
        kernel = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        new_layer = ShortConvolution(hidden_size=new_hid, kernel_size=kernel, activation=layer.activation)
        new_layer.weight.data = layer.weight.data[indices]
        if layer.bias is not None: new_layer.bias.data = layer.bias.data[indices]
        return new_layer.to(device=device, dtype=dtype)

def main():
    parser = argparse.ArgumentParser(description="DeltaNet Channel Pruner")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pruning_ratio", type=float, default=0.5)
    parser.add_argument("--pruning_strategy", type=str, default="dimension", choices=["dimension"], 
                        help="Pruning Strategy: 'dimension' (reduce head_dim)")
    parser.add_argument("--importance_type", type=str, default="taylor", choices=["taylor", "magnitude", "random"],
                        help="Importance Metric: 'taylor' (Gradient * Weight), 'magnitude' (Weight L2), 'random'")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_name", type=str, default="sample-10BT")
    parser.add_argument("--num_examples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = get_deltanet(args.model_path)
    model.eval()
    
    dataloader = None
    if args.importance_type == "taylor":
        tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
        dataset = build_dataset(
            dataset=args.dataset, 
            dataset_name=args.dataset_name, 
            dataset_split="train", 
            streaming=False,
            dp_degree=1,
            num_workers=args.num_workers, 
            seed=args.seed
        )
        dataloader = build_dataloader(
            dataset=dataset, 
            tokenizer=tokenizer, 
            rank=0,
            world_size=1,
            batch_size=args.batch_size, 
            seq_len=args.seq_len, 
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False
        )

    importance_scores = compute_importance(
        model=model,
        dataloader=dataloader,
        importance_type=args.importance_type,
        num_examples=args.num_examples,
        device=device
    )
    model = prune_model(
        model=model, 
        importance_scores=importance_scores, 
        pruning_ratio=args.pruning_ratio, 
        pruning_strategy=args.pruning_strategy
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    try:
        if args.importance_type == "taylor":
            tokenizer.save_pretrained(args.output_dir)
        else:
             tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
             tokenizer.save_pretrained(args.output_dir)
    except Exception as e:
        print(f"Warning: Failed to save tokenizer: {e}")
    print("Done!")

if __name__ == "__main__":
    main()
