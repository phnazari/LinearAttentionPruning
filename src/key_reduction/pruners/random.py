#!/usr/bin/env python3
"""
Apply Random Low-Rank Pruning to DeltaNet Models
===============================================

This script applies structural pruning by randomly dropping dimensions.
It is a data-free approach used as a baseline to identify the importance
of structured selection compared to random selection.

Methodology:
- Randomly selects which dimensions to keep in attention projection layers.
- Per-Head Strategy: Keeps top dimensions per head to preserve architectural layout.
"""

import argparse
import os
import sys
import random
import torch
import math
import numpy as np
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Setup paths for flash-linear-attention and flame
file_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(file_dir, "../../.."))
sys.path.insert(0, os.path.join(project_root, "flash-linear-attention"))
sys.path.insert(0, os.path.join(project_root, "flame"))
sys.path.insert(0, project_root)

# Import flash-linear-attention modules
import fla # noqa
from fla.modules import ShortConvolution
import custom_models.delta_net_2 # noqa

def get_deltanet(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    return model

def get_random_indices_per_head(out_dim, n_keep, num_heads, device):
    """
    Randomly selects indices per head.
    """
    head_dim = out_dim // num_heads
    
    keep_indices = []
    for h in range(num_heads):
        # Generate random permutation of head indices
        head_indices = torch.randperm(head_dim, device=device)
        # Take the first n_keep indices
        top_idx = head_indices[:n_keep]
        
        # Sort indices to maintain relative order (helps stability)
        top_idx, _ = torch.sort(top_idx)
        
        global_indices = top_idx + (h * head_dim)
        keep_indices.append(global_indices)
        
    return torch.cat(keep_indices).to(device).long()

def prune_model_random(model, pruning_ratio, pruning_strategy):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.layers

    device = layers[0].attn.q_proj.weight.device
    print(f"Pruning with Random Strategy: {pruning_strategy}, Ratio: {pruning_ratio}")

    # Prune Layer by Layer
    for i, layer in enumerate(layers):
        if not hasattr(layer, "attn"): continue
        
        print(f"Pruning Layer {i}...")
        attn = layer.attn
        num_heads = attn.num_heads
        head_k_dim = attn.head_k_dim
        head_v_dim = getattr(attn, "head_v_dim", attn.value_dim // num_heads)
        
        qk_indices = []
        v_indices = []

        if pruning_strategy == "dimension":
            new_head_k_dim = math.floor(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            
            # Select random indices for Q/K
            qk_indices = get_random_indices_per_head(
                attn.q_proj.weight.shape[0],
                new_head_k_dim,
                num_heads,
                device
            )
            
            new_head_v_dim = head_v_dim
            new_key_dim = num_heads * new_head_k_dim
            new_value_dim = attn.value_dim
            print(f"  Strategy: Dimension | New Head K Dim: {new_head_k_dim}")

        elif pruning_strategy == "dimension_both":
            # 1. Q/K Dimension
            new_head_k_dim = int(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            
            qk_indices = get_random_indices_per_head(
                attn.q_proj.weight.shape[0],
                new_head_k_dim,
                num_heads,
                device
            )
            
            # 2. V Dimension
            new_head_v_dim = int(head_v_dim * (1 - pruning_ratio))
            if new_head_v_dim < 1: new_head_v_dim = 1
            
            v_indices = get_random_indices_per_head(
                attn.v_proj.weight.shape[0],
                new_head_v_dim,
                num_heads,
                device
            )
            
            # Handle normalization layers (Per-head slicing)
            if hasattr(attn, "o_norm"):
                if hasattr(attn.o_norm, "norms"): # IndependentNorm
                    for h_idx, norm in enumerate(attn.o_norm.norms):
                        head_v_indices = v_indices[h_idx * new_head_v_dim : (h_idx + 1) * new_head_v_dim]
                        local_head_v_indices = head_v_indices % head_v_dim
                        norm.weight.data = norm.weight.data[local_head_v_indices]
                        if hasattr(norm, "bias") and norm.bias is not None:
                            norm.bias.data = norm.bias.data[local_head_v_indices]
                else: # Standard Shared Norm
                    local_indices = v_indices[:new_head_v_dim] % head_v_dim
                    attn.o_norm.weight.data = attn.o_norm.weight.data[local_indices]
                    if hasattr(attn.o_norm, "bias") and attn.o_norm.bias is not None:
                        attn.o_norm.bias.data = attn.o_norm.bias.data[local_indices]
            
            new_key_dim = num_heads * new_head_k_dim
            new_value_dim = num_heads * new_head_v_dim
            print(f"  Strategy: Dimension Both | New K: {new_head_k_dim} | New V: {new_head_v_dim}")

        # --- EXECUTE SLICING ---
        if len(qk_indices) > 0:
            attn.q_proj = slice_linear_layer(attn.q_proj, qk_indices, dim=0)
            attn.k_proj = slice_linear_layer(attn.k_proj, qk_indices, dim=0)
            if hasattr(attn, "q_conv1d"): 
                attn.q_conv1d = slice_conv_layer(attn.q_conv1d, qk_indices)
            if hasattr(attn, "k_conv1d"): 
                attn.k_conv1d = slice_conv_layer(attn.k_conv1d, qk_indices)

        if len(v_indices) > 0:
            attn.v_proj = slice_linear_layer(attn.v_proj, v_indices, dim=0)
            if hasattr(attn, "v_conv1d"): 
                attn.v_conv1d = slice_conv_layer(attn.v_conv1d, v_indices)
            if hasattr(attn, "g_proj") and attn.g_proj is not None:
                attn.g_proj = slice_linear_layer(attn.g_proj, v_indices, dim=0)
            attn.o_proj = slice_linear_layer(attn.o_proj, v_indices, dim=1)

        # Update Config
        attn.head_k_dim = new_head_k_dim
        if hasattr(attn, "head_v_dim"): attn.head_v_dim = new_head_v_dim
        attn.key_dim = new_key_dim
        attn.value_dim = new_value_dim
        if hasattr(attn, "conv_key_dim"): attn.conv_key_dim = new_key_dim

    # Global Config Update
    first_attn = layers[0].attn
    if model.config.model_type == 'gated_deltanet':
        model.config.head_dim = first_attn.head_k_dim
        num_v_heads = getattr(first_attn, "num_v_heads", first_attn.num_heads)
        model.config.expand_v = float(first_attn.value_dim / (num_v_heads * model.config.head_dim))
    else:
        model.config.expand_k = float(first_attn.key_dim / model.config.hidden_size)
        model.config.expand_v = float(first_attn.value_dim / model.config.hidden_size)
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

def slice_conv_layer(layer, indices):
    if hasattr(layer, "shared_kernel") or hasattr(layer, "r"):
        device = layer.conv.bias.device if layer.conv.bias is not None else next(layer.parameters()).device
        dtype = layer.conv.bias.dtype if layer.conv.bias is not None else next(layer.parameters()).dtype
        new_hidden_size = len(indices)
        layer.resize(new_hidden_size=new_hidden_size, new_num_heads=None, device=device, dtype=dtype)
        return layer
    else:
        device = layer.weight.device
        dtype = layer.weight.dtype
        new_hid = len(indices)
        kernel = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        sliced_weight = layer.weight.data[indices]
        sliced_bias = layer.bias.data[indices] if layer.bias is not None else None
        
        new_layer = ShortConvolution(
            hidden_size=new_hid, kernel_size=kernel, 
            activation=layer.activation, bias=layer.bias is not None
        ).to(device=device, dtype=dtype)
        new_layer.weight.data = sliced_weight
        if sliced_bias is not None: new_layer.bias.data = sliced_bias
        return new_layer

def main():
    parser = argparse.ArgumentParser(description="DeltaNet Random Pruner (Data-Free)")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pruning_ratio", type=float, default=0.5)
    parser.add_argument("--pruning_strategy", type=str, default="dimension", 
                        choices=["dimension", "dimension_both"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    model = get_deltanet(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
    
    model = prune_model_random(
        model=model,
        pruning_ratio=args.pruning_ratio, 
        pruning_strategy=args.pruning_strategy
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    try: tokenizer.save_pretrained(args.output_dir)
    except: pass
    print(f"Pruned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
