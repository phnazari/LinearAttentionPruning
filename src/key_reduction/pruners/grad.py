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
        if hasattr(layer, "attn"):
             storage = {
                 "q_proj": torch.zeros_like(layer.attn.q_proj.weight, requires_grad=False).float(),
                 "k_proj": torch.zeros_like(layer.attn.k_proj.weight, requires_grad=False).float(),
                 "v_proj": torch.zeros_like(layer.attn.v_proj.weight, requires_grad=False).float(),
                 "o_proj": torch.zeros_like(layer.attn.o_proj.weight, requires_grad=False).float(),
             }
             if hasattr(layer.attn, "g_proj") and layer.attn.g_proj is not None:
                 storage["g_proj"] = torch.zeros_like(layer.attn.g_proj.weight, requires_grad=False).float()
             if hasattr(layer.attn, "b_proj") and layer.attn.b_proj is not None:
                 storage["b_proj"] = torch.zeros_like(layer.attn.b_proj.weight, requires_grad=False).float()
             importance_scores[i] = storage

    # 1. Random Importance
    if importance_type == "random":
        print("Generating random importance scores...")
        for i in importance_scores:
            for k in importance_scores[i]:
                importance_scores[i][k] = torch.rand_like(importance_scores[i][k])
        return importance_scores

    # 2. Magnitude Importance (Weight Only)
    if importance_type == "magnitude":
        print("Computing magnitude importance (L1 norm of weights)...")
        for i, layer in enumerate(layers):
            if not hasattr(layer, "attn"): continue
            importance_scores[i]["q_proj"] = layer.attn.q_proj.weight.abs().detach().float()
            importance_scores[i]["k_proj"] = layer.attn.k_proj.weight.abs().detach().float()
            importance_scores[i]["v_proj"] = layer.attn.v_proj.weight.abs().detach().float()
            importance_scores[i]["o_proj"] = layer.attn.o_proj.weight.abs().detach().float()
            
            if "g_proj" in importance_scores[i]:
                importance_scores[i]["g_proj"] = layer.attn.g_proj.weight.abs().detach().float()
            if "b_proj" in importance_scores[i]:
                importance_scores[i]["b_proj"] = layer.attn.b_proj.weight.abs().detach().float()
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
        
        # Enable gradients for all attention projections
        for i, layer in enumerate(layers):
            if hasattr(layer, "attn"):
                layer.attn.q_proj.weight.requires_grad_(True)
                layer.attn.k_proj.weight.requires_grad_(True)
                layer.attn.v_proj.weight.requires_grad_(True)
                layer.attn.o_proj.weight.requires_grad_(True)
                if hasattr(layer.attn, "g_proj") and layer.attn.g_proj is not None:
                    layer.attn.g_proj.weight.requires_grad_(True)
                if hasattr(layer.attn, "b_proj") and layer.attn.b_proj is not None:
                    layer.attn.b_proj.weight.requires_grad_(True)
        
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
            
            for proj_name in ["q_proj", "k_proj", "v_proj", "o_proj", "g_proj", "b_proj"]:
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


def prune_model(model, importance_scores, pruning_ratio, pruning_strategy, norm_strategy="shared"):
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
        head_v_dim = getattr(attn, "head_v_dim", attn.value_dim // num_heads)
        key_dim = attn.key_dim
        value_dim = attn.value_dim
        
        score_q = importance_scores[i]["q_proj"] # [key_dim, hidden]
        score_k = importance_scores[i]["k_proj"] # [key_dim, hidden]
        score_v = importance_scores[i]["v_proj"] # [value_dim, hidden]
        score_o = importance_scores[i]["o_proj"] # [hidden, value_dim]
        
        imp_q = score_q.sum(dim=1)  # [key_dim]
        imp_k = score_k.sum(dim=1)  # [key_dim]
        imp_v = score_v.sum(dim=1)  # [value_dim]
        imp_o = score_o.sum(dim=0)  # [value_dim]
        
        qk_imp = (imp_q + imp_k).view(num_heads, head_k_dim)
        vo_imp = (imp_v + imp_o)
        
        if "g_proj" in importance_scores[i]:
            score_g = importance_scores[i]["g_proj"]
            imp_g = score_g.sum(dim=1)
            vo_imp = vo_imp + imp_g
            
        vo_imp = vo_imp.view(num_heads, head_v_dim)
        total_imp = qk_imp.sum(dim=1) + vo_imp.sum(dim=1) # [num_heads]
        
        if "b_proj" in importance_scores[i]:
            score_b = importance_scores[i]["b_proj"]
            total_imp = total_imp + score_b.sum(dim=1)
        
        qk_indices = []
        v_indices = []
        beta_indices = []

        if pruning_strategy == "dimension":
            # STRATEGY 1: Reduce Q/K Dimension (Rank) within each Head
            # Only reduce Q/K, keep V/O unchanged as requested.
            
            # 1. QK Dimension
            new_head_k_dim = math.floor(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            _, topk_k = torch.topk(qk_imp, new_head_k_dim, dim=1) 
            topk_k, _ = torch.sort(topk_k, dim=1)
            head_offsets_k = torch.arange(num_heads, device=device).unsqueeze(1) * head_k_dim
            qk_indices = (topk_k + head_offsets_k).flatten()
            
            # 2. V Dimension (Unchanged)
            v_indices = [] # Signal that no V/O pruning is needed
            new_head_v_dim = head_v_dim
            new_value_dim = value_dim
            
            new_num_heads = num_heads 
            new_key_dim = new_num_heads * new_head_k_dim

            print(f"  Strategy: Dimension | New Head K Dim: {new_head_k_dim} | New Key Dim: {new_key_dim}")

        elif pruning_strategy == "dimension_both":
            # STRATEGY 2: Reduce BOTH Q/K and V/O dimensions
            
            # 1. QK Dimension (Sequential pruning within heads)
            new_head_k_dim = int(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            _, topk_k = torch.topk(qk_imp, new_head_k_dim, dim=1) 
            topk_k, _ = torch.sort(topk_k, dim=1)
            head_offsets_k = torch.arange(num_heads, device=device).unsqueeze(1) * head_k_dim
            qk_indices = (topk_k + head_offsets_k).flatten()
            
            # 2. V Dimension
            if norm_strategy == "shared":
                # Approach A: Global Selection (Averaged Importance)
                new_head_v_dim = int(head_v_dim * (1 - pruning_ratio))
                if new_head_v_dim < 1: new_head_v_dim = 1
                global_v_imp = vo_imp.mean(dim=0)
                _, global_v_topk = torch.topk(global_v_imp, new_head_v_dim)
                global_v_topk, _ = torch.sort(global_v_topk)
                head_offsets_v = torch.arange(num_heads, device=device).unsqueeze(1) * head_v_dim
                v_indices = (global_v_topk.unsqueeze(0).repeat(num_heads, 1) + head_offsets_v).flatten()
                
                if hasattr(attn, "o_norm"):
                    attn.o_norm.weight.data = attn.o_norm.weight.data[global_v_topk]
                    if hasattr(attn.o_norm, "bias") and attn.o_norm.bias is not None:
                        attn.o_norm.bias.data = attn.o_norm.bias.data[global_v_topk]

            elif norm_strategy == "union":
                # Approach B: Union Selection (Inclusive)
                new_head_v_dim_target = int(head_v_dim * (1 - pruning_ratio))
                if new_head_v_dim_target < 1: new_head_v_dim_target = 1
                
                union_indices = set()
                for h in range(num_heads):
                    _, indices_h = torch.topk(vo_imp[h], new_head_v_dim_target)
                    union_indices.update(indices_h.tolist())
                
                global_v_topk = torch.tensor(sorted(list(union_indices)), device=device)
                new_head_v_dim = len(global_v_topk)
                print(f"  Union Strategy: Target Dim {new_head_v_dim_target} -> Final Dim {new_head_v_dim}")
                
                head_offsets_v = torch.arange(num_heads, device=device).unsqueeze(1) * head_v_dim
                v_indices = (global_v_topk.unsqueeze(0).repeat(num_heads, 1) + head_offsets_v).flatten()
                
                if hasattr(attn, "o_norm"):
                    attn.o_norm.weight.data = attn.o_norm.weight.data[global_v_topk]
                    if hasattr(attn.o_norm, "bias") and attn.o_norm.bias is not None:
                        attn.o_norm.bias.data = attn.o_norm.bias.data[global_v_topk]

            elif norm_strategy == "independent":
                # Approach C: Independent Norms per Head
                new_head_v_dim = int(head_v_dim * (1 - pruning_ratio))
                if new_head_v_dim < 1: new_head_v_dim = 1
                
                # Update config to ensure the model initializes correctly later
                model.config.per_head_norm = True
                
                # Dynamically import IndependentNorm based on model type
                if model.config.model_type == 'delta_net_2':
                    from custom_models.delta_net_2.delta_net_2 import IndependentNorm
                else:
                    from fla.layers.delta_net import IndependentNorm
                
                head_v_indices = []
                if hasattr(attn, "o_norm"):
                    new_ind_norm = IndependentNorm(
                        new_head_v_dim, 
                        eps=attn.o_norm.eps, 
                        num_heads=num_heads, 
                        use_gate=getattr(attn, "use_gate", False)
                    ).to(device=device, dtype=attn.o_norm.weight.dtype)
                
                for h in range(num_heads):
                    _, topk_h = torch.topk(vo_imp[h], new_head_v_dim)
                    topk_h, _ = torch.sort(topk_h)
                    head_v_indices.append(topk_h + h * head_v_dim)
                    
                    if hasattr(attn, "o_norm"):
                        new_ind_norm.norms[h].weight.data = attn.o_norm.weight.data[topk_h].clone()
                        if hasattr(attn.o_norm, "bias") and attn.o_norm.bias is not None:
                            new_ind_norm.norms[h].bias.data = attn.o_norm.bias.data[topk_h].clone()

                v_indices = torch.cat(head_v_indices)
                if hasattr(attn, "o_norm"):
                    attn.o_norm = new_ind_norm

            elif norm_strategy == "permute":
                # Approach D: Permutation/Alignment (Gain-based alignment, shared Norm)
                new_head_v_dim = int(head_v_dim * (1 - pruning_ratio))
                if new_head_v_dim < 1: new_head_v_dim = 1
                
                head_v_indices = []
                mapped_norm_weights = []
                
                for h in range(num_heads):
                    # 1. Pick top K important dimensions for this head
                    _, topk_h = torch.topk(vo_imp[h], new_head_v_dim)
                    
                    # 2. ALIGNMENT step (Fixing the previous mistake):
                    # Sort the selected dimensions by their original normalization gain.
                    # This ensures dimensions with similar gains map to the same shared norm slot.
                    if hasattr(attn, "o_norm"):
                        gains = attn.o_norm.weight.data[topk_h]
                        sort_idx = torch.argsort(gains)
                        topk_h = topk_h[sort_idx]
                        mapped_norm_weights.append(attn.o_norm.weight.data[topk_h])
                    
                    head_v_indices.append(topk_h + h * head_v_dim)
                
                v_indices = torch.cat(head_v_indices)
                
                # 3. Optimize the shared norm for the mapped dimensions
                if hasattr(attn, "o_norm"):
                    # Stack weights and take the mean across heads for each slot
                    stacked_weights = torch.stack(mapped_norm_weights)
                    new_shared_weight = stacked_weights.mean(dim=0)
                    
                    # Update parameters (create new parameter of correct size)
                    attn.o_norm.weight = nn.Parameter(new_shared_weight[:new_head_v_dim])
                    if hasattr(attn.o_norm, "bias") and attn.o_norm.bias is not None:
                        # Similar logic for bias if present
                        bias_list = [attn.o_norm.bias.data[topk_h] for _ in range(num_heads)] # Simplified
                        new_shared_bias = torch.stack(bias_list).mean(dim=0)
                        attn.o_norm.bias = nn.Parameter(new_shared_bias[:new_head_v_dim])

            new_num_heads = num_heads
            new_key_dim = new_num_heads * new_head_k_dim
            new_value_dim = new_num_heads * new_head_v_dim
            print(f"  Strategy: Dimension Both ({norm_strategy}) | New Head K Dim: {new_head_k_dim} | New Head V Dim: {new_head_v_dim}")

        elif pruning_strategy == "head":
            new_num_heads = int(num_heads * (1 - pruning_ratio))
            if new_num_heads < 1: new_num_heads = 1
            _, topk_heads = torch.topk(total_imp, new_num_heads)
            topk_heads, _ = torch.sort(topk_heads)
            
            for h in topk_heads:
                qk_indices.append(torch.arange(h * head_k_dim, (h + 1) * head_k_dim, device=device))
                v_indices.append(torch.arange(h * head_v_dim, (h + 1) * head_v_dim, device=device))
            qk_indices = torch.cat(qk_indices)
            v_indices = torch.cat(v_indices)
            beta_indices = topk_heads
            
            new_head_k_dim = head_k_dim
            new_head_v_dim = head_v_dim
            new_key_dim = new_num_heads * new_head_k_dim
            new_value_dim = new_num_heads * new_head_v_dim
            print(f"  Strategy: Head | New Num Heads: {new_num_heads} | New Key Dim: {new_key_dim}")

        # Execute
        if len(qk_indices) > 0:
            attn.q_proj = slice_linear_layer(attn.q_proj, qk_indices, dim=0)
            attn.k_proj = slice_linear_layer(attn.k_proj, qk_indices, dim=0)
            if hasattr(attn, "q_conv1d"): 
                attn.q_conv1d = slice_conv_layer(attn.q_conv1d, qk_indices, head_indices=beta_indices if pruning_strategy == "head" else None)
            if hasattr(attn, "k_conv1d"): 
                attn.k_conv1d = slice_conv_layer(attn.k_conv1d, qk_indices, head_indices=beta_indices if pruning_strategy == "head" else None)

        if len(v_indices) > 0:
            attn.v_proj = slice_linear_layer(attn.v_proj, v_indices, dim=0)
            if hasattr(attn, "v_conv1d"): 
                attn.v_conv1d = slice_conv_layer(attn.v_conv1d, v_indices, head_indices=beta_indices if pruning_strategy == "head" else None)
            if hasattr(attn, "g_proj") and attn.g_proj is not None:
                attn.g_proj = slice_linear_layer(attn.g_proj, v_indices, dim=0)
            attn.o_proj = slice_linear_layer(attn.o_proj, v_indices, dim=1)
            if pruning_strategy == "head" and hasattr(attn, "b_proj") and attn.b_proj is not None:
                attn.b_proj = slice_linear_layer(attn.b_proj, beta_indices, dim=0)

        attn.num_heads = new_num_heads
        attn.head_k_dim = new_head_k_dim
        if hasattr(attn, "head_v_dim"):
            attn.head_v_dim = new_head_v_dim
        attn.key_dim = new_key_dim
        attn.value_dim = new_value_dim
        if hasattr(attn, "conv_key_dim"): attn.conv_key_dim = new_key_dim
            
    if pruning_strategy == "head":
        model.config.num_heads = int(model.config.num_heads * (1 - pruning_ratio))
    
    first_attn = layers[0].attn
    
    # Robust configuration update based on model type
    if model.config.model_type == 'gated_deltanet':
        # GatedDeltaNet uses head_dim for its key_dim and value_dim calculations
        # head_k_dim = head_dim
        # head_v_dim = head_dim * expand_v
        model.config.num_heads = first_attn.num_heads
        model.config.head_dim = first_attn.head_k_dim
        
        num_v_heads = getattr(first_attn, "num_v_heads", first_attn.num_heads)
        if hasattr(model.config, "num_v_heads"):
            model.config.num_v_heads = num_v_heads
            
        # Update expand_v such that head_v_dim = head_dim * expand_v
        # head_v_dim = value_dim / num_v_heads
        # So we set expand_v = value_dim / (num_v_heads * head_dim)
        # This correctly increases expand_v when head_dim is shrunk.
        model.config.expand_v = float(first_attn.value_dim / (num_v_heads * model.config.head_dim))
    else:
        # Standard DeltaNet, DeltaNet2 and others usually use expand_k/expand_v relative to hidden_size
        # key_dim = hidden_size * expand_k
        # value_dim = hidden_size * expand_v
        model.config.num_heads = first_attn.num_heads
        model.config.expand_k = float(first_attn.key_dim / model.config.hidden_size)
        model.config.expand_v = float(first_attn.value_dim / model.config.hidden_size)
        
        # Update head_dim if it exists in config (mostly for legacy/compatibility)
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
    parser.add_argument("--pruning_strategy", type=str, default="dimension", choices=["dimension", "head", "dimension_both"], 
                        help="Pruning Strategy: 'dimension' (reduce head_dim), 'head' (reduce num_heads), 'dimension_both' (reduce both K and V dims)")
    parser.add_argument("--norm_pruning_strategy", type=str, default="shared", choices=["shared", "independent", "union", "permute"],
                        help="Normalization strategy for dimension_both: 'shared' (global selection), 'independent' (separate norm per head), 'union' (keep all important indices), 'permute' (independent selection aligned to shared norm)")
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
        pruning_strategy=args.pruning_strategy,
        norm_strategy=args.norm_pruning_strategy
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
