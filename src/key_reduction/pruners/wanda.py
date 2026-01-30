#!/usr/bin/env python3
"""
Apply Wanda-based Low-Rank Pruning to DeltaNet Models
=====================================================

This script applies structural pruning using the Wanda metric (Weight x Activation Norm).
It identifies important head dimensions based on the magnitude of weights weighted by
the input activation norms collected from a calibration set.

Methodology:
- Collects feature-wise L2 norms of inputs to attention projection layers.
- Calculates dimension importance: Score_i = sum_j (|W_ij| * norm_j).
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
from tqdm import tqdm


# Import flash-linear-attention modules
import fla # noqa
from fla.modules import ShortConvolution
import custom_models.delta_net_2 # noqa
from flame.data import build_dataloader, build_dataset

def get_deltanet(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cuda" if torch.cuda.is_available() else "cpu"
    )
    return model

class ActivationNormCollector:
    """Collects feature-wise activation norms (n_j = sqrt(mean(X_j^2)))."""
    def __init__(self, model, layers_to_hook):
        self.norms_sq_sum = {}
        self.n_tokens = {}
        self.hooks = []
        for layer_idx in layers_to_hook:
            layer = model.model.layers[layer_idx]
            if hasattr(layer, "attn"):
                # We hook the input to q_proj as it represents the input to the whole attention block
                # (after layer norm).
                module = layer.attn.q_proj
                hook = module.register_forward_hook(self.get_hook(layer_idx))
                self.hooks.append(hook)
                
    def get_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # input is a tuple (x,), x is [batch, seq, hidden]
            x = input[0].detach().float()
            x = x.view(-1, x.shape[-1]) # [tokens, hidden]
            
            if layer_idx not in self.norms_sq_sum:
                self.norms_sq_sum[layer_idx] = torch.zeros(x.shape[-1], device=x.device)
                self.n_tokens[layer_idx] = 0
            
            self.norms_sq_sum[layer_idx] += torch.sum(x**2, dim=0)
            self.n_tokens[layer_idx] += x.shape[0]
        return hook_fn

    def remove_hooks(self):
        for hook in self.hooks: hook.remove()
            
    def get_norms(self, layer_idx):
        if layer_idx not in self.norms_sq_sum: return None
        return torch.sqrt(self.norms_sq_sum[layer_idx] / self.n_tokens[layer_idx])

def get_wanda_indices_per_head(weights_list, input_norms, n_keep, num_heads):
    """
    Computes Wanda scores and returns top indices per head.
    Score_i = sum_j ( sum_weights |W_ij| * norm_j )
    """
    # 1. Aggregate weight magnitudes across all provided projections (e.g. Q and K)
    # weights_list is a list of [num_heads * head_dim, hidden_dim] tensors
    device = weights_list[0].device
    out_dim, in_dim = weights_list[0].shape
    head_dim = out_dim // num_heads
    
    # Combined absolute weights
    combined_abs_w = torch.zeros_like(weights_list[0], dtype=torch.float32)
    for w in weights_list:
        combined_abs_w += torch.abs(w.detach().float())
    
    # 2. Compute Wanda Metric: element-wise product followed by row sum
    # input_norms: [hidden_dim]
    wanda_metric = combined_abs_w * input_norms.unsqueeze(0) # [out_dim, in_dim]
    
    # Score per output dimension (row sum)
    dim_scores = torch.sum(wanda_metric, dim=1) # [out_dim]
    
    # 3. Reshape scores and pick top k per head
    dim_scores_reshaped = dim_scores.view(num_heads, head_dim)
    
    keep_indices = []
    for h in range(num_heads):
        head_scores = dim_scores_reshaped[h]
        # Get indices of top n_keep scores
        _, top_idx = torch.topk(head_scores, n_keep, largest=True)
        
        # Sort indices to maintain relative order (helps stability)
        top_idx, _ = torch.sort(top_idx)
        
        global_indices = top_idx + (h * head_dim)
        keep_indices.append(global_indices)
        
    return torch.cat(keep_indices).to(device).long()

def prune_model_wanda(model, dataloader, pruning_ratio, pruning_strategy, num_batches=10):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.layers

    device = layers[0].attn.q_proj.weight.device
    print(f"Pruning with Wanda Strategy: {pruning_strategy}, Ratio: {pruning_ratio}")

    # 1. Collect Activation Norms
    layer_indices = [i for i, l in enumerate(layers) if hasattr(l, "attn")]
    collector = ActivationNormCollector(model, layer_indices)
    
    print(f"Collecting activation norms over {num_batches} batches...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_batches)):
            if i >= num_batches: break
            if isinstance(batch, dict):
                inputs = batch["input_ids"].to(device)
            else:
                inputs = batch[0].to(device)
            model(inputs)
            
    collector.remove_hooks()
    
    # 2. Prune Layer by Layer
    for i, layer in enumerate(layers):
        if not hasattr(layer, "attn"): continue
        
        print(f"Pruning Layer {i}...")
        attn = layer.attn
        num_heads = attn.num_heads
        head_k_dim = attn.head_k_dim
        head_v_dim = getattr(attn, "head_v_dim", attn.value_dim // num_heads)
        
        input_norms = collector.get_norms(i).to(device)
        
        qk_indices = []
        v_indices = []

        if pruning_strategy == "dimension":
            new_head_k_dim = math.floor(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            
            # Combine Q and K for dimension selection
            qk_indices = get_wanda_indices_per_head(
                [attn.q_proj.weight, attn.k_proj.weight],
                input_norms,
                new_head_k_dim,
                num_heads
            )
            
            new_head_v_dim = head_v_dim
            new_key_dim = num_heads * new_head_k_dim
            new_value_dim = attn.value_dim
            print(f"  Strategy: Dimension | New Head K Dim: {new_head_k_dim}")

        elif pruning_strategy == "dimension_both":
            # 1. Q/K Dimension
            new_head_k_dim = int(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            
            qk_indices = get_wanda_indices_per_head(
                [attn.q_proj.weight, attn.k_proj.weight],
                input_norms,
                new_head_k_dim,
                num_heads
            )
            
            # 2. V Dimension
            new_head_v_dim = int(head_v_dim * (1 - pruning_ratio))
            if new_head_v_dim < 1: new_head_v_dim = 1
            
            v_weights = [attn.v_proj.weight]
            if hasattr(attn, "g_proj") and attn.g_proj is not None:
                v_weights.append(attn.g_proj.weight)
                
            v_indices = get_wanda_indices_per_head(
                v_weights,
                input_norms,
                new_head_v_dim,
                num_heads
            )
            
            # Handle normalization layers (Per-head slicing)
            if hasattr(attn, "o_norm"):
                v_idx_t = v_indices.long()
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
    parser = argparse.ArgumentParser(description="DeltaNet Wanda Pruner")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pruning_ratio", type=float, default=0.5)
    parser.add_argument("--pruning_strategy", type=str, default="dimension", 
                        choices=["dimension", "dimension_both"])
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_name", type=str, default="sample-10BT")
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    model = get_deltanet(args.model_path)
    
    tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
    dataset = build_dataset(
        dataset=args.dataset, dataset_name=args.dataset_name, dataset_split="train", 
        streaming=False, dp_degree=1, num_workers=args.num_workers, seed=args.seed
    )
    dataloader = build_dataloader(
        dataset=dataset, tokenizer=tokenizer, rank=0, world_size=1, 
        batch_size=args.batch_size, seq_len=args.seq_len, 
        num_workers=args.num_workers, pin_memory=True, persistent_workers=False
    )
    
    model = prune_model_wanda(
        model=model, dataloader=dataloader,
        pruning_ratio=args.pruning_ratio, pruning_strategy=args.pruning_strategy,
        num_batches=args.num_batches
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    try: tokenizer.save_pretrained(args.output_dir)
    except: pass
    print(f"Pruned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()
