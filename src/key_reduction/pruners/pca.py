#!/usr/bin/env python3
"""
Apply PCA-based Low-Rank Pruning to DeltaNet Models
==================================================

This script applies structural pruning using PCA (Principal Component Analysis).
It identifies important dimensions in the key (and optionally query) space by
collecting activation statistics on a calibration dataset.

Methodology:
- Collects activation statistics (K and optionally Q) from forward passes.
- Computes PCA projection matrices per layer and head.
- Projects weights into the lower-dimensional subspace.
- Updates model config to reflect reduced dimensions.
"""

import argparse
import os
import sys
import torch
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from einops import rearrange
from tqdm import tqdm
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Import flash-linear-attention modules
import fla # noqa
from fla.modules import ShortConvolution
from fla.modules.l2norm import l2_norm
import custom_models.delta_net_2 # noqa

from flame.data import build_dataloader, build_dataset

# ==============================================================================
# PCA UTILITY FUNCTIONS (Adapted from legacy compression_utils.py)
# ==============================================================================

def compute_pca_projection(
    k_states: Optional[List[torch.Tensor]],
    target_rank: Optional[int],
    head_dim: int,
    num_heads: int,
    variance_threshold: float = 0.95,
    max_compression_rate: Optional[float] = None,
    compression_mode: str = "pca",
    pad_to_original_dim: bool = False,
    adversarial: bool = False,
    matrix_seed: Optional[int] = None,
    k_sum: Optional[torch.Tensor] = None,
    k_outer: Optional[torch.Tensor] = None,
    n: Optional[int] = None,
) -> Tuple[torch.Tensor, int]:
    """
    Compute PCA projection matrix using eigendecomposition for dimensionality reduction.
    """
    all_eigenvectors = []
    actual_ranks = []

    if k_states is not None and len(k_states) > 0:
        K = torch.cat(k_states, dim=0).float()
    else:
        K = None

    if compression_mode == "pca" or target_rank is None:
        for h in range(num_heads):
            if K is not None:
                k_h = K[:, h, :]
                mean = k_h.mean(dim=0, keepdim=True)
                k_centered = k_h - mean
                cov = k_centered.t() @ k_centered / (k_h.shape[0] - 1)
            elif k_sum is not None and k_outer is not None and n is not None:
                mu = k_sum[h].float() / n
                cov = (k_outer[h].float() - n * torch.outer(mu, mu)) / (n - 1)
            else:
                raise ValueError("Must provide either k_states or (k_sum, k_outer, n)")

            eigenvalues, eigenvectors = torch.linalg.eigh(cov)

            if adversarial:
                pass
            else:
                eigenvalues = eigenvalues.flip(0)
                eigenvectors = eigenvectors.flip(1)
            
            eigenvalues = eigenvalues.clamp(min=0)
            all_eigenvectors.append(eigenvectors)

            if target_rank is None:
                total_var = eigenvalues.sum()
                cumulative_var = torch.cumsum(eigenvalues, dim=0)
                cumulative_ratio = cumulative_var / total_var
                rank = (cumulative_ratio < variance_threshold).sum().item() + 1
                rank = min(rank, head_dim)
                if max_compression_rate is not None:
                    min_rank = max(1, int(head_dim / max_compression_rate))
                    rank = max(rank, min_rank)
                actual_ranks.append(rank)
            else:
                actual_ranks.append(target_rank)
    
    if compression_mode in ["random", "random_nonorth"]:
        random_eigenvectors = []
        if matrix_seed: torch.manual_seed(matrix_seed)
        for h in range(num_heads):
            random_matrix = torch.randn(head_dim, head_dim, dtype=torch.float32)
            Q, _ = torch.linalg.qr(random_matrix)
            random_eigenvectors.append(random_matrix if compression_mode == "random_nonorth" else Q)
        all_eigenvectors = random_eigenvectors
    elif compression_mode == "trivial":
        all_eigenvectors = [torch.eye(head_dim, dtype=torch.float32) for _ in range(num_heads)]

    final_rank = max(actual_ranks) if target_rank is None else (target_rank or head_dim)
    
    projections = []
    for h in range(num_heads):
        top_eigenvectors = all_eigenvectors[h][:, :final_rank]
        if pad_to_original_dim:
            V_padded = torch.zeros(head_dim, head_dim, dtype=top_eigenvectors.dtype, device=top_eigenvectors.device)
            V_padded[:, :final_rank] = top_eigenvectors
            T = V_padded.t()
        else:
            T = top_eigenvectors.t()
        projections.append(T)

    return torch.stack(projections, dim=0), final_rank

def absorb_and_compress_layer(
    layer_module,
    projection_matrix: torch.Tensor,
    reduced_rank: int,
    pad_to_original_dim: bool = False,
) -> None:
    """
    Apply PCA-based dimensionality reduction to a DeltaNet2 layer.
    """
    device = layer_module.q_proj.weight.device
    dtype = layer_module.q_proj.weight.dtype
    num_heads = layer_module.num_heads
    T_matrix = projection_matrix.to(device=device, dtype=dtype)
    
    def project_linear_layer(linear_layer, proj_matrix, pad_mode=False):
        w_old = rearrange(linear_layer.weight.data, "(h d) i -> h d i", h=num_heads)
        w_new = torch.einsum("h r d, h d i -> h r i", proj_matrix, w_old)
        w_new = rearrange(w_new, "h r i -> (h r) i")
        
        if pad_mode:
            linear_layer.weight.data = w_new.to(dtype=dtype)
            return linear_layer
        else:
            new_layer = nn.Linear(linear_layer.in_features, num_heads * reduced_rank, bias=False)
            new_layer.weight.data = w_new.to(dtype=dtype)
            return new_layer.to(device=device)

    layer_module.q_proj = project_linear_layer(layer_module.q_proj, T_matrix, pad_mode=pad_to_original_dim)
    layer_module.k_proj = project_linear_layer(layer_module.k_proj, T_matrix, pad_mode=pad_to_original_dim)
    
    if layer_module.use_short_conv and not pad_to_original_dim:
        layer_module.q_conv1d.resize(num_heads * reduced_rank, device=device, dtype=dtype)
        layer_module.k_conv1d.resize(num_heads * reduced_rank, device=device, dtype=dtype)
    
    if not pad_to_original_dim:
        layer_module.key_dim = num_heads * reduced_rank
        layer_module.head_k_dim = reduced_rank

def absorb_and_compress_deltanet_nonshared(
    layer_module,
    projection_matrix: torch.Tensor,
    reduced_rank: int,
    pad_to_original_dim: bool = False,
):
    """
    Apply PCA-based dimensionality reduction to a DeltaNet layer with non-shared convolutions.
    """
    device = layer_module.q_proj.weight.device
    dtype = layer_module.q_proj.weight.dtype
    num_heads = layer_module.num_heads
    T_matrix = projection_matrix.to(device=device, dtype=dtype)
    
    new_total_key_dim = num_heads * reduced_rank

    def project_linear_layer(linear_layer, proj_matrix):
        w_old = rearrange(linear_layer.weight.data, '(h d) i -> h d i', h=num_heads)
        w_new = torch.matmul(proj_matrix, w_old)
        new_layer = nn.Linear(linear_layer.in_features, new_total_key_dim, bias=False).to(device=device, dtype=dtype)
        new_layer.weight.data.copy_(rearrange(w_new, 'h r i -> (h r) i'))
        return new_layer

    layer_module.q_proj = project_linear_layer(layer_module.q_proj, T_matrix)
    layer_module.k_proj = project_linear_layer(layer_module.k_proj, T_matrix)

    def transform_conv_weights(old_conv_layer, mix_weights):
        kernel_size = old_conv_layer.kernel_size[0] if isinstance(old_conv_layer.kernel_size, tuple) else old_conv_layer.kernel_size
        new_conv = ShortConvolution(hidden_size=new_total_key_dim, kernel_size=kernel_size, activation=old_conv_layer.activation).to(device=device, dtype=dtype)
        w_old = rearrange(old_conv_layer.weight.data, '(h d) 1 k -> h d k', h=num_heads)
        w_new = torch.einsum('h r d, h d k -> h r k', mix_weights, w_old)
        new_conv.weight.data.copy_(rearrange(w_new, 'h r k -> (h r) 1 k'))
        return new_conv

    mixing_weights = T_matrix ** 2
    if hasattr(layer_module, 'q_conv1d') and hasattr(layer_module, 'k_conv1d'):
        layer_module.q_conv1d = transform_conv_weights(layer_module.q_conv1d, mixing_weights)
        layer_module.k_conv1d = transform_conv_weights(layer_module.k_conv1d, mixing_weights)

    if not pad_to_original_dim:
        layer_module.head_k_dim = reduced_rank
        layer_module.key_dim = new_total_key_dim
        layer_module.conv_key_dim = layer_module.key_dim

def collect_layer_statistics(model, dataloader, device: str, num_layers: int):
    """
    Collect key statistics from model layers during forward passes.
    """
    layer_stats = { i: { "k_sum": None, "k_outer": None, "q_sum": None, "q_outer": None, "n": 0 } for i in range(num_layers) }
    hooks = []

    if hasattr(model, "model") and hasattr(model.model, "layers"): layers = model.model.layers
    elif hasattr(model, "layers"): layers = model.layers
    else: raise ValueError("Cannot find layers in model structure")

    def get_hook(layer_idx, module):
        def hook(mod, args, kwargs):
            x = args[0] if len(args) > 0 else kwargs.get("hidden_states")
            if x is None: return
            q, k = mod.q_proj(x), mod.k_proj(x)
            if hasattr(mod, "use_short_conv") and mod.use_short_conv:
                q, _ = mod.q_conv1d(x=q, cache=None, output_final_state=False, cu_seqlens=kwargs.get('cu_seqlens'))
                k, _ = mod.k_conv1d(x=k, cache=None, output_final_state=False, cu_seqlens=kwargs.get('cu_seqlens'))
            elif hasattr(mod, "qk_activation"):
                act = getattr(nn.functional, mod.qk_activation, lambda x: x)
                q, k = act(q), act(k)
            q = l2_norm(rearrange(q, "... (h d) -> ... h d", h=mod.num_heads)).flatten(0, 1).float()
            k = l2_norm(rearrange(k, "... (h d) -> ... h d", h=mod.num_heads)).flatten(0, 1).float()
            n_batch = q.shape[0]
            if layer_stats[layer_idx]["k_sum"] is None:
                H, D = q.shape[1], q.shape[2]
                layer_stats[layer_idx].update({
                    "k_sum": torch.zeros((H, D)), "k_outer": torch.zeros((H, D, D)),
                    "q_sum": torch.zeros((H, D)), "q_outer": torch.zeros((H, D, D))
                })
            layer_stats[layer_idx]["n"] += n_batch
            layer_stats[layer_idx]["k_sum"] += k.sum(dim=0).cpu()
            layer_stats[layer_idx]["q_sum"] += q.sum(dim=0).cpu()
            layer_stats[layer_idx]["k_outer"] += torch.einsum("thd,the->hde", k, k).cpu()
            layer_stats[layer_idx]["q_outer"] += torch.einsum("thd,the->hde", q, q).cpu()
        return hook

    for i, layer in enumerate(layers):
        target = layer.attn if hasattr(layer, "attn") else (layer.mixer if hasattr(layer, "mixer") else layer)
        hooks.append(target.register_forward_pre_hook(get_hook(i, target), with_kwargs=True))

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting statistics"):
            batch = {k: v.to(device) for k, v in batch.items()}
            model(**batch)
    for h in hooks: h.remove()
    return layer_stats

# ==============================================================================
# MAIN SCRIPT LOGIC
# ==============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="DeltaNet PCA Pruner")
    parser.add_argument("--model_path", type=str, required=True, help="Path to pre-trained model")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save pruned model")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu", help="Calibration dataset")
    parser.add_argument("--dataset_name", type=str, default="sample-10BT")
    parser.add_argument("--n_calibration_samples", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--target_rank", type=int, default=None, help="Target rank per head")
    parser.add_argument("--variance_threshold", type=float, default=0.95)
    parser.add_argument("--compression_mode", type=str, default="pca", choices=["pca", "random", "trivial"])
    parser.add_argument("--include_queries", action="store_true", help="Include Q in PCA statistics")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading model from {args.model_path}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
    tokenizer = AutoTokenizer.from_pretrained("fla-hub/transformer-1.3B-100B", trust_remote_code=True)
    
    layers = model.model.layers if hasattr(model, "model") else model.layers
    num_layers = len(layers)

    # 1. Collect Statistics
    print(f"Building calibration dataset: {args.dataset}")
    dataset = build_dataset(dataset=args.dataset, dataset_name=args.dataset_name, dataset_split="train", num_workers=4, seed=args.seed)
    dataloader = build_dataloader(dataset=dataset, tokenizer=tokenizer, batch_size=args.batch_size, seq_len=args.seq_len, num_workers=4)
    
    limited_dataloader = []
    for i, batch in enumerate(dataloader):
        if i >= args.n_calibration_samples // args.batch_size: break
        limited_dataloader.append(batch)
    
    print("Collecting statistics...")
    layer_stats = collect_layer_statistics(model=model, dataloader=limited_dataloader, device=device, num_layers=num_layers)

    # 2. Compute Projections and Compress
    print("Compressing layers...")
    for i, layer in enumerate(layers):
        if not hasattr(layer, "attn"): continue
        target_module = layer.attn
        stats = layer_stats[i]
        pca_k_sum, pca_k_outer, pca_n = stats["k_sum"], stats["k_outer"], stats["n"]
        if args.include_queries:
            pca_k_sum = (pca_k_sum + stats["q_sum"]) / 2
            pca_k_outer = (pca_k_outer + stats["q_outer"]) / 2
            pca_n = 2 * pca_n

        T, reduced_rank = compute_pca_projection(k_states=None, target_rank=args.target_rank, head_dim=target_module.head_k_dim, num_heads=target_module.num_heads, variance_threshold=args.variance_threshold, compression_mode=args.compression_mode, k_sum=pca_k_sum, k_outer=pca_k_outer, n=pca_n)

        class_name = type(target_module).__name__
        if class_name == "DeltaNet2": absorb_and_compress_layer(target_module, T, reduced_rank)
        elif class_name in ["DeltaNet", "Attention", "GatedDeltaNet"]: absorb_and_compress_deltanet_nonshared(target_module, T, reduced_rank)
        print(f"Layer {i}: New head_k_dim = {reduced_rank}")

    # 3. Update Config
    print("Updating model config...")
    first_attn = layers[0].attn
    if model.config.model_type == 'gated_deltanet':
        model.config.head_dim = first_attn.head_k_dim
    else:
        if hasattr(model.config, "expand_k"):
            if isinstance(model.config.expand_k, dict):
                model.config.expand_k = {str(i): l.attn.key_dim / l.attn.hidden_size for i, l in enumerate(layers) if hasattr(l, "attn")}
            else:
                model.config.expand_k = first_attn.key_dim / first_attn.hidden_size
        if hasattr(model.config, "head_dim"):
            model.config.head_dim = first_attn.head_k_dim

    # 4. Save
    print(f"Saving to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")

if __name__ == "__main__":
    main()
