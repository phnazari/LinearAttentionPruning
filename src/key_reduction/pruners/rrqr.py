#!/usr/bin/env python3
"""
Apply Strong RRQR (Algorithm 5) Activation Pruning to DeltaNet
==============================================================

Corrections:
1. Implements full Strong RRQR swap condition (Gu & Eisenstat, 1996).
   Checks metric: sqrt((A_inv*B)^2 + (gamma_C / omega_A)^2) > f.
2. Includes --permute_columns flag for randomized tie-breaking.
3. Fixed UnboundLocalError for new_head_v_dim.
"""

import argparse
import os
import sys
import random
import torch
import math
import numpy as np
import scipy.linalg
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

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

class ActivationCapturer:
    def __init__(self, model, layers_to_hook):
        self.activations = {}
        self.hooks = []
        for layer_idx, proj_name in layers_to_hook:
            layer = model.model.layers[layer_idx]
            if hasattr(layer, "attn") and hasattr(layer.attn, proj_name):
                module = getattr(layer.attn, proj_name)
                hook = module.register_forward_hook(self.get_hook(layer_idx, proj_name))
                self.hooks.append(hook)
                
    def get_hook(self, layer_idx, proj_name):
        def hook_fn(module, input, output):
            key = (layer_idx, proj_name)
            if key not in self.activations: self.activations[key] = []
            
            # Helper: Handle tuple outputs (e.g. from ShortConvolution which returns (output, cache))
            if isinstance(output, tuple):
                o = output[0]
            else:
                o = output

            self.activations[key].append(o.detach().cpu())
        return hook_fn

    def remove_hooks(self):
        for hook in self.hooks: hook.remove()
            
    def get_concatenated_activations(self, layer_idx, proj_name):
        key = (layer_idx, proj_name)
        if key not in self.activations or not self.activations[key]: return None
        concat = torch.cat(self.activations[key], dim=0)
        return concat.view(-1, concat.shape[-1])

    def clear(self):
        self.activations = {}

def solve_triangular(A, B):
    """Wrapper around scipy.linalg.solve_triangular."""
    return scipy.linalg.solve_triangular(A, B, lower=False)

def strong_rrqr_indices(activation_tensor, n_keep, f_param=1.5, max_swaps=20):
    """
    Implements Strong RRQR logic (Gu & Eisenstat, 1996, Algorithm 4/5).
    
    Metric for swapping (i, j):
       rho_{ij} = sqrt( (A_k^{-1} B_k)_{ij}^2 + (gamma_j(C_k) / omega_i(A_k))^2 )
       
    Where:
       omega_i(A_k) = 1 / || (A_k^{-1})_i ||_2  (reciprocal of row norm of inverse)
       gamma_j(C_k) = || (C_k)_j ||_2           (column norm of residual)
    """
    # 1. Handle Input
    if isinstance(activation_tensor, list):
        if torch.is_tensor(activation_tensor[0]):
            weights = torch.cat(activation_tensor, dim=0)
        else:
            weights = np.concatenate(activation_tensor, axis=0)
    else:
        weights = activation_tensor

    M = weights.float().numpy()
    
    print(f"Using {M.shape[0]} samples with dimension {M.shape[1]} for RRQR...")
    # Subsample for speed if N_samples is huge
    N_SAMPLES_THRESHOLD = 5000
    if M.shape[0] > N_SAMPLES_THRESHOLD:
        print(f"using a random subset of {N_SAMPLES_THRESHOLD} samples for RRQR")
        indices = np.random.choice(M.shape[0], N_SAMPLES_THRESHOLD, replace=False)
        M = M[indices]
    
    # 2. Initial QR with Column Pivoting (Algorithm 1)
    try:
        Q, R, P = scipy.linalg.qr(M, pivoting=True, mode='economic')
    except Exception as e:
        print(f"    QRCP failed: {e}")
        exit(1)

    # 3. Strong RRQR Enforcement
    current_P = P.copy()
    num_total_cols = R.shape[1]
    
    for swap_iter in range(max_swaps):
        A_k = R[:n_keep, :n_keep]
        B_k = R[:n_keep, n_keep:]
        
        # Check for singularity of A_k
        if np.abs(np.diag(A_k)).min() < 1e-9:
            print("    Singularity detected in A_k, stopping swaps.")
            break

        # --- CALCULATE METRICS (Gu & Eisenstat Lemma 3.1) ---
        
        # 1. Linear Dependence Term: W = A_k^{-1} * B_k
        # A_k is upper triangular, so we solve A_k * W = B_k
        W = solve_triangular(A_k, B_k)
        
        # 2. Volume/Norm Terms: gamma / omega
        # We need A_k^{-1} explicitly to get its row norms for omega
        try:
            A_inv = solve_triangular(A_k, np.eye(n_keep))
        except:
            break
            
        # omega_i(A_k) = 1 / || row_i(A_k^{-1}) ||
        A_inv_row_norms = np.linalg.norm(A_inv, axis=1)
        # Avoid div/0
        omega_A = 1.0 / (A_inv_row_norms + 1e-12)
        
        # gamma_j(C_k) = || col_j(C_k) ||
        # C_k is the bottom-right block. 
        # R is typically (min(M,N), N). If M > N (tall), R is (N,N).
        if R.shape[0] > n_keep and R.shape[1] > n_keep:
            C_k = R[n_keep:, n_keep:]
            gamma_C = np.linalg.norm(C_k, axis=0)
        else:
            gamma_C = np.zeros(num_total_cols - n_keep)
            
        # 3. Combine into Rho Metric
        # Term 1: |W_ij|^2
        term1_sq = np.abs(W)**2
        
        # Term 2: (gamma_j / omega_i)^2
        # Use outer product to broadcast: (N_keep, 1) x (1, N_discard) -> (N_keep, N_discard)
        term2_sq = np.outer(1.0 / omega_A, gamma_C)**2
        
        rho = np.sqrt(term1_sq + term2_sq)
        
        max_val = np.max(rho)
        
        if max_val <= f_param:
            # Satisfied Strong RRQR condition
            break
            
        # 4. Perform Swap
        idx_flat = np.argmax(rho)
        i_local, j_local = np.unravel_index(idx_flat, rho.shape)
        
        col_idx_A = i_local        
        col_idx_B = n_keep + j_local 
        
        # Swap indices in permutation vector
        current_P[col_idx_A], current_P[col_idx_B] = current_P[col_idx_B], current_P[col_idx_A]
        
        # Re-factorize (Inefficient but robust implementation)
        M_permuted = M[:, current_P]
        Q_new, R_new = scipy.linalg.qr(M_permuted, mode='economic')
        R = R_new

    return current_P[:n_keep]

def get_rrqr_indices_per_head_activations(activation_tensor, n_keep, num_heads, permute_columns=False):
    """Applies Strong RRQR per head, optionally permuting columns to break ties randomly."""
    if isinstance(activation_tensor, list):
        # Concatenate along sample dimension to avoid sign cancellation during sum
        weights = torch.cat(activation_tensor, dim=0)
    else:
        weights = activation_tensor

    total_samples, hidden_dim = weights.shape
    head_dim = hidden_dim // num_heads
    
    weights_reshaped = weights.view(total_samples, num_heads, head_dim).permute(1, 0, 2)
    keep_indices = []

    print(f"  Computing Strong RRQR (Algo 5) on activations (permute={permute_columns})...")
    
    for h in range(num_heads):
        act_head = weights_reshaped[h] 
        
        if permute_columns:
            # 1. Generate random permutation
            perm = torch.randperm(head_dim)
            # 2. Shuffle columns
            act_head_shuffled = act_head[:, perm]
            # 3. Run RRQR on shuffled matrix
            selected_local_indices_shuffled = strong_rrqr_indices(act_head_shuffled, n_keep, f_param=1.5)
            # 4. Map back to original indices
            # If RRQR picked index 'i' in the shuffled matrix, the real index is perm[i]
            selected_local_indices = perm[selected_local_indices_shuffled].numpy()
        else:
            selected_local_indices = strong_rrqr_indices(act_head, n_keep, f_param=1.5)
            
        global_offset = h * head_dim
        global_indices = selected_local_indices + global_offset
        keep_indices.append(global_indices)

    return np.concatenate(keep_indices)

def prune_model_strong_rrqr(model, dataloader, pruning_ratio, pruning_strategy, num_batches=10, permute_columns=False):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    else:
        layers = model.layers

    device = layers[0].attn.q_proj.weight.device
    print(f"Pruning with Strong RRQR (Algo 5) | Ratio: {pruning_ratio} | Batches: {num_batches} | Permute: {permute_columns}")

    # Hook targets
    hook_targets = []
    for i, layer in enumerate(layers):
        if hasattr(layer, "attn"):
            attn = layer.attn
            if pruning_strategy in ["dimension", "dimension_both"]:
                # Prefer hooking the output of the convolution (input to the delta rule)
                # if it exists, otherwise hook the projection.
                # Q
                if hasattr(attn, "q_conv1d"): hook_targets.append((i, "q_conv1d"))
                else: hook_targets.append((i, "q_proj"))
                
                # K
                if hasattr(attn, "k_conv1d"): hook_targets.append((i, "k_conv1d"))
                else: hook_targets.append((i, "k_proj"))

            if pruning_strategy == "dimension_both":
                # V
                if hasattr(attn, "v_conv1d"): hook_targets.append((i, "v_conv1d"))
                else: hook_targets.append((i, "v_proj"))

    capturer = ActivationCapturer(model, hook_targets)
    
    print("Running calibration forward passes...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            if i >= num_batches: break
            if isinstance(batch, dict):
                inputs = batch["input_ids"].to(device)
            else:
                inputs = batch[0].to(device)
            model(inputs)

    print("Calibrated. Starting Strong RRQR Pruning...")

    for i, layer in enumerate(layers):
        if not hasattr(layer, "attn"): continue
        attn = layer.attn
        num_heads = attn.num_heads
        head_k_dim = attn.head_k_dim
        head_v_dim = getattr(attn, "head_v_dim", attn.value_dim // num_heads)
        
        qk_indices = []
        v_indices = []
        print(f"Layer {i}:")

        if pruning_strategy == "dimension":
            # STRATEGY 1: Reduce Q/K
            new_head_k_dim = math.floor(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            
            # FIX: Initialize new_head_v_dim to unchanged value
            new_head_v_dim = head_v_dim 
            
            # Determine which keys to look up based on what we hooked
            q_key = "q_conv1d" if hasattr(attn, "q_conv1d") else "q_proj"
            k_key = "k_conv1d" if hasattr(attn, "k_conv1d") else "k_proj"
            
            q_act = capturer.get_concatenated_activations(i, q_key)
            k_act = capturer.get_concatenated_activations(i, k_key)
            
            if q_act is not None:
                qk_indices = get_rrqr_indices_per_head_activations(
                    [q_act, k_act], new_head_k_dim, num_heads, permute_columns=permute_columns
                )
            print(f"  QK: {head_k_dim} -> {new_head_k_dim} (V unchanged)")

        elif pruning_strategy == "dimension_both":
            # STRATEGY 2: Reduce Q/K and V
            new_head_k_dim = int(head_k_dim * (1 - pruning_ratio))
            if new_head_k_dim < 1: new_head_k_dim = 1
            
            # Determine keys
            q_key = "q_conv1d" if hasattr(attn, "q_conv1d") else "q_proj"
            k_key = "k_conv1d" if hasattr(attn, "k_conv1d") else "k_proj"
            
            q_act = capturer.get_concatenated_activations(i, q_key)
            k_act = capturer.get_concatenated_activations(i, k_key)
            if q_act is not None:
                qk_indices = get_rrqr_indices_per_head_activations(
                    [q_act, k_act], new_head_k_dim, num_heads, permute_columns=permute_columns
                )

            new_head_v_dim = int(head_v_dim * (1 - pruning_ratio))
            if new_head_v_dim < 1: new_head_v_dim = 1
            
            v_key = "v_conv1d" if hasattr(attn, "v_conv1d") else "v_proj"
            v_act = capturer.get_concatenated_activations(i, v_key)
            if v_act is not None:
                v_indices = get_rrqr_indices_per_head_activations(
                    v_act, new_head_v_dim, num_heads, permute_columns=permute_columns
                )
                
                if hasattr(attn, "o_norm"):
                    v_idx_t = torch.tensor(v_indices, device=device).long()
                    # Handle IndependentNorm vs Shared Norm
                    if hasattr(attn.o_norm, "norms"): # IndependentNorm
                        for h_idx, norm in enumerate(attn.o_norm.norms):
                            # extract the indices for this head
                            head_v_indices = v_indices[h_idx * new_head_v_dim : (h_idx + 1) * new_head_v_dim]
                            # these are global indices, convert to local head-space
                            local_head_v_indices = torch.tensor(head_v_indices % head_v_dim, device=device).long()
                            norm.weight.data = norm.weight.data[local_head_v_indices]
                            if hasattr(norm, "bias") and norm.bias is not None:
                                norm.bias.data = norm.bias.data[local_head_v_indices]
                    else: # Standard Shared Norm
                        # Use local indices for slicing the shared norm parameters.
                        # We use the first head's indices as a heuristic.
                        local_indices = torch.tensor(v_indices[:new_head_v_dim] % head_v_dim, device=device).long()
                        attn.o_norm.weight.data = attn.o_norm.weight.data[local_indices]
                        if hasattr(attn.o_norm, "bias") and attn.o_norm.bias is not None:
                            attn.o_norm.bias.data = attn.o_norm.bias.data[local_indices]

            print(f"  QK: {head_k_dim} -> {new_head_k_dim} | V: {head_v_dim} -> {new_head_v_dim}")
        
        else:
            # Safety fallback
            new_head_k_dim = head_k_dim
            new_head_v_dim = head_v_dim

        # --- SLICING ---
        if len(qk_indices) > 0:
            qk_idx_t = torch.tensor(qk_indices, device=device).long()
            attn.q_proj = slice_linear_layer(attn.q_proj, qk_idx_t, dim=0)
            attn.k_proj = slice_linear_layer(attn.k_proj, qk_idx_t, dim=0)
            if hasattr(attn, "q_conv1d"): 
                attn.q_conv1d = slice_conv_layer(attn.q_conv1d, qk_idx_t)
            if hasattr(attn, "k_conv1d"): 
                attn.k_conv1d = slice_conv_layer(attn.k_conv1d, qk_idx_t)

        if len(v_indices) > 0:
            v_idx_t = torch.tensor(v_indices, device=device).long()
            attn.v_proj = slice_linear_layer(attn.v_proj, v_idx_t, dim=0)
            if hasattr(attn, "v_conv1d"): 
                attn.v_conv1d = slice_conv_layer(attn.v_conv1d, v_idx_t)
            if hasattr(attn, "g_proj") and attn.g_proj is not None:
                attn.g_proj = slice_linear_layer(attn.g_proj, v_idx_t, dim=0)
            attn.o_proj = slice_linear_layer(attn.o_proj, v_idx_t, dim=1)

        # Update Config
        attn.head_k_dim = new_head_k_dim
        if hasattr(attn, "head_v_dim"): attn.head_v_dim = new_head_v_dim
        attn.key_dim = num_heads * new_head_k_dim
        
        # Safe fallback for value_dim
        current_head_v = new_head_v_dim
        attn.value_dim = num_heads * current_head_v

    capturer.remove_hooks()
    capturer.clear()

    # Final Config
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

def slice_conv_layer(layer, indices, head_indices=None):
    if hasattr(layer, "shared_kernel") or hasattr(layer, "r"):
        device = layer.conv.bias.device if layer.conv.bias is not None else next(layer.parameters()).device
        dtype = layer.conv.bias.dtype if layer.conv.bias is not None else next(layer.parameters()).dtype
        new_hidden_size = len(indices)
        layer.resize(new_hidden_size=new_hidden_size, new_num_heads=None, device=device, dtype=dtype)
        return layer
    else:
        # Standard ShortConvolution
        device = layer.weight.device
        dtype = layer.weight.dtype
        new_hid = len(indices)
        kernel = layer.kernel_size[0] if isinstance(layer.kernel_size, tuple) else layer.kernel_size
        
        # Capture old weights and slice before creating new layer
        sliced_weight = layer.weight.data[indices]
        sliced_bias = layer.bias.data[indices] if layer.bias is not None else None
        
        new_layer = ShortConvolution(
            hidden_size=new_hid, 
            kernel_size=kernel, 
            activation=layer.activation,
            bias=layer.bias is not None
        ).to(device=device, dtype=dtype)
        
        new_layer.weight.data = sliced_weight
        if sliced_bias is not None:
            new_layer.bias.data = sliced_bias
        return new_layer

def main():
    parser = argparse.ArgumentParser(description="DeltaNet Strong RRQR Pruner")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pruning_ratio", type=float, default=0.5)
    parser.add_argument("--pruning_strategy", type=str, default="dimension", 
                        choices=["dimension", "dimension_both"])
    parser.add_argument("--permute_columns", action="store_true", help="Shuffle columns before RRQR to randomize tie-breaking")
    parser.add_argument("--dataset", type=str, default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--dataset_name", type=str, default="sample-10BT")
    parser.add_argument("--num_batches", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=111)
    args = parser.parse_args()
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
    model = get_deltanet(args.model_path)
    model.eval()
    
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
    
    model = prune_model_strong_rrqr(
        model=model, dataloader=dataloader,
        pruning_ratio=args.pruning_ratio, pruning_strategy=args.pruning_strategy,
        num_batches=args.num_batches,
        permute_columns=args.permute_columns
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    try: tokenizer.save_pretrained(args.output_dir)
    except: pass
    print(f"Pruned model saved to {args.output_dir}")

if __name__ == "__main__":
    main()