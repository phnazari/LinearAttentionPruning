"""
Compute and visualize rank utilization for DeltaNet models.

This script analyzes the rank utilization of recurrent states during forward passes.
Rank utilization is defined as u(S) = sr(S) / min(d,d') where sr(S) = ||S||_F^2 / ||S||_2^2.
It works by:
1. Loading a DeltaNet model from a checkpoint
2. Running forward passes on calibration data
3. Collecting activations (q, k, v, beta) from attention layers
4. Computing rank utilization evolution during sequence processing
5. Visualizing the results

Usage:
    python effective_state_rank.py \
        --model_path /path/to/model \
        --dataset HuggingFaceFW/fineweb-edu \
        --dataset_name sample-10BT \
        --n_samples 128 \
        --batch_size 4 \
        --seq_len 4096 \
        --train_context_len 4096
"""

import argparse
import logging
import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm, trange
from transformers import AutoModelForCausalLM, AutoTokenizer

# Add necessary paths for imports
script_dir = Path(__file__).resolve().parent
repo_root = script_dir.parent.parent
# We rely on PYTHONPATH for submodules like flare and flash-linear-attention

# Import fla modules
try:
    import fla  # noqa: register models
    from fla.modules.l2norm import l2norm
    from fla.ops.delta_rule.chunk import chunk_delta_rule
except ImportError as e:
    print(f"Warning: Could not import fla modules: {e}")
    print("Some functionality may be limited.")

# Import custom models
try:
    import flame.custom_models.delta_net_2  # noqa: register DeltaNet2
except ImportError:
    print("Warning: Could not import flame.custom_models. DeltaNet2 models may not be available.")

# Import data utilities
try:
    from flame.data import build_dataloader, build_dataset
except ImportError:
    print("Warning: Could not import flame.data. Dataset loading may not work.")


def delta_rule_recurrence(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = True,
    paninetto: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[list[float]]]:
    q, k, v, beta = map(lambda x: x.transpose(1, 2), [q, k, v, beta])

    orig_dtype = q.dtype
    b, h, l, d_k = q.shape  # noqa
    q, k, v, beta = map(lambda x: x.float(), [q, k, v, beta])
    d_v = v.shape[-1]
    o = torch.zeros_like(v)
    S = torch.zeros(b, h, d_k, d_v).to(v)
    q = q * (d_k**-0.5)

    if beta.ndim < v.ndim:
        beta = beta[..., None]

    if initial_state is not None:
        S += initial_state

    rank_utilization_list = []
    for i in trange(l):
        # Check for numerical issues in state matrix
        if not torch.isfinite(S).all():
            logger.warning(f"Non-finite values detected in state matrix at position {i}")
        
        rank_utilization_list.append(
            [
                rank_utilization(S[0, head_idx].cpu().numpy())
                for head_idx in range(S.shape[1])
            ]
        )
        _k = k[:, :, i]
        _q = q[:, :, i]
        _v = v[:, :, i].clone()
        beta_i = beta[:, :, i]
        _v = _v - (S.clone() * _k[..., None]).sum(-2)
        _v = _v * beta_i
        S = S.clone() + _k.unsqueeze(-1) * _v.unsqueeze(-2)
        o[:, :, i] = torch.einsum(
            "bhd,bhdm->bhm", _q, S.transpose(-1, -2) if paninetto else S
        )
    return (
        o.to(orig_dtype).transpose(1, 2),
        (None if output_final_state is False else S),
        rank_utilization_list,
    )  # type: ignore


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale

    rank_utilization_list = []
    for i in trange(T):
        rank_utilization_list.append(
            [
                rank_utilization(h[0, head_idx].cpu().numpy())
                for head_idx in range(h.shape[1])
            ]
        )

        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)

    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h, rank_utilization_list


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v

    attn = -(k_beta @ k.transpose(-1, -2)).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumdecay = attn @ k_beta
    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i] * decay[:, :, i, :, None].exp()) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = (
            S * decay[:, :, i, -1, None, None].exp()
            + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, "b h n c d -> b h (n c) d")
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


def collect_activations(
    model,
    dataloader,
    device: str,
    num_layers: int,
    logger,
) -> dict:
    """
    Collect activations (q, k, v, beta) from model layers during forward passes.
    
    Args:
        model: The model to collect activations from
        dataloader: DataLoader providing batches
        device: Device to run on
        num_layers: Number of layers in the model
        logger: Logger for progress messages
    
    Returns:
        Dictionary with keys like 'model.layers.{layer}.attn.{q,k,v,beta}_id'
        containing collected activation tensors
    """
    model.eval()
    
    # Storage for activations
    activations = {}
    for layer_idx in range(num_layers):
        activations[f"model.layers.{layer_idx}.attn.q_id"] = []
        activations[f"model.layers.{layer_idx}.attn.k_id"] = []
        activations[f"model.layers.{layer_idx}.attn.v_id"] = []
        activations[f"model.layers.{layer_idx}.attn.beta_id"] = []
    
    # Get model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot find layers in model structure")
    
    # Register hooks to capture activations
    handles = []
    
    def make_hook(layer_idx, activation_type):
        def hook(module, input, output):
            # Output is a tuple for conv layers: (output_tensor, cache)
            if isinstance(output, tuple):
                output = output[0]
            # Store on CPU to save GPU memory
            activations[f"model.layers.{layer_idx}.attn.{activation_type}_id"].append(
                output.detach().cpu()
            )
        return hook
    
    # Register hooks for each layer
    for layer_idx, layer in enumerate(layers):
        if not hasattr(layer, "attn"):
            continue
        
        attn = layer.attn
        
        # Hook after convolutions (if present) or projections
        if hasattr(attn, "q_conv1d"):
            handles.append(attn.q_conv1d.register_forward_hook(make_hook(layer_idx, "q")))
            handles.append(attn.k_conv1d.register_forward_hook(make_hook(layer_idx, "k")))
            if hasattr(attn, "v_conv1d"):
                handles.append(attn.v_conv1d.register_forward_hook(make_hook(layer_idx, "v")))
            elif hasattr(attn, "share_kv_conv") and attn.share_kv_conv:
                # For shared conv, we need to hook v_proj instead
                handles.append(attn.v_proj.register_forward_hook(make_hook(layer_idx, "v")))
        else:
            handles.append(attn.q_proj.register_forward_hook(make_hook(layer_idx, "q")))
            handles.append(attn.k_proj.register_forward_hook(make_hook(layer_idx, "k")))
            handles.append(attn.v_proj.register_forward_hook(make_hook(layer_idx, "v")))
        
        # Hook for beta
        if hasattr(attn, "b_proj"):
            handles.append(attn.b_proj.register_forward_hook(make_hook(layer_idx, "beta")))
    
    # Run forward passes
    logger.info(f"Collecting activations from {len(dataloader)} batches...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Collecting activations")):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            
            # Forward pass
            try:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    # Concatenate collected activations
    logger.info("Concatenating activations...")
    has_any_activations = False
    for key in activations:
        if activations[key]:
            activations[key] = torch.cat(activations[key], dim=0)
            has_any_activations = True
        else:
            logger.warning(f"No activations collected for {key}")
    
    # Check if any activations were collected
    if not has_any_activations:
        logger.error("No activations were collected! Check your dataset and batch configuration.")
        logger.error(f"Batches processed: {len(dataloader)}")
        logger.error(f"n_samples={args.n_samples}, batch_size={args.batch_size}")
        sys.exit(1)
    
    return activations


def rank_utilization(matrix):
    """
    Computes the rank utilization of a matrix based on stable rank.
    
    Rank utilization is defined as:
        u(S) = sr(S) / min(d, d')
    where stable rank sr(S) = ||S||_F^2 / ||S||_2^2

    Args:
        matrix (numpy.ndarray): The input matrix with shape (d, d').

    Returns:
        float: The rank utilization. Returns NaN if computation fails or matrix is zero.
    """
    # Check for NaN or Inf values
    if not np.isfinite(matrix).all():
        return np.nan
    
    # Get matrix dimensions
    d, d_prime = matrix.shape
    max_rank = min(d, d_prime)
    
    # Compute Frobenius norm: ||A||_F = sqrt(sum of squared elements)
    frobenius_norm = np.linalg.norm(matrix, ord='fro')
    
    if frobenius_norm == 0:
        return 0.0  # Zero matrix has rank utilization of 0
    
    # Compute spectral norm: ||A||_2 = largest singular value
    try:
        spectral_norm = np.linalg.norm(matrix, ord=2)
    except np.linalg.LinAlgError:
        return np.nan
    
    if spectral_norm == 0:
        return 0.0
    
    # Compute stable rank: sr(A) = ||A||_F^2 / ||A||_2^2
    stable_rank = (frobenius_norm ** 2) / (spectral_norm ** 2)
    
    # Compute rank utilization: u(S) = sr(S) / min(d, d')
    rank_util = stable_rank / max_rank
    
    return rank_util


if __name__ == "__main__":
    import logging

    # Find repository root (script is in scripts/eval/)
    repo_root = Path(__file__).resolve().parent.parent.parent

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("main")
    
    parser = argparse.ArgumentParser(
        description="Compute rank utilization for DeltaNet models"
    )
    
    # Model and data arguments
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to model checkpoint (HuggingFace format)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="HuggingFaceFW/fineweb-edu",
        help="Dataset to use for calibration",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="sample-10BT",
        help="Dataset configuration name",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=128,
        help="Number of samples to process for activation collection",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=2048,
        help="Sequence length for processing",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of workers for data loading",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--train_context_len",
        type=int,
        default=2048,
        help="Training context length for visualization",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: repo_root/outputs)",
    )
    parser.add_argument(
        "--first_layer_only",
        action="store_true",
        help="Only process the first layer (for debugging)",
    )
    args = parser.parse_args()

    # Set output directory
    if args.output_dir is None:
        args.output_dir = repo_root / "outputs"
    else:
        args.output_dir = Path(args.output_dir)
    
    # Extract model name for subfolder
    model_path = str(Path(args.model_path).resolve()).rstrip("/")
    if os.path.basename(model_path) == "checkpoints":
        # e.g., /fast/.../delta_net_2_head/340m/10BT/checkpoints -> delta_net_2_head/340m/10BT
        parent = os.path.dirname(model_path)
        model_subfolder = os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(parent))),
            os.path.basename(os.path.dirname(parent)),
            os.path.basename(parent)
        )
    else:
        model_subfolder = os.path.basename(model_path)
    
    # Create model-specific output directory
    model_output_dir = args.output_dir / model_subfolder
    ranks_dir = model_output_dir / "ranks"
    plots_dir = model_output_dir / "plots"
    ranks_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Output directory: {model_output_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("=" * 80)
    logger.info("Loading model and collecting activations")
    logger.info("=" * 80)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            "fla-hub/transformer-1.3B-100B",
            trust_remote_code=True,
        )
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        sys.exit(1)
    
    # Load model
    logger.info(f"Loading model from {args.model_path}...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to(device)
        model.eval()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)
    
    # Get model info
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        logger.error("Cannot find layers in model structure")
        sys.exit(1)
    
    num_layers = len(layers)
    logger.info(f"Model has {num_layers} layers")
    
    # Build dataset
    logger.info(f"Building dataset from {args.dataset}...")
    try:
        dataset = build_dataset(
            dataset=args.dataset,
            dataset_name=args.dataset_name,
            dataset_split="train",
            streaming=False,
            dp_degree=1,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    except Exception as e:
        logger.error(f"Failed to build dataset: {e}")
        sys.exit(1)
    
    # Build dataloader
    logger.info("Building dataloader...")
    try:
        dataloader = build_dataloader(
            dataset=dataset,
            tokenizer=tokenizer,
            rank=0,
            world_size=1,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=False,
        )
    except Exception as e:
        logger.error(f"Failed to build dataloader: {e}")
        sys.exit(1)
    
    # Limit the number of batches (ensure at least 1 batch)
    limited_dataloader = []
    max_batches = max(1, args.n_samples // args.batch_size)
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break
        limited_dataloader.append(batch)
    
    logger.info(f"Using {len(limited_dataloader)} batches ({len(limited_dataloader) * args.batch_size} samples)")
    
    # Collect activations
    data = collect_activations(
        model=model,
        dataloader=limited_dataloader,
        device=device,
        num_layers=num_layers,
        logger=logger,
    )
    
    # Set metadata for file naming
    dataset_name = args.dataset_name
    seq_len = args.seq_len
    model_name = Path(args.model_path).name
    
    logger.info(f"Collected activations from {model_name}")
    logger.info(f"  Dataset: {dataset_name}")
    logger.info(f"  Sequence length: {seq_len}")
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()
    
    # Process layers
    logger.info("\n" + "=" * 80)
    logger.info("Processing layers...")
    logger.info("=" * 80)

    # Determine which layers to process
    layers_to_process = range(1) if args.first_layer_only else range(num_layers)
    if args.first_layer_only:
        logger.info("First layer only mode enabled (debugging)")

    # Store rank utilization lists for all layers
    all_rank_utilization = {}

    for layer in layers_to_process:
        # Check if results already exist
        rank_file = ranks_dir / f"rank_utilization_list_{dataset_name}_{seq_len}_layer_{layer}_{model_name}.pt"
        if rank_file.exists():
            # Print colored warning
            print(f"\033[93mWARNING: Layer {layer}: Loading existing rank utilization from {rank_file}\033[0m")
            logger.warning(f"Layer {layer}: Loading existing results instead of recomputing")
            try:
                rank_utilization_list = torch.load(rank_file, weights_only=False)
                all_rank_utilization[layer] = rank_utilization_list
                continue  # Only skip recomputing if load succeeded
            except Exception as e:
                logger.error(f"Layer {layer}: Failed to load existing results: {e}")
                logger.info(f"Layer {layer}: Will recompute due to load failure")
            
        logger.info(f"Processing layer {layer}/{num_layers-1}...")
        
        # Load activations for this layer
        try:
            ks = data[f"model.layers.{layer}.attn.k_id"]
            vs = data[f"model.layers.{layer}.attn.v_id"]
            qs = data[f"model.layers.{layer}.attn.q_id"]
            betas = data[f"model.layers.{layer}.attn.beta_id"]
            
            # Check if activations are empty
            if isinstance(ks, list) or isinstance(vs, list) or isinstance(qs, list) or isinstance(betas, list):
                logger.error(f"Layer {layer}: Activations are empty lists, not tensors")
                continue
            
            ks = ks.to(device)
            vs = vs.to(device)
            qs = qs.to(device)
            betas = betas.to(device)
        except KeyError as e:
            logger.error(f"Layer {layer}: Missing activation key: {e}")
            continue

        with torch.autocast(
            device_type=("cuda" if device == "cuda" else "cpu"), dtype=torch.bfloat16
        ):
            logger.info(f"  qs.shape (before reshape): {qs.shape}")
            logger.info(f"  ks.shape (before reshape): {ks.shape}")
            logger.info(f"  vs.shape (before reshape): {vs.shape}")
            logger.info(f"  betas.shape: {betas.shape}")
            
            # Get number of heads from beta shape
            num_heads = betas.shape[-1]
            
            # Reshape q, k, v to include head dimension: [batch, seq_len, features] -> [batch, seq_len, heads, dim_per_head]
            batch_size, seq_len, qk_features = qs.shape
            qk_head_dim = qk_features // num_heads
            
            # vs may have different feature dimension than qs/ks
            v_features = vs.shape[-1]
            v_head_dim = v_features // num_heads
            
            qs = qs.view(batch_size, seq_len, num_heads, qk_head_dim)
            ks = ks.view(batch_size, seq_len, num_heads, qk_head_dim)
            vs = vs.view(batch_size, seq_len, num_heads, v_head_dim)
            
            # Apply sigmoid to beta (b_proj output needs to be converted to [0,1])
            betas = betas.sigmoid()
            
            ks = l2norm(ks)
            qs = l2norm(qs)

            logger.info(f"  qs.shape (after reshape): {qs.shape}")
            logger.info(f"  ks.shape (after reshape): {ks.shape}")
            logger.info(f"  vs.shape (after reshape): {vs.shape}")
            logger.info("  Computing rank utilization with delta rule recurrence...")
            o, S, rank_utilization_list = delta_rule_recurrence(
                q=qs,
                k=ks,
                v=vs,
                beta=betas,
                initial_state=None,
                output_final_state=True,
            )
            
            # Save rank utilization list
            logger.info(f"  Saving results to: {rank_file}")
            torch.save(rank_utilization_list, rank_file)

            # Optional consistency check with chunked implementation
            try:
                with torch.autocast(
                    device_type=("cuda" if device == "cuda" else "cpu"),
                    dtype=torch.bfloat16,
                ):
                    o_chunk, S_chunk = chunk_delta_rule(
                        q=qs.to(torch.bfloat16),
                        k=ks.to(torch.bfloat16),
                        v=vs.to(torch.bfloat16),
                        beta=betas.to(torch.bfloat16),
                        initial_state=None,
                        output_final_state=True,
                    )
                logger.info(
                    f"  (S - S_chunk).abs().mean(): {(S - S_chunk.float()).abs().mean().item():.4f}"
                )
            except Exception as e:
                logger.warning(f"  Chunk delta rule check failed: {e}")
            
            # Store rank utilization list for this layer
            all_rank_utilization[layer] = rank_utilization_list
    
    # Create combined visualization with all layers
    logger.info("\n" + "=" * 80)
    logger.info("Creating combined visualization...")
    logger.info("=" * 80)
    
    if all_rank_utilization:
        from matplotlib.gridspec import GridSpec
        
        num_layers_processed = len(all_rank_utilization)
        layer_indices = sorted(all_rank_utilization.keys())
        
        # Calculate global y-axis limits across all layers
        global_min = float('inf')
        global_max = float('-inf')
        for layer_idx in layer_indices:
            rank_utilization_list = all_rank_utilization[layer_idx]
            num_heads = len(rank_utilization_list[0])
            seq_length = len(rank_utilization_list)
            for head in range(num_heads):
                values = [rank_utilization_list[j][head] for j in range(seq_length)]
                global_min = min(global_min, min(values))
                global_max = max(global_max, max(values))
        
        # Add some padding to the limits
        y_range = global_max - global_min
        global_min -= 0.05 * y_range
        global_max += 0.05 * y_range
        
        # Split layers into two columns for compact layout
        half = (num_layers_processed + 1) // 2
        num_rows = half
        
        fig = plt.figure(figsize=(16, 3 * num_rows))
        gs = GridSpec(num_rows, 3, figure=fig, width_ratios=[1, 0.1, 1], wspace=0.3, hspace=0.4)
        
        # Create axes manually
        axes = np.empty((num_rows, 2), dtype=object)
        for row in range(num_rows):
            axes[row, 0] = fig.add_subplot(gs[row, 0])
            axes[row, 1] = fig.add_subplot(gs[row, 2])
        
        # Create a legend axis in the middle column of first row only
        legend_ax = fig.add_subplot(gs[0, 1])
        legend_ax.axis('off')
        
        # Store handles and labels from first plot for legend
        legend_handles = []
        legend_labels = []
        
        for i, layer_idx in enumerate(layer_indices):
            rank_utilization_list = all_rank_utilization[layer_idx]
            num_heads = len(rank_utilization_list[0])
            seq_length = len(rank_utilization_list)
            
            # Determine which half (left or right) and which row
            if i < half:
                row_idx = i
                col_idx = 0
            else:
                row_idx = i - half
                col_idx = 1
            
            ax = axes[row_idx, col_idx]
            
            # Use a colormap for smooth color progression across heads
            colors = plt.cm.viridis(np.linspace(0, 1, num_heads))
            
            for head in range(num_heads):
                line, = ax.plot(
                    range(seq_length),
                    [rank_utilization_list[j][head] for j in range(seq_length)],
                    label=f"Head {head}",
                    linewidth=0.8,
                    color=colors[head],
                )
                # Collect handles and labels from first plot
                if i == 0:
                    legend_handles.append(line)
                    legend_labels.append(f"Head {head}")

            # Mark document boundaries if available
            if "s_start_idx" in data:
                for idx, token_idx in enumerate(data["s_start_idx"]):
                    ax.axvline(
                        x=token_idx,
                        color="red",
                        linestyle="--",
                        alpha=0.5,
                        label="BOS" if idx == 0 else "",
                    )

            # Only add labels to first subplot
            if i == 0:
                ax.set_xlabel("Sequence Position")
                ax.set_ylabel("Rank Utilization")
            
            # Set the same y-axis limits for all subplots
            ax.set_ylim(global_min, global_max)
            
            ax.set_title(f"Layer {layer_idx}")
            ax.grid(True, alpha=0.3)
        
        # Add legend to middle space
        if legend_handles:
            legend_ax.legend(legend_handles, legend_labels, fontsize=6, loc="center")
        
        # Hide empty axes if odd number of layers
        if num_layers_processed % 2 == 1:
            axes[num_rows - 1, 1].axis('off')
        
        # plt.suptitle(f"Rank Utilization - {dataset_name} (len={seq_len})", fontsize=14, y=0.995)
        
        # Save combined plot
        plot_file = plots_dir / f"rank_utilization_all_layers_{dataset_name}_{seq_len}_{model_name}.png"
        logger.info(f"Saving combined plot to: {plot_file}")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create averaged plot across all heads and layers
        logger.info("Creating averaged plot across all heads and layers...")
        
        # Collect all rank utilization values at each sequence position
        seq_length = len(all_rank_utilization[layer_indices[0]])
        all_values_per_position = [[] for _ in range(seq_length)]
        
        for layer_idx in layer_indices:
            rank_utilization_list = all_rank_utilization[layer_idx]
            num_heads = len(rank_utilization_list[0])
            for pos in range(seq_length):
                for head in range(num_heads):
                    all_values_per_position[pos].append(rank_utilization_list[pos][head])
        
        # Compute mean and std at each position
        mean_values = [np.mean(values) for values in all_values_per_position]
        std_values = [np.std(values) for values in all_values_per_position]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        positions = np.arange(seq_length)
        ax.plot(positions, mean_values, color='blue', linewidth=2, label='Mean')
        ax.fill_between(
            positions,
            np.array(mean_values) - np.array(std_values),
            np.array(mean_values) + np.array(std_values),
            alpha=0.3,
            color='blue',
            label='±1 Std Dev'
        )
        
        ax.set_xlabel("Sequence Position")
        ax.set_ylabel("Rank Utilization")
        ax.set_title("Average Rank Utilization Across All Heads and Layers")
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Save averaged plot
        avg_plot_file = plots_dir / f"rank_utilization_averaged_{dataset_name}_{seq_len}_{model_name}.png"
        logger.info(f"Saving averaged plot to: {avg_plot_file}")
        plt.savefig(avg_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info("\nProcessing complete!")
    logger.info(f"Results saved to: {ranks_dir}")
    logger.info(f"Plots saved to: {plots_dir}")
