#!/usr/bin/env python3
"""
Plot Convolutional Filters from Model Checkpoints

This script:
1. Loads a pre-trained DeltaNet model (or similar)
2. Extracts convolutional filter weights from each layer
3. Plots and saves visualizations of the filters

Usage:
    python plot_conv_filters.py --model_id fla-hub/delta_net-1.3B-100B \
                                --output_dir ./conv_filter_plots
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch


# Import fla to register DeltaNet and other models with transformers
import fla  # noqa
import custom_models.delta_net_2

from transformers import AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract and plot convolutional filters from model checkpoints"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="fla-hub/delta_net-1.3B-100B",
        help="HuggingFace model ID or path to load",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./conv_filter_plots",
        help="Directory to save filter plots",
    )
    parser.add_argument(
        "--plot_format",
        type=str,
        default="png",
        choices=["png", "pdf", "svg"],
        help="Output format for plots",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots",
    )
    parser.add_argument(
        "--show_plots",
        action="store_true",
        help="Display plots interactively",
    )
    return parser.parse_args()


def extract_conv_filters(model):
    """Extract convolutional filters from all layers of the model.
    
    Returns:
        dict: Mapping of layer_idx -> {
            'q_conv': tensor or None,
            'k_conv': tensor or None,
            'v_conv': tensor or None,
            'shared_kernel_q': tensor or None,  # For SharedKernelConv1d
            'shared_kernel_k': tensor or None,
            'shared_kernel_v': tensor or None,
            'num_heads': int or None,
            'head_k_dim': int or None,
            'head_v_dim': int or None,
        }
    """
    # Get model layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = model.model.layers
    elif hasattr(model, "layers"):
        layers = model.layers
    else:
        raise ValueError("Cannot find layers in model structure")
    
    conv_filters = {}
    
    for i, layer in enumerate(layers):
        layer_filters = {
            'q_conv': None,
            'k_conv': None,
            'v_conv': None,
            'shared_kernel_q': None,
            'shared_kernel_k': None,
            'shared_kernel_v': None,
            'num_heads': None,
            'head_k_dim': None,
            'head_v_dim': None,
        }
        
        # Find the attention/mixer module
        target_module = None
        if hasattr(layer, "attn"):
            target_module = layer.attn
        elif hasattr(layer, "mixer"):
            target_module = layer.mixer
        
        if target_module is None:
            conv_filters[i] = layer_filters
            continue
        
        # Extract num_heads and head dimensions if available
        if hasattr(target_module, "num_heads"):
            layer_filters['num_heads'] = target_module.num_heads
        if hasattr(target_module, "head_k_dim"):
            layer_filters['head_k_dim'] = target_module.head_k_dim
        if hasattr(target_module, "head_v_dim"):
            layer_filters['head_v_dim'] = target_module.head_v_dim
        
        # Extract Q conv filters
        if hasattr(target_module, "q_conv1d"):
            q_conv = target_module.q_conv1d
            if hasattr(q_conv, "shared_kernel"):
                # SharedKernelConv1d - extract the actual shared kernel
                layer_filters['shared_kernel_q'] = q_conv.shared_kernel.detach().cpu()
            elif hasattr(q_conv, "conv") and hasattr(q_conv.conv, "weight"):
                # SharedKernelConv1d without direct shared_kernel access
                layer_filters['q_conv'] = q_conv.conv.weight.detach().cpu()
            elif hasattr(q_conv, "weight"):
                # Regular Conv1d
                layer_filters['q_conv'] = q_conv.weight.detach().cpu()
        
        # Extract K conv filters
        if hasattr(target_module, "k_conv1d"):
            k_conv = target_module.k_conv1d
            if hasattr(k_conv, "shared_kernel"):
                # SharedKernelConv1d - extract the actual shared kernel
                layer_filters['shared_kernel_k'] = k_conv.shared_kernel.detach().cpu()
            elif hasattr(k_conv, "conv") and hasattr(k_conv.conv, "weight"):
                # SharedKernelConv1d without direct shared_kernel access
                layer_filters['k_conv'] = k_conv.conv.weight.detach().cpu()
            elif hasattr(k_conv, "weight"):
                # Regular Conv1d
                layer_filters['k_conv'] = k_conv.weight.detach().cpu()
        
        # Extract V conv filters
        if hasattr(target_module, "v_conv1d"):
            v_conv = target_module.v_conv1d
            if hasattr(v_conv, "shared_kernel"):
                # SharedKernelConv1d - extract the actual shared kernel
                layer_filters['shared_kernel_v'] = v_conv.shared_kernel.detach().cpu()
            elif hasattr(v_conv, "conv") and hasattr(v_conv.conv, "weight"):
                # SharedKernelConv1d without direct shared_kernel access
                layer_filters['v_conv'] = v_conv.conv.weight.detach().cpu()
            elif hasattr(v_conv, "weight"):
                # Regular Conv1d
                layer_filters['v_conv'] = v_conv.weight.detach().cpu()
        
        # Also check for single short_conv (some architectures like Rodimus)
        if hasattr(target_module, "short_conv"):
            short_conv = target_module.short_conv
            if hasattr(short_conv, "weight"):
                layer_filters['short_conv'] = short_conv.weight.detach().cpu()
        
        conv_filters[i] = layer_filters
    
    return conv_filters


def draw_head_boundaries(ax, n_channels, num_heads, head_dim):
    """Draw horizontal red lines at head boundaries."""
    if num_heads is None or num_heads <= 1:
        return
    
    # For shared kernels, n_channels == num_heads, so each row is a head
    # For regular convs, n_channels == num_heads * head_dim
    if n_channels == num_heads:
        # Each row is a head - draw lines between rows
        for h in range(1, num_heads):
            ax.axhline(y=h - 0.5, color='red', linewidth=1.5, linestyle='-')
    elif head_dim is not None and n_channels == num_heads * head_dim:
        # Draw lines every head_dim rows
        for h in range(1, num_heads):
            ax.axhline(y=h * head_dim - 0.5, color='red', linewidth=1.5, linestyle='-')


def plot_filter_weights(conv_filters, output_dir, plot_format="png", dpi=150):
    """Plot all convolutional filter weights in one large figure.
    
    Creates a grid with:
    - Rows: layers
    - Columns: Q, K, V convolutions
    
    All channels are shown (no subsampling).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter to only layers that have conv filters
    layers_with_filters = {
        layer_idx: filters for layer_idx, filters in conv_filters.items()
        if any(v is not None for v in filters.values())
    }
    
    if not layers_with_filters:
        print("No convolutional filters found in model.")
        return
    
    num_layers = len(layers_with_filters)
    layer_indices = sorted(layers_with_filters.keys())
    
    # Collect all filters for statistics plots
    all_q_filters = []
    all_k_filters = []
    all_v_filters = []
    
    # Split layers into two halves for two-column layout
    half = (num_layers + 1) // 2  # ceiling division
    num_rows = half
    
    # Create figure with GridSpec: 3 cols + gap + 3 cols
    fig = plt.figure(figsize=(22, 2.5 * num_rows))
    gs = GridSpec(num_rows, 7, figure=fig, width_ratios=[1, 1, 1, 0.3, 1, 1, 1], wspace=0.3, hspace=0.3)
    
    # Create axes array manually
    axes = np.empty((num_rows, 6), dtype=object)
    for row in range(num_rows):
        for col in range(3):
            axes[row, col] = fig.add_subplot(gs[row, col])
        for col in range(3):
            axes[row, col + 3] = fig.add_subplot(gs[row, col + 4])  # Skip column 3 (gap)
    
    # Column labels for both halves
    for col_idx, label in enumerate(['Query', 'Key', 'Value']):
        axes[0, col_idx].set_title(label, fontsize=12, fontweight='bold')
        axes[0, col_idx + 3].set_title(label, fontsize=12, fontweight='bold')
    
    for i, layer_idx in enumerate(layer_indices):
        filters = layers_with_filters[layer_idx]
        
        # Determine which half (left or right) and which row
        if i < half:
            row_idx = i
            col_offset = 0  # left half
        else:
            row_idx = i - half
            col_offset = 3  # right half
        
        for col_idx, conv_type in enumerate(['q', 'k', 'v']):
            ax = axes[row_idx, col_offset + col_idx]
            
            # Check for shared kernel first, then regular conv
            shared_key = f'shared_kernel_{conv_type}'
            conv_key = f'{conv_type}_conv'
            
            filter_tensor = filters.get(shared_key)
            if filter_tensor is None:
                filter_tensor = filters.get(conv_key)
            
            if filter_tensor is not None:
                filter_np = filter_tensor.float().numpy()
                
                # Handle different shapes
                if filter_np.ndim == 3:
                    # Shape: [channels/heads, 1, kernel_size] or [hidden_size, 1, kernel_size]
                    filter_np = filter_np.squeeze(1)  # -> [channels, kernel_size]
                
                # Flip so that leftmost = current token, rightmost = oldest token
                filter_np = filter_np[:, ::-1]
                
                # Collect for statistics
                if conv_type == 'q':
                    all_q_filters.append((layer_idx, filter_np))
                elif conv_type == 'k':
                    all_k_filters.append((layer_idx, filter_np))
                else:
                    all_v_filters.append((layer_idx, filter_np))
                
                # Plot all channels as heatmap
                im = ax.imshow(filter_np, aspect='auto', cmap='RdBu_r', interpolation='nearest')
                #ax.set_xlabel('Position')
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Draw head boundaries
                n_channels, kernel_size = filter_np.shape
                num_heads = filters.get('num_heads')
                head_dim = filters.get('head_k_dim') if conv_type in ['q', 'k'] else filters.get('head_v_dim')
                draw_head_boundaries(ax, n_channels, num_heads, head_dim)
            else:
                ax.text(0.5, 0.5, 'No filter', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            
            # Row label (layer index) on the left column of each half
            if col_idx == 0:
                ax.set_ylabel(f'Layer {layer_idx}', fontsize=10)
            else:
                ax.set_ylabel('Channel')
    
    # Hide empty axes if odd number of layers
    if num_layers % 2 == 1:
        for col_idx in range(3):
            axes[num_rows - 1, 3 + col_idx].axis('off')
    
    plt.savefig(
        os.path.join(output_dir, f"all_conv_filters.{plot_format}"), 
        dpi=dpi, bbox_inches='tight'
    )
    plt.close()
    
    # Plot normalized versions
    plot_filter_weights_normalized(conv_filters, output_dir, plot_format, dpi)
    plot_filter_weights_normalized_abs(conv_filters, output_dir, plot_format, dpi)
    plot_filter_weights_softmax(conv_filters, output_dir, plot_format, dpi)
    
    # Plot filter statistics
    plot_filter_statistics(all_q_filters, all_k_filters, all_v_filters, output_dir, plot_format, dpi)
    
    print(f"Saved plots to {output_dir}")


def plot_filter_weights_normalized(conv_filters, output_dir, plot_format="png", dpi=150):
    """Plot filters normalized so that max absolute value is 1 per filter."""
    
    layers_with_filters = {
        layer_idx: filters for layer_idx, filters in conv_filters.items()
        if any(v is not None for v in filters.values())
    }
    
    if not layers_with_filters:
        return
    
    num_layers = len(layers_with_filters)
    layer_indices = sorted(layers_with_filters.keys())
    
    # Split layers into two halves for two-column layout
    half = (num_layers + 1) // 2
    num_rows = half
    
    fig = plt.figure(figsize=(22, 2.5 * num_rows))
    gs = GridSpec(num_rows, 7, figure=fig, width_ratios=[1, 1, 1, 0.3, 1, 1, 1], wspace=0.3, hspace=0.3)
    
    axes = np.empty((num_rows, 6), dtype=object)
    for row in range(num_rows):
        for col in range(3):
            axes[row, col] = fig.add_subplot(gs[row, col])
        for col in range(3):
            axes[row, col + 3] = fig.add_subplot(gs[row, col + 4])
    
    for col_idx, label in enumerate(['Query', 'Key', 'Value']):
        axes[0, col_idx].set_title(label, fontsize=12, fontweight='bold')
        axes[0, col_idx + 3].set_title(label, fontsize=12, fontweight='bold')
    
    for i, layer_idx in enumerate(layer_indices):
        filters = layers_with_filters[layer_idx]
        
        if i < half:
            row_idx = i
            col_offset = 0
        else:
            row_idx = i - half
            col_offset = 3
        
        for col_idx, conv_type in enumerate(['q', 'k', 'v']):
            ax = axes[row_idx, col_offset + col_idx]
            
            shared_key = f'shared_kernel_{conv_type}'
            conv_key = f'{conv_type}_conv'
            
            filter_tensor = filters.get(shared_key)
            if filter_tensor is None:
                filter_tensor = filters.get(conv_key)
            
            if filter_tensor is not None:
                filter_np = filter_tensor.float().numpy()
                
                if filter_np.ndim == 3:
                    filter_np = filter_np.squeeze(1)
                
                # Flip so that leftmost = current token, rightmost = oldest token
                filter_np = filter_np[:, ::-1]
                
                # Normalize each filter (row) by the value at argmax position
                # So the largest absolute value becomes exactly 1.0
                normalized = np.zeros_like(filter_np)
                for i in range(filter_np.shape[0]):
                    idx = np.argmax(np.abs(filter_np[i]))
                    max_val = filter_np[i][idx]  # actual value (with sign)
                    if np.abs(max_val) > 1e-10:
                        normalized[i] = filter_np[i] / max_val
                    else:
                        normalized[i] = filter_np[i]
                
                # Fixed scale [-1, 1] since each filter is normalized to max |value| = 1
                im = ax.imshow(normalized, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
                #ax.set_xlabel('Position')
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Draw head boundaries
                n_channels, kernel_size = filter_np.shape
                num_heads = filters.get('num_heads')
                head_dim = filters.get('head_k_dim') if conv_type in ['q', 'k'] else filters.get('head_v_dim')
                draw_head_boundaries(ax, n_channels, num_heads, head_dim)
            else:
                ax.text(0.5, 0.5, 'No filter', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            
            if col_idx == 0:
                ax.set_ylabel(f'Layer {layer_idx}', fontsize=10)
            else:
                ax.set_ylabel('Channel')
    
    # Hide empty axes if odd number of layers
    if num_layers % 2 == 1:
        for col_idx in range(3):
            axes[num_rows - 1, 3 + col_idx].axis('off')
    
    plt.savefig(
        os.path.join(output_dir, f"all_conv_filters_normalized.{plot_format}"), 
        dpi=dpi, bbox_inches='tight'
    )
    plt.close()


def plot_filter_weights_normalized_abs(conv_filters, output_dir, plot_format="png", dpi=150):
    """Plot absolute value of normalized filters (max value = 1 per filter)."""
    
    layers_with_filters = {
        layer_idx: filters for layer_idx, filters in conv_filters.items()
        if any(v is not None for v in filters.values())
    }
    
    if not layers_with_filters:
        return
    
    num_layers = len(layers_with_filters)
    layer_indices = sorted(layers_with_filters.keys())
    
    # Split layers into two halves for two-column layout
    half = (num_layers + 1) // 2
    num_rows = half
    
    fig = plt.figure(figsize=(22, 2.5 * num_rows))
    gs = GridSpec(num_rows, 7, figure=fig, width_ratios=[1, 1, 1, 0.3, 1, 1, 1], wspace=0.3, hspace=0.3)
    
    axes = np.empty((num_rows, 6), dtype=object)
    for row in range(num_rows):
        for col in range(3):
            axes[row, col] = fig.add_subplot(gs[row, col])
        for col in range(3):
            axes[row, col + 3] = fig.add_subplot(gs[row, col + 4])
    
    for col_idx, label in enumerate(['Query', 'Key', 'Value']):
        axes[0, col_idx].set_title(label, fontsize=12, fontweight='bold')
        axes[0, col_idx + 3].set_title(label, fontsize=12, fontweight='bold')
    
    for i, layer_idx in enumerate(layer_indices):
        filters = layers_with_filters[layer_idx]
        
        if i < half:
            row_idx = i
            col_offset = 0
        else:
            row_idx = i - half
            col_offset = 3
        
        for col_idx, conv_type in enumerate(['q', 'k', 'v']):
            ax = axes[row_idx, col_offset + col_idx]
            
            shared_key = f'shared_kernel_{conv_type}'
            conv_key = f'{conv_type}_conv'
            
            filter_tensor = filters.get(shared_key)
            if filter_tensor is None:
                filter_tensor = filters.get(conv_key)
            
            if filter_tensor is not None:
                filter_np = filter_tensor.float().numpy()
                
                if filter_np.ndim == 3:
                    filter_np = filter_np.squeeze(1)
                
                # Flip so that leftmost = current token, rightmost = oldest token
                filter_np = filter_np[:, ::-1]
                
                # Normalize each filter (row) by the value at argmax position
                # Then take absolute value
                normalized_abs = np.zeros_like(filter_np)
                for i in range(filter_np.shape[0]):
                    idx = np.argmax(np.abs(filter_np[i]))
                    max_val = filter_np[i][idx]
                    if np.abs(max_val) > 1e-10:
                        normalized_abs[i] = np.abs(filter_np[i] / max_val)
                    else:
                        normalized_abs[i] = np.abs(filter_np[i])
                
                # Scale 0 to 1 since all values are non-negative
                im = ax.imshow(normalized_abs, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
                #ax.set_xlabel('Position')
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Draw head boundaries
                n_channels, kernel_size = filter_np.shape
                num_heads = filters.get('num_heads')
                head_dim = filters.get('head_k_dim') if conv_type in ['q', 'k'] else filters.get('head_v_dim')
                draw_head_boundaries(ax, n_channels, num_heads, head_dim)
            else:
                ax.text(0.5, 0.5, 'No filter', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            
            if col_idx == 0:
                ax.set_ylabel(f'Layer {layer_idx}', fontsize=10)
            else:
                ax.set_ylabel('Channel')
    
    # Hide empty axes if odd number of layers
    if num_layers % 2 == 1:
        for col_idx in range(3):
            axes[num_rows - 1, 3 + col_idx].axis('off')
    
    plt.savefig(
        os.path.join(output_dir, f"all_conv_filters_normalized_abs.{plot_format}"), 
        dpi=dpi, bbox_inches='tight'
    )
    plt.close()


def plot_filter_weights_softmax(conv_filters, output_dir, plot_format="png", dpi=150):
    """Plot filters with softmax applied to each filter."""
    
    layers_with_filters = {
        layer_idx: filters for layer_idx, filters in conv_filters.items()
        if any(v is not None for v in filters.values())
    }
    
    if not layers_with_filters:
        return
    
    num_layers = len(layers_with_filters)
    layer_indices = sorted(layers_with_filters.keys())
    
    # Split layers into two halves for two-column layout
    half = (num_layers + 1) // 2
    num_rows = half
    
    fig = plt.figure(figsize=(22, 2.5 * num_rows))
    gs = GridSpec(num_rows, 7, figure=fig, width_ratios=[1, 1, 1, 0.3, 1, 1, 1], wspace=0.3, hspace=0.3)
    
    axes = np.empty((num_rows, 6), dtype=object)
    for row in range(num_rows):
        for col in range(3):
            axes[row, col] = fig.add_subplot(gs[row, col])
        for col in range(3):
            axes[row, col + 3] = fig.add_subplot(gs[row, col + 4])
    
    for col_idx, label in enumerate(['Query', 'Key', 'Value']):
        axes[0, col_idx].set_title(label, fontsize=12, fontweight='bold')
        axes[0, col_idx + 3].set_title(label, fontsize=12, fontweight='bold')
    
    for i, layer_idx in enumerate(layer_indices):
        filters = layers_with_filters[layer_idx]
        
        if i < half:
            row_idx = i
            col_offset = 0
        else:
            row_idx = i - half
            col_offset = 3
        
        for col_idx, conv_type in enumerate(['q', 'k', 'v']):
            ax = axes[row_idx, col_offset + col_idx]
            
            shared_key = f'shared_kernel_{conv_type}'
            conv_key = f'{conv_type}_conv'
            
            filter_tensor = filters.get(shared_key)
            if filter_tensor is None:
                filter_tensor = filters.get(conv_key)
            
            if filter_tensor is not None:
                filter_np = filter_tensor.float().numpy()
                
                if filter_np.ndim == 3:
                    filter_np = filter_np.squeeze(1)
                
                # Flip so that leftmost = current token, rightmost = oldest token
                filter_np = filter_np[:, ::-1]
                
                # Apply softmax to each filter (row)
                # softmax(x) = exp(x) / sum(exp(x))
                exp_filters = np.exp(filter_np - np.max(filter_np, axis=1, keepdims=True))  # numerical stability
                softmax_filters = exp_filters / np.sum(exp_filters, axis=1, keepdims=True)
                
                im = ax.imshow(softmax_filters, aspect='auto', cmap='viridis', vmin=0, vmax=1, interpolation='nearest')
                #ax.set_xlabel('Position')
                plt.colorbar(im, ax=ax, shrink=0.8)
                
                # Draw head boundaries
                n_channels, kernel_size = filter_np.shape
                num_heads = filters.get('num_heads')
                head_dim = filters.get('head_k_dim') if conv_type in ['q', 'k'] else filters.get('head_v_dim')
                draw_head_boundaries(ax, n_channels, num_heads, head_dim)
            else:
                ax.text(0.5, 0.5, 'No filter', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=10, color='gray')
                ax.set_xticks([])
                ax.set_yticks([])
            
            if col_idx == 0:
                ax.set_ylabel(f'Layer {layer_idx}', fontsize=10)
            else:
                ax.set_ylabel('Channel')
    
    # Hide empty axes if odd number of layers
    if num_layers % 2 == 1:
        for col_idx in range(3):
            axes[num_rows - 1, 3 + col_idx].axis('off')
    
    plt.savefig(
        os.path.join(output_dir, f"all_conv_filters_softmax.{plot_format}"), 
        dpi=dpi, bbox_inches='tight'
    )
    plt.close()


def plot_filter_statistics(all_q_filters, all_k_filters, all_v_filters, output_dir, plot_format, dpi):
    """Plot statistics (mean, std, min, max) of filters across layers."""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    #fig.suptitle("Filter Weight Statistics Across Layers", fontsize=14)
    
    stats = {
        'Query': all_q_filters,
        'Key': all_k_filters,
        'Value': all_v_filters
    }
    
    for label, filters_list in stats.items():
        if not filters_list:
            continue
        
        layers = []
        means = []
        stds = []
        mins = []
        maxs = []
        
        for layer_idx, filter_np in filters_list:
            layers.append(layer_idx)
            means.append(np.mean(filter_np))
            stds.append(np.std(filter_np))
            mins.append(np.min(filter_np))
            maxs.append(np.max(filter_np))
        
        # Mean
        axes[0, 0].plot(layers, means, 'o-', label=label, alpha=0.8)
        axes[0, 0].set_title('Mean Weight')
        axes[0, 0].set_xlabel('Layer')
        axes[0, 0].set_ylabel('Mean')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Std
        axes[0, 1].plot(layers, stds, 'o-', label=label, alpha=0.8)
        axes[0, 1].set_title('Weight Std Dev')
        axes[0, 1].set_xlabel('Layer')
        axes[0, 1].set_ylabel('Std')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Min
        axes[1, 0].plot(layers, mins, 'o-', label=label, alpha=0.8)
        axes[1, 0].set_title('Min Weight')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Min')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Max
        axes[1, 1].plot(layers, maxs, 'o-', label=label, alpha=0.8)
        axes[1, 1].set_title('Max Weight')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Max')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"filter_statistics.{plot_format}"), 
                dpi=dpi, bbox_inches='tight')
    plt.close()


def analyze_shift_hypothesis(conv_filters, output_dir, plot_format="png", dpi=150):
    """Analyze whether filters approximate shift operations.
    
    A pure shift filter is a one-hot vector like [0, 0, 1, 0] (shift by 2).
    Note: Filters are flipped so position 0 = current token, position K-1 = oldest.
    
    Metrics computed:
    1. Peak concentration: What fraction of L1 norm is at the argmax?
       - 1.0 = perfect shift, lower = more distributed
    2. Cosine similarity to ideal shift: How close to one-hot at argmax?
    3. Dominant shift position: Where is the peak (0 = current token, K-1 = oldest)?
    """
    os.makedirs(output_dir, exist_ok=True)
    
    results = {'q': [], 'k': [], 'v': []}
    
    for layer_idx, filters in conv_filters.items():
        for conv_type in ['q', 'k', 'v']:
            shared_key = f'shared_kernel_{conv_type}'
            conv_key = f'{conv_type}_conv'
            
            filter_tensor = filters.get(shared_key)
            if filter_tensor is None:
                filter_tensor = filters.get(conv_key)
            if filter_tensor is None:
                continue
            
            filter_np = filter_tensor.float().numpy()
            if filter_np.ndim == 3:
                filter_np = filter_np.squeeze(1)  # -> [channels, kernel_size]
            
            # Flip so that leftmost = current token, rightmost = oldest token
            filter_np = filter_np[:, ::-1]
            
            # For each channel, compute shift metrics
            for ch_idx in range(filter_np.shape[0]):
                kernel = filter_np[ch_idx]
                kernel_size = len(kernel)
                
                # Metric 1: Peak concentration (fraction of L1 norm at argmax)
                abs_kernel = np.abs(kernel)
                l1_norm = np.sum(abs_kernel)
                if l1_norm > 1e-10:
                    peak_idx = np.argmax(abs_kernel)
                    peak_concentration = abs_kernel[peak_idx] / l1_norm
                else:
                    peak_idx = 0
                    peak_concentration = 0.0
                
                # Metric 2: Cosine similarity to ideal shift (one-hot at peak)
                ideal_shift = np.zeros_like(kernel)
                ideal_shift[peak_idx] = kernel[peak_idx]  # Same sign and magnitude
                
                norm_kernel = np.linalg.norm(kernel)
                norm_ideal = np.linalg.norm(ideal_shift)
                if norm_kernel > 1e-10 and norm_ideal > 1e-10:
                    cosine_sim = np.dot(kernel, ideal_shift) / (norm_kernel * norm_ideal)
                else:
                    cosine_sim = 0.0
                
                # Metric 3: Normalized shift position (0 to 1)
                normalized_shift = peak_idx / (kernel_size - 1) if kernel_size > 1 else 0
                
                results[conv_type].append({
                    'layer': layer_idx,
                    'channel': ch_idx,
                    'peak_concentration': peak_concentration,
                    'cosine_to_shift': cosine_sim,
                    'shift_position': peak_idx,
                    'normalized_shift': normalized_shift,
                    'peak_value': kernel[peak_idx],
                    'kernel': kernel,
                })
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("SHIFT HYPOTHESIS ANALYSIS")
    print("=" * 70)
    print("Peak concentration: fraction of L1 norm at argmax (1.0 = pure shift)")
    print("Cosine to shift: similarity to one-hot at peak position")
    print("-" * 70)
    
    for conv_type, label in [('q', 'Query'), ('k', 'Key'), ('v', 'Value')]:
        if not results[conv_type]:
            continue
        
        concentrations = [r['peak_concentration'] for r in results[conv_type]]
        cosines = [r['cosine_to_shift'] for r in results[conv_type]]
        shifts = [r['shift_position'] for r in results[conv_type]]
        
        print(f"\n{label} Conv:")
        print(f"  Peak concentration: mean={np.mean(concentrations):.3f}, "
              f"std={np.std(concentrations):.3f}, min={np.min(concentrations):.3f}, max={np.max(concentrations):.3f}")
        print(f"  Cosine to shift:    mean={np.mean(cosines):.3f}, "
              f"std={np.std(cosines):.3f}, min={np.min(cosines):.3f}, max={np.max(cosines):.3f}")
        print(f"  Shift positions:    {dict(zip(*np.unique(shifts, return_counts=True)))}")
    
    print("=" * 70 + "\n")
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    #fig.suptitle("Shift Hypothesis Analysis", fontsize=14)
    
    colors = {'q': 'tab:blue', 'k': 'tab:orange', 'v': 'tab:green'}
    labels = {'q': 'Query', 'k': 'Key', 'v': 'Value'}
    
    # Row 1: Peak concentration and cosine similarity distributions
    for conv_type in ['q', 'k', 'v']:
        if not results[conv_type]:
            continue
        
        concentrations = [r['peak_concentration'] for r in results[conv_type]]
        cosines = [r['cosine_to_shift'] for r in results[conv_type]]
        
        axes[0, 0].hist(concentrations, bins=20, alpha=0.6, label=labels[conv_type], color=colors[conv_type])
        axes[0, 1].hist(cosines, bins=20, alpha=0.6, label=labels[conv_type], color=colors[conv_type])
    
    axes[0, 0].set_xlabel('Peak Concentration')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('Peak Concentration Distribution\n(1.0 = pure shift)')
    axes[0, 0].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect shift')
    axes[0, 0].legend()
    
    axes[0, 1].set_xlabel('Cosine Similarity to Ideal Shift')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Cosine Similarity to One-Hot\n(1.0 = perfect match)')
    axes[0, 1].axvline(x=1.0, color='red', linestyle='--', alpha=0.7, label='Perfect shift')
    axes[0, 1].legend()
    
    # Row 1, Col 3: Shift position histogram
    for conv_type in ['q', 'k', 'v']:
        if not results[conv_type]:
            continue
        shifts = [r['shift_position'] for r in results[conv_type]]
        unique_shifts = sorted(set(shifts))
        counts = [shifts.count(s) for s in unique_shifts]
        x_offset = {'q': -0.25, 'k': 0, 'v': 0.25}[conv_type]
        axes[0, 2].bar([s + x_offset for s in unique_shifts], counts, width=0.25, 
                       alpha=0.7, label=labels[conv_type], color=colors[conv_type])
    
    axes[0, 2].set_xlabel('Shift Position (kernel index)')
    axes[0, 2].set_ylabel('Count')
    axes[0, 2].set_title('Distribution of Shift Positions\n(0 = no shift)')
    axes[0, 2].legend()
    
    # Row 2: Per-layer analysis
    for col_idx, conv_type in enumerate(['q', 'k', 'v']):
        ax = axes[1, col_idx]
        if not results[conv_type]:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{labels[conv_type]} Conv')
            continue
        
        # Group by layer
        layer_data = {}
        for r in results[conv_type]:
            layer = r['layer']
            if layer not in layer_data:
                layer_data[layer] = []
            layer_data[layer].append(r['peak_concentration'])
        
        layers = sorted(layer_data.keys())
        means = [np.mean(layer_data[l]) for l in layers]
        stds = [np.std(layer_data[l]) for l in layers]
        
        ax.errorbar(layers, means, yerr=stds, fmt='o-', capsize=3, color=colors[conv_type])
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Perfect shift')
        ax.set_xlabel('Layer')
        ax.set_ylabel('Peak Concentration')
        ax.set_title(f'{labels[conv_type]} Conv\n(mean ± std per layer)')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shift_hypothesis.{plot_format}"), 
                dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Additional plot: Show example filters vs ideal shifts
    plot_example_filters_vs_shifts(results, output_dir, plot_format, dpi)
    
    return results


def plot_example_filters_vs_shifts(results, output_dir, plot_format, dpi):
    """Plot example filters compared to their ideal shift versions."""
    
    fig, axes = plt.subplots(3, 4, figsize=(16, 10))
    #fig.suptitle("Example Filters vs Ideal Shifts (sorted by peak concentration)", fontsize=14)
    
    for row_idx, (conv_type, label) in enumerate([('q', 'Query'), ('k', 'Key'), ('v', 'Value')]):
        if not results[conv_type]:
            for col in range(4):
                axes[row_idx, col].text(0.5, 0.5, 'No data', ha='center', va='center', 
                                        transform=axes[row_idx, col].transAxes)
            continue
        
        # Sort by peak concentration
        sorted_results = sorted(results[conv_type], key=lambda x: x['peak_concentration'])
        
        # Pick examples: worst, 25th percentile, 75th percentile, best
        n = len(sorted_results)
        indices = [0, n // 4, 3 * n // 4, n - 1]
        titles = ['Lowest', '25th %ile', '75th %ile', 'Highest']
        
        for col_idx, (idx, title) in enumerate(zip(indices, titles)):
            ax = axes[row_idx, col_idx]
            r = sorted_results[idx]
            kernel = r['kernel']
            kernel_size = len(kernel)
            
            # Create ideal shift
            peak_idx = r['shift_position']
            ideal = np.zeros_like(kernel)
            ideal[peak_idx] = kernel[peak_idx]
            
            x = np.arange(kernel_size)
            width = 0.35
            
            ax.bar(x - width/2, kernel, width, label='Actual', alpha=0.8)
            ax.bar(x + width/2, ideal, width, label='Ideal shift', alpha=0.8)
            ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
            
            # ax.set_xlabel('Position')
            ax.set_xticks(x)
            if col_idx == 0:
                ax.set_ylabel(f'{label}\nWeight')
            
            conc = r['peak_concentration']
            ax.set_title(f'{title}\nL{r["layer"]} Ch{r["channel"]}\nConc={conc:.3f}')
            
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"shift_examples.{plot_format}"), 
                dpi=dpi, bbox_inches='tight')
    plt.close()


def plot_filter_frequency_response(conv_filters, output_dir, plot_format="png", dpi=150):
    """Plot the frequency response (FFT) of the convolutional filters."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all shared kernels or a sample of regular convs
    q_kernels = []
    k_kernels = []
    v_kernels = []
    
    for layer_idx, filters in conv_filters.items():
        for conv_type, kernel_list in [('q', q_kernels), ('k', k_kernels), ('v', v_kernels)]:
            shared_key = f'shared_kernel_{conv_type}'
            conv_key = f'{conv_type}_conv'
            
            filter_tensor = filters.get(shared_key)
            if filter_tensor is None:
                filter_tensor = filters.get(conv_key)
            if filter_tensor is not None:
                filter_np = filter_tensor.float().numpy().squeeze()
                if filter_np.ndim == 1:
                    kernel_list.append((layer_idx, filter_np))
                else:
                    # Take mean across channels
                    kernel_list.append((layer_idx, np.mean(filter_np, axis=0)))
    
    if not any([q_kernels, k_kernels, v_kernels]):
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    #fig.suptitle("Filter Frequency Response (FFT Magnitude)", fontsize=14)
    
    for ax, (kernels, label) in zip(axes, [
        (q_kernels, 'Query'),
        (k_kernels, 'Key'),
        (v_kernels, 'Value')
    ]):
        if not kernels:
            ax.text(0.5, 0.5, 'No filters', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(label)
            continue
        
        # Pad kernels for better frequency resolution
        pad_length = 64
        for layer_idx, kernel in kernels:
            padded = np.zeros(pad_length)
            padded[:len(kernel)] = kernel
            fft_mag = np.abs(np.fft.rfft(padded))
            freqs = np.fft.rfftfreq(pad_length)
            ax.plot(freqs, fft_mag, alpha=0.6, label=f'L{layer_idx}')
        
        #ax.set_xlabel('Normalized Frequency')
        ax.set_ylabel('Magnitude')
        ax.set_title(f'{label} Conv')
        ax.grid(True, alpha=0.3)
        if len(kernels) <= 12:
            ax.legend(fontsize=7, ncol=2)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"frequency_response.{plot_format}"), 
                dpi=dpi, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()
    
    # Set device (CPU is fine for just extracting weights)
    device = "cpu"
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {args.model_id}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.float32,  # Use float32 for better precision in plotting
        trust_remote_code=True,
    ).to(device)
    
    # Create output directory with model name as subfolder
    # Handle both HuggingFace IDs (e.g., "fla-hub/delta_net_1.3B") and local paths
    model_path = args.model_id.rstrip("/")
    # If path ends with "checkpoints", use three parent directories for naming
    if os.path.basename(model_path) == "checkpoints":
        # e.g., /fast/.../delta_net_2_head/340m/10BT/checkpoints -> delta_net_2_head/340m/10BT
        parent = os.path.dirname(model_path)
        model_name = os.path.join(
            os.path.basename(os.path.dirname(os.path.dirname(parent))),  # delta_net_2_head
            os.path.basename(os.path.dirname(parent)),                   # 340m
            os.path.basename(parent)                                     # 10BT
        )
    else:
        model_name = os.path.basename(model_path)
    output_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Extract convolutional filters
    print("Extracting convolutional filters...")
    conv_filters = extract_conv_filters(model)
    
    # Count layers with filters
    layers_with_filters = sum(1 for filters in conv_filters.values() 
                              if any(v is not None for v in filters.values()))
    print(f"Found convolutional filters in {layers_with_filters}/{len(conv_filters)} layers")
    
    # Print filter shapes
    print("\nFilter shapes per layer:")
    for layer_idx, filters in conv_filters.items():
        shapes = []
        for key, value in filters.items():
            if value is not None and hasattr(value, 'shape'):
                shapes.append(f"{key}: {list(value.shape)}")
        if shapes:
            print(f"  Layer {layer_idx}: {', '.join(shapes)}")
    
    # Plot filters
    print(f"\nGenerating plots...")
    plot_filter_weights(conv_filters, output_dir, args.plot_format, args.dpi)
    plot_filter_frequency_response(conv_filters, output_dir, args.plot_format, args.dpi)
    
    # Analyze shift hypothesis
    analyze_shift_hypothesis(conv_filters, output_dir, args.plot_format, args.dpi)
    
    if args.show_plots:
        plt.show()
    
    print("Done!")


if __name__ == "__main__":
    main()

