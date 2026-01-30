#!/usr/bin/env python3
"""
Create a LaTeX table from DeltaNet evaluation results.

This script reads all *_results.json files from an evaluation directory
and generates a formatted LaTeX table ready for copy-pasting into papers.

Usage:
    # Basic usage - generates full table
    python create_eval_table.py /path/to/flame/dump/eval
    
    # Generate only tabular (without \\begin{table}...\\end{table})
    python create_eval_table.py /path/to/flame/dump/eval --simple
    
    # Sort by a specific task
    python create_eval_table.py /path/to/flame/dump/eval --sort-by hellaswag
    
    # Use standard accuracy instead of normalized accuracy
    python create_eval_table.py /path/to/flame/dump/eval --metric "acc,none"
    
    # Custom caption and label
    python create_eval_table.py /path/to/flame/dump/eval \
        --caption "My custom caption" \
        --label "tab:my_results"

Output:
    The script prints LaTeX code to stdout and info messages to stderr.
    You can redirect to a file: python create_eval_table.py /path > table.tex
    
    The LaTeX table uses the booktabs package. Make sure to add to your preamble:
        \\usepackage{booktabs}
        \\usepackage{pifont} % For checkmarks/xmarks if needed, or amssymb
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import sys
import re


def extract_head_dim(model_name: str) -> int:
    """Extract head dimension from model name. Defaults to 128 if not found."""
    # Try to extract the number after _head_ (head dim)
    m = re.search(r'_head_(\d+)', model_name)
    if m:
        return int(m.group(1))
    # For delta_net_2_head, delta_net_2, delta_net, treat as 128 if no number
    if re.match(r"delta_net_2_head$", model_name) or re.match(r"deltanet_2_head$", model_name):
        return 128
    if re.match(r"delta_net$", model_name) or re.match(r"deltanet$", model_name):
        return 128
    if re.match(r"delta_net_2$", model_name) or re.match(r"deltanet_2$", model_name):
        return 128
    # For delta_net_<dim> (not head), extract <dim>
    m2 = re.match(r"delta_net_(\d+)$", model_name)
    if m2:
        return int(m2.group(1))
    # Fallback: largest number in the name (handles ...compressed_102 etc)
    numbers = [int(x) for x in re.findall(r'(\d+)', model_name)]
    if numbers:
        return max(numbers)
    return 128


def get_compression_method(model_name: str) -> str:
    """
    Determine compression method:
    - 'Struct' if 'structured' is in the name.
    - 'dl1' if 'l1_data' is in the name.
    - 'L1', 'L2', 'Random' if explicit keywords present.
    - 'PCA' if 'compressed' or 'finetuned' is in the name but not specific above.
    - '--' for baselines.
    """
    name = model_name.lower()
    
    # Check if it's a baseline (no compressed/finetuned/distill keywords)
    is_variant = any(x in name for x in ["compressed", "finetuned", "distill"])
    
    if not is_variant:
        return "--"
        
    if "structured" in name:
        return "Struct"
    
    if "rrqr_data" in name:
        return "drrqr"
    if "rrqr" in name:
        return "rrqr"
    if "wanda" in name:
        return "wanda"
    
    # --- NEW MODELS ADDED HERE ---
    # Check l1_data before l1 to identify dl1 correctly
    if "l1_data" in name:
        return "DL1"
    if "l1" in name:
        return "L1"
    if "l2" in name:
        return "L2"
    if "random" in name:
        return "Random"
    # -----------------------------
    
    # Default to PCA for other compressed/finetuned variants
    return "PCA"


def load_results(results_dir: Path) -> Dict[str, Dict]:
    """Load all result JSON files from the directory."""
    results = {}
    
    for json_file in sorted(results_dir.glob("*_results.json")):
        model_name = json_file.stem.replace("_results", "")
        
        try:
            with open(json_file) as f:
                data = json.load(f)
                results[model_name] = data.get("results", {})
        except Exception as e:
            print(f"Warning: Failed to load {json_file}: {e}", file=sys.stderr)
    
    return results


def format_model_name(model_name: str) -> str:
    """Format model name for display in table."""
    
    # Start with the original name
    name = model_name
    
    # Clean up "distill" from the name since it has its own column now
    name = name.replace("_distill", "").replace("distill_", "")
    
    # Clean up "structured" since it has its own column now
    name = name.replace("_structured", "").replace("structured_", "")

    # Clean up "rrqr" and "wanda" related names
    name = name.replace("_rrqr_data", "").replace("rrqr_data_", "")
    name = name.replace("_rrqr", "").replace("rrqr_", "")
    name = name.replace("_wanda", "").replace("wanda_", "")
    
    # Clean up L1/L2/Random from the name string if they are now the "Method"
    # Note: We must clean l1_data before l1
    name = name.replace("_l1_data", "").replace("l1_data_", "")
    name = name.replace("_l1", "").replace("l1_", "")
    name = name.replace("_l2", "").replace("l2_", "")
    # Note: We keep 'random' cleaning specific to variant logic below to avoid over-cleaning
    
    # Determine the base model first
    base_model = None
    # Special case: treat delta_net_2_head_<N> as baseline if no variant keywords
    if (re.match(r"delta_net_2_head_\d+$", model_name) or re.match(r"deltanet_2_head_\d+$", model_name)) and not any(x in model_name for x in ["compressed", "finetuned", "random", "l1", "l2"]):
        base_model = "DeltaNet-2-Head"
        name = ""
    # Also treat delta_net_<N> as DeltaNet if no variant keywords
    elif (re.match(r"delta_net_\d+$", model_name) or re.match(r"deltanet_\d+$", model_name)) and not any(x in model_name for x in ["compressed", "finetuned", "random", "head", "l1", "l2"]):
        base_model = "DeltaNet"
        name = ""
    elif "delta_net_no_conv" in name or "deltanet_no_conv" in name:
        base_model = "DeltaNet-NoConv"
        name = name.replace("delta_net_no_conv", "").replace("deltanet_no_conv", "")
    elif "delta_net_2_head" in name or "deltanet_2_head" in name:
        base_model = "DeltaNet-2-Head"
        name = name.replace("delta_net_2_head", "").replace("deltanet_2_head", "")
    elif "delta_net_2" in name or "deltanet_2" in name:
        base_model = "DeltaNet-2"
        name = name.replace("delta_net_2", "").replace("deltanet_2", "")
    elif "delta_net" in name or "deltanet" in name:
        base_model = "DeltaNet"
        name = name.replace("delta_net", "").replace("deltanet", "")
    
    # If this is just a base model with no suffix, return it
    name = name.strip("_")
    if not name and base_model:
        return base_model
    
    # Now check for variants/modifications
    variant_parts = []
    
    # Check for finetuned variants
    if "finetuned_selective_random" in name:
        match = re.search(r'finetuned_selective_random_(\d+)', name)
        if match:
            variant_parts.append(f"Finetuned-Selective-Random-{match.group(1)}")
        else:
            variant_parts.append("Finetuned-Selective-Random")
        name = re.sub(r'_?finetuned_selective_random_?\d*', '', name)
    elif "finetuned_selective" in name:
        match = re.search(r'finetuned_selective_(\d+)', name)
        if match:
            variant_parts.append(f"Finetuned-Selective-{match.group(1)}")
        else:
            variant_parts.append("Finetuned-Selective")
        name = re.sub(r'_?finetuned_selective_?\d*', '', name)
    elif "finetuned" in name:
        match = re.search(r'finetuned_(\w+)', name)
        if match:
            # If the suffix was l1/l2/random, it's now covered by Method column
            # so we just call it "Finetuned"
            suffix = match.group(1)
            if suffix in ["l1", "l2", "random", "l1_data"]:
                variant_parts.append("Finetuned")
            else:
                variant_parts.append(f"Finetuned-{suffix.capitalize()}")
        else:
            variant_parts.append("Finetuned")
        name = re.sub(r'_?finetuned_?\w*', '', name)
    
    # Check for compressed variants
    if "compressed_random" in name:
        match = re.search(r'compressed_random_(\d+)', name)
        if match:
            # If 'Random' is the method, we might just want "Compressed-<Dim>"
            # But let's keep it descriptive
            variant_parts.append(f"Compressed-Random-{match.group(1)}")
        else:
            variant_parts.append("Compressed-Random")
        name = re.sub(r'_?compressed_random_?\d*', '', name)
    elif "compressed" in name:
        match = re.search(r'compressed_(\d+)', name)
        if match:
            variant_parts.append(f"Compressed-{match.group(1)}")
        else:
            variant_parts.append("Compressed")
        name = re.sub(r'_?compressed_?\d*', '', name)
    
    # Build the final name
    if base_model and variant_parts:
        return f"{base_model}-{'-'.join(variant_parts)}"
    elif base_model:
        return base_model
    elif variant_parts:
        return "-".join(variant_parts)
    
    # Fallback: generic cleanup
    name = name.strip("_").replace("_", "-")
    if name:
        return "-".join(word.capitalize() for word in name.split("-"))
    
    return "Unknown"


def get_model_category(display_name: str) -> str:
    """
    Classify model into a readable category for the table.
    
    Categories:
        - \\textbf{Baseline}
        - Compressed
        - Finetuned
        - Other
    """
    # Anything without finetuning/compression markers is treated as a baseline
    has_finetuned = "Finetuned" in display_name
    has_compressed = "Compressed" in display_name
    
    if not (has_finetuned or has_compressed):
        return r"\textbf{Baseline}"
    
    if has_finetuned:
        return "Finetuned"
    
    if has_compressed:
        return "Compressed"
    
    return "Other"

def get_task_display_name(task: str) -> str:
    """Get display name for task."""
    task_names = {
        "arc_challenge": "ARC-C",
        "arc_easy": "ARC-E",
        "hellaswag": "HellaSwag",
        "winogrande": "WinoGrande",
        "mmlu": "MMLU",
        "piqa": "PIQA",
        "sciq": "SciQ",
        "triviaqa": "TriviaQA",
    }
    return task_names.get(task, task.replace("_", " ").title())


def extract_metrics(results: Dict[str, Dict], metric: str = "acc_norm,none") -> Tuple[List[str], List[str], List[List[float]], List[str]]:
    """
    Extract metrics from results.
    """
    # Get all unique tasks across all models
    all_tasks = set()
    for model_results in results.values():
        all_tasks.update(model_results.keys())

    # Sort tasks for consistent ordering - wikitext goes first, lombada_openai second
    task_order = ["arc_easy", "arc_challenge", "hellaswag", "winogrande", "mmlu", "piqa", "sciq"]
    tasks = sorted(all_tasks, key=lambda x: (x != "wikitext", x != "lambada_openai", task_order.index(x) if x in task_order else 999))

    model_names, tasks, data, actual_metrics = _extract_metrics_with_task_filter(results, tasks, metric)
    return model_names, tasks, data, actual_metrics

# New helper to allow task filtering
def _extract_metrics_with_task_filter(results: Dict[str, Dict], tasks: List[str], metric: str = "acc_norm,none") -> Tuple[List[str], List[str], List[List[float]], List[str]]:
    model_names = sorted(results.keys())
    data = []
    # Track which metric was actually used for each task
    actual_metrics_used = []
    
    # Expand tasks to include both perplexity and accuracy for lambada_openai
    expanded_tasks = []
    for task in tasks:
        if task == "lambada_openai":
            expanded_tasks.append("lambada_openai_ppl")
            expanded_tasks.append("lambada_openai_acc")
        else:
            expanded_tasks.append(task)
    
    for task_idx, task in enumerate(expanded_tasks):
        # Determine which metric to use for this task by checking the first model
        first_model = model_names[0]
        base_task = task.replace("_ppl", "").replace("_acc", "")
        task_results = results[first_model].get(base_task, {})
        actual_metric = None
        
        if task == "wikitext":
            for metric_name in ["word_perplexity,none", "word_perplexity"]:
                if metric_name in task_results:
                    actual_metric = metric_name
                    break
        elif task == "lambada_openai_ppl":
            # For lambada_openai perplexity
            for metric_name in ["perplexity,none", "perplexity"]:
                if metric_name in task_results:
                    actual_metric = metric_name
                    break
        elif task == "lambada_openai_acc":
            # For lambada_openai accuracy
            for metric_name in [metric, "acc,none", "acc_norm", "acc"]:
                if metric_name in task_results:
                    actual_metric = metric_name
                    break
        else:
            for metric_name in [metric, "acc,none", "acc_norm", "acc"]:
                if metric_name in task_results:
                    actual_metric = metric_name
                    break
        
        actual_metrics_used.append(actual_metric if actual_metric else metric)
    
    for model_name in model_names:
        model_data = []
        for task_idx, task in enumerate(expanded_tasks):
            base_task = task.replace("_ppl", "").replace("_acc", "")
            task_results = results[model_name].get(base_task, {})
            value = None
            if task == "wikitext":
                # Use perplexity for wikitext
                for metric_name in ["word_perplexity,none", "word_perplexity"]:
                    if metric_name in task_results:
                        value = task_results[metric_name]
                        break
            elif task == "lambada_openai_ppl":
                # For lambada_openai, use perplexity
                for metric_name in ["perplexity,none", "perplexity"]:
                    if metric_name in task_results:
                        value = task_results[metric_name]
                        break
            elif task == "lambada_openai_acc":
                # For lambada_openai, use accuracy
                for metric_name in [metric, "acc,none", "acc_norm", "acc"]:
                    if metric_name in task_results:
                        value = task_results[metric_name]
                        break
            else:
                for metric_name in [metric, "acc,none", "acc_norm", "acc"]:
                    if metric_name in task_results:
                        value = task_results[metric_name]
                        break
            model_data.append(value)
        data.append(model_data)
    return model_names, expanded_tasks, data, actual_metrics_used


def get_channels_shared(model_name: str) -> str:
    """Return checkmark/times for channel sharing: 'xx'→$\times$$\times$, 'xo'→$\times$\checkmark, 'oo'→\checkmark\checkmark, 'no conv' for DeltaNet-NoConv and variants."""
    name = model_name.lower()
    times = r'$\times$'
    check = r'\checkmark'
    if 'shared' in name:
        return 'kv shared'
    if 'deltanet_no_conv' in name or 'delta_net_no_conv' in name:
        return 'no conv'
    # delta_net_2_head: cross + check
    if 'deltanet-2-head' in name or 'delta_net_2_head' in name:
        return f'{times}{check}'
    # pure delta_net_2 (not head): two checks
    if (('deltanet-2' in name or 'delta_net_2' in name) and not ('head' in name)):
        return f'{check}{check}'
    if 'deltanet' in name or 'delta_net' in name:
        return f'{times}{times}'
    return '--'  # fallback

def create_latex_table(model_names: List[str], tasks: List[str], data: List[List[float]], 
                       caption: str = "Evaluation results on downstream tasks",
                       label: str = "tab:eval_results",
                       actual_metrics: List[str] = None) -> str:
    """Create a LaTeX table from the data."""
    # Use get_channels_shared for the new column
    channels_shared = [get_channels_shared(name) for name in model_names]
    
    # Calculate methods (PCA vs Struct)
    methods = [get_compression_method(name) for name in model_names]
    
    # Categories and formatting (format_model_name cleans "structured" out of display name)
    categories = [get_model_category(format_model_name(name)) for name in model_names]
    head_dims = [extract_head_dim(name) for name in model_names]
    
    # Add line break and arrow (up for accuracy, down for perplexity)
    task_display_names = []
    # Abbreviation mapping for task names
    task_abbrev = {
        "wikitext": "Wiki.",
        "lambada_openai_ppl": "LMB.",
        "lambada_openai_acc": "LMB.",
        "hellaswag": "Hella.",
        "winogrande": "Wino.",
        "arc_easy": "ARC-e",
        "arc_challenge": "ARC-c",
        "piqa": "PIQA",
        "sciq": "SciQ",
        "boolq": "BoolQ",
        "mmlu": "MMLU",
        "triviaqa": "TriviaQA",
    }
    for task_idx, task in enumerate(tasks):
        if task == "wikitext":
            task_display_names.append(f"\\thead[t]{{Wiki.\\\\ ppl $\\downarrow$}}")
        elif task == "lambada_openai_ppl":
            task_display_names.append(f"\\thead[t]{{LMB.\\\\ ppl $\\downarrow$}}")
        elif task == "lambada_openai_acc":
            # Use actual metric for this task
            metric_name = actual_metrics[task_idx] if actual_metrics else "acc"
            metric_display = metric_name.replace(',none', '').replace('acc_norm', 'acc_n').replace('_', '\\_')
            task_display_names.append(f"\\thead[t]{{LMB.\\\\ {metric_display} $\\uparrow$}}")
        else:
            # Use actual metric for this task
            metric_name = actual_metrics[task_idx] if actual_metrics else "acc_norm"
            metric_display = metric_name.replace(',none', '').replace('acc_norm', 'acc_n').replace('_', '\\_')
            task_abbr = task_abbrev.get(task, get_task_display_name(task))
            task_display_names.append(f"\\thead[t]{{{task_abbr}\\\\ {metric_display} $\\uparrow$}}")
    
    # Identify non-perplexity tasks for averaging (exclude wikitext and lambada_openai perplexity)
    accuracy_task_indices = [i for i, task in enumerate(tasks) if task not in ["wikitext", "lambada_openai_ppl"]]
    
    # Add DISTILLED indicator
    def is_distilled(name):
        n = name.lower()
        return 'distill' in n
    
    def is_benchmark(cat):
        # Treat any row with 'Baseline' as a benchmark row
        return 'Baseline' in cat
    
    distilled_marks = []
    for i, n in enumerate(model_names):
        if is_benchmark(categories[i]):
            distilled_marks.append('--')
        else:
            distilled_marks.append(r'\checkmark' if is_distilled(n) else r'$\times$')
            
    # Separate out 'no conv' and 'kv shared' entries
    main_rows = []
    no_conv_rows = []
    kv_shared_rows = []
    
    for i in range(len(model_names)):
        ch = channels_shared[i]
        
        row_data = []
        accuracy_values = []  # For computing average
        for task_idx, task in enumerate(tasks):
            value = data[i][task_idx]
            if value is not None:
                if task in ["wikitext", "lambada_openai_ppl"]:
                    row_data.append(f"{value:.1f}")
                else:
                    row_data.append(f"{value * 100:.1f}")
                    # Collect accuracy values for averaging
                    if task_idx in accuracy_task_indices:
                        accuracy_values.append(value * 100)
            else:
                row_data.append("--")
        
        # Calculate average accuracy (excluding perplexity metrics)
        avg_accuracy = sum(accuracy_values) / len(accuracy_values) if accuracy_values else None
        avg_str = f"{avg_accuracy:.1f}" if avg_accuracy is not None else "--"
        
        # Row format: (Category, Method, Shared/Conv, HeadDim, DistilledMark, Data, Avg)
        row = (categories[i], methods[i], ch, head_dims[i], distilled_marks[i], row_data, avg_str)
        
        if ch == 'no conv':
            no_conv_rows.append(row)
        elif ch == 'kv shared':
            kv_shared_rows.append(row)
        else:
            main_rows.append(row)
    
    # Sort by head_dim (as int), but for each dim, put all Baseline rows first, then others
    def row_sort_key(row):
        # row: (cat, method, ch, head_dim, dist, row_data, avg)
        # Baseline rows get 0, others get 1, then sort by dim
        is_baseline = 0 if 'Baseline' in row[0] else 1
        return (row[3], is_baseline, row[0], row[2])
    
    main_rows = sorted(main_rows, key=row_sort_key)
    no_conv_rows = sorted(no_conv_rows, key=lambda x: x[3])
    
    # Start building the table
    lines = []
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\small")
    
    # Column specification: 
    # Type (l), Method (c), Shared/Conv (l), Head Dim (l), Dist (l) | Perplexity (c...) | Accuracy (c...) | Avg (c)
    num_perplexity_tasks = len([t for t in tasks if t in ["wikitext", "lambada_openai_ppl"]])
    num_accuracy_tasks = len([t for t in tasks if t not in ["wikitext", "lambada_openai_ppl"]])
    col_spec = "l" + "c" + "l" + "l" + "l|" + "c" * num_perplexity_tasks + "|" + "c" * num_accuracy_tasks + "|c"
    
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    # Header row
    perplexity_names = [task_display_names[i] for i, t in enumerate(tasks) if t in ["wikitext", "lambada_openai_ppl"]]
    accuracy_task_names = [task_display_names[i] for i, t in enumerate(tasks) if t not in ["wikitext", "lambada_openai_ppl"]]
    
    header = "\\thead[t]{Type} & \\thead[t]{Method} & \\thead[t]{Shared\\\\Conv} & \\thead[t]{$d_k$} & \\thead[t]{dist.} & " + \
             (" & ".join(perplexity_names) if perplexity_names else "") + (" & " if perplexity_names else "") + \
             " & ".join(accuracy_task_names) + " & \\thead[t]{Avg\\\\ $\\uparrow$} \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Helper to print row
    def print_rows(rows):
        for cat, method, ch, head_dim, dist, row_data, avg in rows:
            # Split row_data into perplexity and accuracy task parts
            perplexity_data = [row_data[i] for i in range(len(row_data)) if tasks[i] in ["wikitext", "lambada_openai_ppl"]]
            accuracy_task_data = [row_data[i] for i in range(len(row_data)) if tasks[i] not in ["wikitext", "lambada_openai_ppl"]]
            
            row_str = f"{cat} & {method} & {ch} & {head_dim} & {dist} & "
            if perplexity_data:
                row_str += " & ".join(perplexity_data) + " & "
            
            row_str += " & ".join(accuracy_task_data) + f" & {avg} \\\\"
            lines.append(row_str)

    # Main rows
    print_rows(main_rows)
    
    # Add separator and no conv rows at the end
    if no_conv_rows:
        lines.append("\\hline")
        print_rows(no_conv_rows)
            
    # Always put kv shared rows at the very end
    if kv_shared_rows:
        lines.append("\\hline")
        print_rows(kv_shared_rows)
            
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\label{{{label}}}")
    lines.append("\\end{table}")

    return "\n".join(lines)


def create_simple_latex_table(model_names: List[str], tasks: List[str], data: List[List[float]], actual_metrics: List[str] = None) -> str:
    """Create a simpler LaTeX table without table environment (for easy embedding)."""
    # Format model names
    display_names = [format_model_name(name) for name in model_names]
    methods = [get_compression_method(name) for name in model_names]
    categories = [get_model_category(name) for name in display_names]
    head_dims = [extract_head_dim(name) for name in model_names]
    
    task_display_names = []
    # Abbreviation mapping for task names
    task_abbrev = {
        "wikitext": "Wiki.",
        "lambada_openai_ppl": "LMB.",
        "lambada_openai_acc": "LMB.",
        "hellaswag": "Hella.",
        "winogrande": "Wino.",
        "arc_easy": "ARC-e",
        "arc_challenge": "ARC-c",
        "piqa": "PIQA",
        "sciq": "SciQ",
        "boolq": "BoolQ",
        "mmlu": "MMLU",
        "triviaqa": "TriviaQA",
    }
    for task_idx, task in enumerate(tasks):
        if task == "wikitext":
            task_display_names.append(f"\\thead[t]{{Wiki.\\\\ ppl $\\downarrow$}}")
        elif task == "lambada_openai_ppl":
            task_display_names.append(f"\\thead[t]{{LMB.\\\\ ppl $\\downarrow$}}")
        elif task == "lambada_openai_acc":
            metric_name = actual_metrics[task_idx] if actual_metrics else "acc"
            metric_display = metric_name.replace(',none', '').replace('acc_norm', 'acc_n').replace('_', '\\_')
            task_display_names.append(f"\\thead[t]{{LMB.\\\\ {metric_display} $\\uparrow$}}")
        else:
            metric_name = actual_metrics[task_idx] if actual_metrics else "acc_norm"
            metric_display = metric_name.replace(',none', '').replace('acc_norm', 'acc_n').replace('_', '\\_')
            task_abbr = task_abbrev.get(task, get_task_display_name(task))
            task_display_names.append(f"\\thead[t]{{{task_abbr}\\\\ {metric_display} $\\uparrow$}}")
    
    lines = []
    
    # Column specification: l for type, c for method, l for model name, l for head dim, c for each task
    col_spec = "l" + "c" + "l" + "l" + "c" * len(tasks)
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    
    # Header row
    header = "\\thead[t]{Type} & \\thead[t]{Method} & \\thead[t]{Model} & \\thead[t]{Head Dim} & " + " & ".join(task_display_names) + " \\\\"
    lines.append(header)
    lines.append("\\midrule")
    
    # Data rows
    for model_idx, display_name in enumerate(display_names):
        category = categories[model_idx]
        method = methods[model_idx]
        head_dim = head_dims[model_idx]
        row_data = []
        for task_idx, task in enumerate(tasks):
            value = data[model_idx][task_idx]
            if value is not None:
                if task in ["wikitext", "lambada_openai_ppl"]:
                    row_data.append(f"{value:.1f}")
                else:
                    row_data.append(f"{value * 100:.1f}")
            else:
                row_data.append("--")
        row = f"{category} & {method} & {display_name} & {head_dim} & " + " & ".join(row_data) + " \\\\"
        lines.append(row)
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Create LaTeX table from evaluation results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=""
    )
    
    parser.add_argument(
        "--tasks",
        type=str,
        default="arc_easy,arc_challenge,hellaswag,winogrande,piqa,wikitext,lambada_openai",
        help="Comma-separated list of tasks/metrics to include in the table (e.g., arc_easy,arc_challenge,hellaswag,winogrande,mmlu,piqa,wikitext,openbookqa). If not set, include all."
    )

    parser.add_argument(
        "results_dir",
        type=str,
        help="Directory containing *_results.json files"
    )
    
    parser.add_argument(
        "--metric",
        type=str,
        default="acc_norm,none",
        help="Metric to extract (default: acc_norm,none)"
    )
    
    parser.add_argument(
        "--simple",
        action="store_true",
        help="Generate only tabular without table environment"
    )
    
    parser.add_argument(
        "--caption",
        type=str,
        default="Evaluation results on downstream tasks",
        help="Table caption"
    )
    
    parser.add_argument(
        "--label",
        type=str,
        default="tab:eval_results",
        help="Table label"
    )
    
    parser.add_argument(
        "--sort-by",
        type=str,
        default=None,
        help="Task to sort by (e.g., 'hellaswag'). Sorts in descending order."
    )
    
    args = parser.parse_args()
    
    # Load results
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Loading results from: {results_dir}", file=sys.stderr)
    results = load_results(results_dir)
    
    if not results:
        print(f"Error: No result files found in {results_dir}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(results)} models", file=sys.stderr)
    

    # Determine tasks to include
    if args.tasks:
        selected_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        # Validate tasks
        all_results = set()
        for model_results in results.values():
            all_results.update(model_results.keys())
        missing = [t for t in selected_tasks if t not in all_results]
        if missing:
            print(f"Warning: Some specified tasks not found in results: {', '.join(missing)}", file=sys.stderr)
        tasks = [t for t in selected_tasks if t in all_results]
        if not tasks:
            print(f"Error: No valid tasks specified. Available: {', '.join(sorted(all_results))}", file=sys.stderr)
            sys.exit(1)
        model_names, tasks, data, actual_metrics = _extract_metrics_with_task_filter(results, tasks, args.metric)
    else:
        model_names, tasks, data, actual_metrics = extract_metrics(results, args.metric)

    # Sort by specified task if requested
    if args.sort_by:
        if args.sort_by in tasks:
            task_idx = tasks.index(args.sort_by)
            # Sort model_names and data together by the task values
            sorted_pairs = sorted(
                zip(model_names, data),
                key=lambda x: x[1][task_idx] if x[1][task_idx] is not None else -1,
                reverse=True
            )
            model_names, data = zip(*sorted_pairs)
            model_names = list(model_names)
            data = list(data)
            print(f"Sorted by {args.sort_by} (descending)", file=sys.stderr)
        else:
            print(f"Warning: Task '{args.sort_by}' not found in results. Available tasks: {', '.join(tasks)}", 
                  file=sys.stderr)

    # Generate table
    print(f"Generating LaTeX table...", file=sys.stderr)
    print(f"Tasks: {', '.join(tasks)}", file=sys.stderr)
    print("", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print("Copy the LaTeX code below:", file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    print()

    if args.simple:
        latex_table = create_simple_latex_table(model_names, tasks, data, actual_metrics)
    else:
        latex_table = create_latex_table(model_names, tasks, data, args.caption, args.label, actual_metrics)

    print(latex_table)

    print()
    print("=" * 80, file=sys.stderr)
    print("Note: Make sure to include \\usepackage{booktabs} and \\usepackage{amssymb} (for \\times) in your LaTeX preamble", file=sys.stderr)
    print("=" * 80, file=sys.stderr)


if __name__ == "__main__":
    main()