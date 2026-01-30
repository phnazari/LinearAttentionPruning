#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluate FLAME models from training checkpoints using lm-evaluation-harness.

This script loads a model from a distributed checkpoint and evaluates it on
standard benchmarks using the lm-evaluation-harness framework.

Usage:
    # Evaluate from a checkpoint
    python -m flame.eval_checkpoint \
        --checkpoint_dir /path/to/output/dir \
        --step 1000 \
        --config /path/to/config.yaml \
        --tokenizer /path/to/tokenizer \
        --tasks hellaswag,arc_easy,winogrande \
        --batch_size 8

    # Use converted HuggingFace checkpoint
    python -m flame.eval_checkpoint \
        --pretrained /path/to/hf/checkpoint \
        --tasks mmlu \
        --num_fewshot 5
"""

import argparse
import io
import json
import os
import sys
import tempfile
from datetime import timedelta
from pathlib import Path
from typing import Optional

import numpy as np

# Setup paths for local libraries
# eval_checkpoint.py is at: project_root/exp/flame/flame/eval_checkpoint.py

print(f"DEBUG: Adding to sys.path: {fla_path}")
print(f"DEBUG: Adding to sys.path: {flame_path}")

sys.path.insert(0, fla_path)
sys.path.insert(0, flame_path)

import fla  # noqa
import fla.models  # noqa - ensures all model configs are registered with AutoConfig
import custom_models.delta_net_2
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models

# Import lm-eval components
try:
    from lm_eval import evaluator
    from lm_eval.models.huggingface import HFLM
except ImportError:
    print("Error: lm-eval not installed. Please install it with: pip install lm-eval")
    sys.exit(1)


# Custom JSON encoder to handle numpy types and torch dtypes
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (torch.dtype, np.dtype)):
            return str(obj)
        return super().default(obj)


class FLAMECheckpointWrapper(HFLM):
    """
    Wrapper to make FLAME models loaded from checkpoints compatible with lm-eval-harness.
    """

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir: str,
        step: int,
        config_path: str,
        tokenizer_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_batch_size: int = 512,
        max_length: int = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        """
        Load a FLAME model from a distributed checkpoint.

        Args:
            checkpoint_dir: Directory containing checkpoints/step-{step} folder
            step: Checkpoint step to load
            config_path: Path to model config
            tokenizer_path: Path to tokenizer
            device: Device to load model on
            batch_size: Batch size for evaluation
            max_batch_size: Maximum batch size
            max_length: Maximum context length (truncation limit)
            dtype: Model dtype (float32, float16, bfloat16)
            trust_remote_code: Whether to trust remote code
        """
        logger.info(f"Loading model from checkpoint at step {step}")
        logger.info(f"Checkpoint directory: {checkpoint_dir}")

        # Load config
        logger.info(f"Loading config from {config_path}")
        config = AutoConfig.from_pretrained(config_path, trust_remote_code=trust_remote_code)

        # Load tokenizer
        logger.info(f"Loading tokenizer from {tokenizer_path}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=trust_remote_code
        )

        # Convert DCP to torch save format
        # Use /fast for temp directory to avoid disk space issues with large checkpoints
        tmpdir_base = "/tmp"
        os.makedirs(tmpdir_base, exist_ok=True)
        
        with tempfile.TemporaryDirectory(dir=tmpdir_base) as tmpdir:
            checkpoint_path_dcp = os.path.join(checkpoint_dir, f"checkpoints/step-{step}")
            if not os.path.exists(checkpoint_path_dcp):
                raise ValueError(
                    f"Checkpoint not found at {checkpoint_path_dcp}. "
                    f"Available checkpoints: {os.listdir(os.path.join(checkpoint_dir, 'checkpoints'))}"
                )

            checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
            logger.info(f"Converting DCP checkpoint to torch format: {checkpoint_path_dcp}")
            dcp_to_torch_save(checkpoint_path_dcp, checkpoint_path)

            # Initialize model
            logger.info("Initializing model from config")
            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(dtype, torch.bfloat16)

            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)

            # Load state dict
            logger.info("Loading state dict from checkpoint")
            # Add safe globals for checkpoint loading
            torch.serialization.add_safe_globals([timedelta, io.BytesIO])
            checkpoint_data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

            if "model" in checkpoint_data:
                state_dict = checkpoint_data["model"]
            else:
                state_dict = checkpoint_data

            # Load the state dict
            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                logger.warning(f"Unexpected keys: {unexpected_keys}")

            logger.info(f"Moving model to {device}")
            model = model.to(device)
            model.eval()

        # Create instance
        instance = cls.__new__(cls)
        HFLM.__init__(
            instance,
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            max_length=max_length,
            device=device,
        )
        return instance

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: str = "cuda",
        batch_size: int = 1,
        max_batch_size: int = 512,
        max_length: int = None,
        dtype: str = "bfloat16",
        trust_remote_code: bool = True,
    ):
        """
        Load a FLAME model from a HuggingFace-style pretrained checkpoint.

        Args:
            pretrained_path: Path to pretrained model (HF format)
            device: Device to load model on
            batch_size: Batch size for evaluation
            max_batch_size: Maximum batch size
            max_length: Maximum context length (truncation limit)
            dtype: Model dtype (float32, float16, bfloat16)
            trust_remote_code: Whether to trust remote code
        """
        logger.info(f"Loading pretrained model from {pretrained_path}")

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).to(device)
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_path, trust_remote_code=trust_remote_code
        )

        # Create instance
        instance = cls.__new__(cls)
        HFLM.__init__(
            instance,
            pretrained=model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            max_length=max_length,
            device=device,
        )
        return instance


def find_latest_checkpoint(checkpoint_dir: str) -> Optional[int]:
    """Find the latest checkpoint step in the checkpoint directory."""
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoints")
    if not os.path.exists(checkpoint_path):
        return None

    steps = []
    for item in os.listdir(checkpoint_path):
        if item.startswith("step-"):
            try:
                step = int(item.split("-")[1])
                steps.append(step)
            except (ValueError, IndexError):
                continue

    return max(steps) if steps else None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate FLAME models from checkpoints using lm-evaluation-harness"
    )

    # Model loading arguments
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Directory containing checkpoints/step-{step} folders",
    )
    model_group.add_argument(
        "--pretrained",
        type=str,
        help="Path to HuggingFace-style pretrained model",
    )

    parser.add_argument(
        "--step",
        type=int,
        default=-1,
        help="Checkpoint step to load (default: -1 for latest)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to model config (required for --checkpoint_dir)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Path to tokenizer (required for --checkpoint_dir, optional for --pretrained)",
    )

    # Evaluation arguments
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,arc_challenge,winogrande",
        help="Comma-separated list of tasks to evaluate on",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=512,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=None,
        help="Maximum context length for truncation (e.g. 10000). Leave unset for model default.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (for debugging)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="eval_results.json",
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        help="Log individual sample outputs",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        default=True,
        help="Trust remote code when loading models",
    )

    args = parser.parse_args()

    # Initialize logger
    init_logger()

    # Validate arguments
    if args.checkpoint_dir:
        if not args.config:
            parser.error("--config is required when using --checkpoint_dir")
        if not args.tokenizer:
            parser.error("--tokenizer is required when using --checkpoint_dir")

        # Find latest checkpoint if step is -1
        if args.step == -1:
            args.step = find_latest_checkpoint(args.checkpoint_dir)
            if args.step is None:
                raise ValueError(f"No checkpoints found in {args.checkpoint_dir}/checkpoints/")
            logger.info(f"Using latest checkpoint: step {args.step}")

    # Load model
    if args.checkpoint_dir:
        model_wrapper = FLAMECheckpointWrapper.from_checkpoint(
            checkpoint_dir=args.checkpoint_dir,
            step=args.step,
            config_path=args.config,
            tokenizer_path=args.tokenizer,
            device=args.device,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            max_length=args.max_length,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    else:  # pretrained
        model_wrapper = FLAMECheckpointWrapper.from_pretrained(
            pretrained_path=args.pretrained,
            device=args.device,
            batch_size=args.batch_size,
            max_batch_size=args.max_batch_size,
            max_length=args.max_length,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )

    # Parse tasks
    task_list = args.tasks.split(",")
    logger.info(f"Running evaluation on tasks: {task_list}")
    logger.info(f"Number of few-shot examples: {args.num_fewshot}")
    logger.info(f"Batch size: {args.batch_size}")
    if args.max_length:
        logger.info(f"Max context length: {args.max_length}")

    # Run evaluation
    results = evaluator.simple_evaluate(
        model=model_wrapper,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
        limit=args.limit,
        log_samples=args.log_samples,
    )

    # Add checkpoint metadata to results for permanent record
    checkpoint_metadata = {
        "evaluation_args": {
            "tasks": args.tasks,
            "num_fewshot": args.num_fewshot,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "device": args.device,
            "dtype": args.dtype,
        }
    }
    
    if args.checkpoint_dir:
        checkpoint_metadata["checkpoint_type"] = "DCP"
        checkpoint_metadata["checkpoint_dir"] = args.checkpoint_dir
        checkpoint_metadata["checkpoint_step"] = args.step
        checkpoint_metadata["checkpoint_path"] = os.path.join(args.checkpoint_dir, f"checkpoints/step-{args.step}")
        checkpoint_metadata["config_path"] = args.config
        checkpoint_metadata["tokenizer_path"] = args.tokenizer
    else:
        checkpoint_metadata["checkpoint_type"] = "HuggingFace"
        checkpoint_metadata["pretrained_path"] = args.pretrained
        checkpoint_metadata["config_path"] = os.path.join(args.pretrained, "config.json")
    
    results["checkpoint_metadata"] = checkpoint_metadata

    # Save results
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    logger.info(f"Results saved to {output_path}")
    logger.info(f"Checkpoint metadata saved in results file")

    # Print summary
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    if args.checkpoint_dir:
        print(f"Checkpoint type: DCP (Distributed)")
        print(f"Checkpoint dir:  {args.checkpoint_dir}")
        print(f"Checkpoint step: {args.step}")
        print(f"Checkpoint path: {os.path.join(args.checkpoint_dir, f'checkpoints/step-{args.step}')}")
        print(f"Config:          {args.config}")
    else:
        print(f"Checkpoint type: HuggingFace (Pretrained)")
        print(f"Model path:      {args.pretrained}")
        print(f"Config:          {os.path.join(args.pretrained, 'config.json')}")

    print(f"Tasks:           {args.tasks}")
    print(f"Few-shot:        {args.num_fewshot}")
    print(f"Max Length:      {args.max_length if args.max_length else 'Default (None)'}")
    print("=" * 70)

    for task_name, task_results in results["results"].items():
        print(f"\n{task_name.upper()}:")
        print("-" * 70)
        for metric, value in task_results.items():
            if isinstance(value, (int, float)) and not metric.endswith("_stderr"):
                stderr_key = f"{metric}_stderr"
                stderr = task_results.get(stderr_key, None)
                if stderr is not None:
                    print(f"  {metric:30s}: {value:.4f} ± {stderr:.4f}")
                else:
                    print(f"  {metric:30s}: {value:.4f}")

    print("\n" + "=" * 70)

    # Print aggregate score if available
    if "results" in results:
        all_accs = []
        for task_name, task_results in results["results"].items():
            # Try common accuracy metric names
            for acc_key in ["acc", "acc_norm", "accuracy"]:
                if acc_key in task_results:
                    all_accs.append(task_results[acc_key])
                    break

        if all_accs:
            avg_acc = sum(all_accs) / len(all_accs)
            print(f"Average Accuracy: {avg_acc:.4f}")
            print("=" * 70)

    logger.info("Evaluation completed successfully")


if __name__ == "__main__":
    main()