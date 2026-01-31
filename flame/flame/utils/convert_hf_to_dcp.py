# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as DCP
from transformers import AutoModelForCausalLM

# Add flash-linear-attention to path to use local version
workspace_root = Path(__file__).resolve().parents[4]  # Go up from flame/utils/convert_hf_to_dcp.py
sys.path.insert(0, str(workspace_root / "flash-linear-attention"))
sys.path.insert(0, str(workspace_root / "exp/flame"))

import fla  # noqa
import fla.models  # noqa - ensures all model configs are registered with AutoConfig
import flame.custom_models.delta_net_2
from torchtitan.tools.logging import init_logger, logger


@torch.inference_mode()
def convert_hf_weights(model: str, checkpoint: str):
    logger.info(f"Loading model from {model}")
    model = AutoModelForCausalLM.from_pretrained(model)
    state_dict = model.state_dict()

    logger.info(f"Writing to DCP at '{checkpoint}'")
    checkpoint.mkdir(parents=True, exist_ok=True)
    storage_writer = DCP.filesystem.FileSystemWriter(checkpoint, thread_count=8)
    # DCP.save({"model": state_dict}, storage_writer=storage_writer)
    DCP.save(state_dict, storage_writer=storage_writer)

if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser(description="Convert huggingface-style model weights to DCP format.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    convert_hf_weights(args.model, args.checkpoint)
