# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import argparse
import io
import os
import tempfile
from datetime import timedelta

import fla  # noqa
import fla.models  # noqa - ensures all model configs are registered with AutoConfig
import custom_models.delta_net_2
import torch
import torch.serialization
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save
from torchtitan.tools.logging import init_logger, logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models


@torch.inference_mode()
def save_pretrained(
    path: str,
    step: int,
    config: str,
    tokenizer: str,
    lora_rank: int = 0,
    lora_target_modules: str = None
):
    logger.info(f"Loading the config from {config}")
    config = AutoConfig.from_pretrained(config, trust_remote_code=True)

    logger.info(f"Saving the config to {path}")
    config.save_pretrained(path)
    logger.info(f"Loading the tokenizer from {tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
    logger.info(f"Saving the tokenizer to {path}")
    tokenizer.save_pretrained(path)

    tmp_base = "/tmp"
    os.makedirs(tmp_base, exist_ok=True)
    
    with tempfile.TemporaryDirectory(dir=tmp_base) as tmpdir:
        checkpoint = os.path.join(path, f'step-{step}')
        checkpoint_path = os.path.join(tmpdir, 'checkpoint.pt')
        logger.info(f"Saving the distributed checkpoint to {checkpoint_path}")
        dcp_to_torch_save(checkpoint, checkpoint_path)

        logger.info(f"Initializing the model from config\n{config}")
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
        # Add datetime.timedelta and io.BytesIO to safe globals
        torch.serialization.add_safe_globals([timedelta, io.BytesIO])
        state_dict = torch.load(checkpoint_path, map_location='cpu')['model']
        
        # Check if the state dict contains PEFT/LoRA keys
        is_lora = any("lora_" in key for key in state_dict.keys())
        
        if is_lora:
            logger.info("LoRA keys detected in checkpoint. Applying LoRA and merging...")
            from peft import LoraConfig, get_peft_model, TaskType
            
            if not lora_target_modules:
                # Try to infer target modules from state_dict keys
                # Example key: base_model.model.model.layers.0.attn.q_proj.lora_A.default.weight
                detected_targets = set()
                for key in state_dict.keys():
                    if ".lora_A." in key:
                        parts = key.split('.')
                        # Extract the module name (usually the part before .lora_A)
                        for i, p in enumerate(parts):
                            if p == "lora_A":
                                detected_targets.add(parts[i-1])
                lora_target_modules = list(detected_targets)
                logger.info(f"Inferred LoRA target modules: {lora_target_modules}")
            elif isinstance(lora_target_modules, str):
                lora_target_modules = [m.strip() for m in lora_target_modules.split(',')]

            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank if lora_rank > 0 else 8,
                lora_alpha=lora_rank * 2 if lora_rank > 0 else 16,
                target_modules=lora_target_modules,
            )
            model = get_peft_model(model, peft_config)
            
            logger.info("Loading LoRA state dict")
            model.load_state_dict(state_dict)
            
            logger.info("Merging LoRA weights into base model")
            model = model.merge_and_unload()
        else:
            logger.info("Loading standard state dict")
            model.load_state_dict(state_dict)

        logger.info(f"Saving the model to {path}")
        model.save_pretrained(path)


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser("Convert DCP format model weights to huggingface-style.")
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, required=True)
    parser.add_argument("--lora_rank", type=int, default=0)
    parser.add_argument("--lora_target_modules", type=str, default=None)
    args = parser.parse_args()
    save_pretrained(args.path, args.step, args.config, args.tokenizer, args.lora_rank, args.lora_target_modules)
