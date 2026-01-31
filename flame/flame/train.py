# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# CRITICAL: Add local FLA to path BEFORE any imports
#from pathlib import Path
#import sys
#workspace_root = Path(__file__).resolve().parents[3]  # Go up #from exp/flame/flame/train.py
#sys.path.insert(0, str(workspace_root / #"flash-linear-attention"))
#sys.path.insert(0, str(workspace_root / "exp/flame"))

import json
import os
import time
from datetime import timedelta
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        pass

import fla  # noqa
import fla.models  # noqa - ensures all model configs are registered with AutoConfig
import flame.custom_models.delta_net_2
import torch
import torch.nn as nn
import torch.nn.functional as F
from fla.modules.fused_linear_cross_entropy import FusedLinearCrossEntropyLoss
from fla.ops.utils import prepare_position_ids
from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.lr_scheduler import build_lr_schedulers
from torchtitan.components.metrics import build_device_memory_monitor, build_metrics_processor, ensure_pp_loss_visible
from torchtitan.components.optimizer import build_optimizers
from torchtitan.distributed import ParallelDims
from torchtitan.distributed import utils as dist_utils
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import TrainSpec, get_train_spec, register_train_spec
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import maybe_enable_memory_snapshot, maybe_enable_profiling
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import custom_models
from flame.components.checkpoint import TrainState
from flame.config_manager import JobConfig
from flame.data import build_dataloader, build_dataset
from flame.models.parallelize_fla import parallelize_fla
from flame.models.pipeline_fla import pipeline_fla
from flame.flame.tools.utils import get_nparams_and_flops

def build_tokenizer(job_config: JobConfig) -> AutoTokenizer:
    return AutoTokenizer.from_pretrained(job_config.model.tokenizer_path)

load_dotenv()

register_train_spec(
    TrainSpec(
        name="fla",
        cls=AutoModelForCausalLM,
        config=AutoConfig,
        parallelize_fn=parallelize_fla,
        pipelining_fn=pipeline_fla,
        build_optimizers_fn=build_optimizers,
        build_lr_schedulers_fn=build_lr_schedulers,
        build_dataloader_fn=build_dataloader,
        build_tokenizer_fn=build_tokenizer,
        build_loss_fn=build_cross_entropy_loss,
    )
)


# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    # used for colorful printing
    color = utils.NoColor if job_config.metrics.disable_color_printing else utils.Color

    if job_config.job.print_args:
        logger.info(
            f"{color.green}{json.dumps(job_config.to_dict(), indent=2, sort_keys=True)}{color.reset}"
        )

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )
    dist_utils.init_distributed(job_config)
    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = build_device_memory_monitor()
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0

    if parallel_dims.pp_enabled:
        raise NotImplementedError(
            "Pipeline parallelism is not supported in this version"
        )
        """
        ! TODO[flame]: We need to fix the pipeline parallelism for flame
        [x] Match the key of models' components with the actual naming
        [ ] Fix the post-init and tie-embedding for pipeline parallelism, HF's transformer automatically
            forces to tie if head is None, we need to handle this case
        [ ]
        """
        pp_mesh = world_mesh["pp"]

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        job_config.model.tokenizer_path,
        trust_remote_code=True,
        model_max_length=int(1e10),
    )
    logger.info(f"{tokenizer}")
    logger.info(
        f"Loading dataset {job_config.training.dataset}"
        f":{job_config.training.dataset_name}"
        if job_config.training.dataset_name is not None
        else ""
    )
    dataset = build_dataset(
        dataset=job_config.training.dataset,
        dataset_name=job_config.training.dataset_name,
        dataset_split=job_config.training.dataset_split,
        data_dir=job_config.training.data_dir,
        data_files=job_config.training.data_files,
        data_probs=job_config.training.data_probs,
        streaming=job_config.training.streaming,
        dp_degree=dp_degree,
        num_workers=job_config.training.num_workers,
        seed=job_config.training.seed,
    )

    logger.info("Building dataloader...")
    dataloader = build_dataloader(
        dataset=dataset,
        tokenizer=tokenizer,
        rank=dp_rank,
        world_size=dp_degree,
        batch_size=job_config.training.batch_size,
        seq_len=job_config.training.seq_len,
        context_len=job_config.training.context_len,
        varlen=job_config.training.varlen,
        num_workers=job_config.training.num_workers,
        pin_memory=job_config.training.pin_memory,
        persistent_workers=job_config.training.persistent_workers,
        snapshot_every_n_steps=job_config.checkpoint.interval,
    )

    logger.info(f"Loading model config from {job_config.model.config}")
    model_config = AutoConfig.from_pretrained(job_config.model.config)
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. disable fused norm if TP is enabled
    # 3. vocab size from tokenizer
    # 4. context_len base on inputs
    if parallel_dims.tp_enabled:
        if model_config.fuse_norm:
            logger.warning(
                f"{color.red}"
                f"Fused norm is not compatible with tensor parallelism. "
                f"Disabling it for now."
                f"{color.reset}"
            )
            model_config.fuse_norm = False
    if parallel_dims.loss_parallel_enabled:
        if model_config.fuse_linear_cross_entropy:
            logger.warning(
                f"{color.red}"
                f"Loss parallel enabled. Disabling fused cross entropy for now."
                f"{color.reset}"
            )
            model_config.fuse_linear_cross_entropy = False
    model_config.vocab_size = max(tokenizer.vocab_size, model_config.vocab_size)

    logger.info(
        f"Building model from the config\n{color.green}{model_config}{color.reset}"
    )
    with torch.device("meta"):
        model = AutoModelForCausalLM.from_config(model_config)
        if (
            getattr(model_config, "fuse_linear_cross_entropy", False)
            and FusedLinearCrossEntropyLoss is not None
        ):
            model.criterion = FusedLinearCrossEntropyLoss(
                num_chunks=8 // parallel_dims.tp
            )
        # defer weight initialization until after parallelisms are applied
        model.apply(lambda m: setattr(m, "_is_hf_initialized", False))
    logger.info(f"{color.blue}\n{model}{color.reset}\n")

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # calculate model size and flops per token
    model_param_count, num_flops_per_token = get_nparams_and_flops(
        model, model_config, job_config.training.context_len
    )

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
    else:
        init_device = device_type

    # Defer compilation if loading from a checkpoint to avoid pickling issues
    #will_load_checkpoint = (
    #    job_config.checkpoint.enable_checkpoint and 
    #    (job_config.checkpoint.load_step >= 0 or job_config.checkpoint.initial_load_path is not None)
    #)
    #defer_compile = will_load_checkpoint and job_config.training.compile
    
    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config) # , defer_compile=defer_compile)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.post_init()
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(parallel_dims, job_config, color)
    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config) #, defer_compile=defer_compile)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.post_init()
        model.train()

        model_parts = [model]

    freeze_params = getattr(job_config.model, 'freeze_non_target', False)
    use_lora = getattr(job_config.model, 'use_lora', False)
    
    if freeze_params and use_lora:
        raise ValueError("Cannot use both --model.freeze_non_target and --model.use_lora. Choose one finetuning strategy.")
    
    if freeze_params:
        target_modules = getattr(job_config.model, 'target_modules', ["q_proj", "k_proj"])
        logger.info(f"{color.cyan}Freezing all parameters except: {target_modules}{color.reset}")
        
        for model_part in model_parts:
            for name, param in model_part.named_parameters():
                if any(target in name for target in target_modules):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        
        # Log stats to verify memory savings
        trainable_params = sum(p.numel() for m in model_parts for p in m.parameters() if p.requires_grad)
        total_params = sum(p.numel() for m in model_parts for p in m.parameters())
        logger.info(f"{color.green}✓ Freezing complete. Trainable: {trainable_params:,} / {total_params:,}{color.reset}. Relative -> {100.0 * trainable_params / total_params:.2f}%")
    
    # Note: LoRA is applied AFTER checkpoint loading to avoid key mismatch issues
    # See the LoRA application block after checkpoint.load()

    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    # Note: For LoRA, we build optimizers for the base model first (for checkpoint loading),
    # then rebuild them after applying LoRA adapters
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config, ft_manager)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=dataloader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert world_size == 1, (
            "Must create seed checkpoint using a single device, to disable sharding"
        )
        assert job_config.checkpoint.enable_checkpoint, (
            "Must enable checkpointing when creating a seed checkpoint"
        )
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint_loaded = checkpoint.load(step=job_config.checkpoint.load_step)
    
    if not checkpoint_loaded and job_config.checkpoint.initial_load_path:
        raise RuntimeError(
            f"Failed to load checkpoint from {job_config.checkpoint.initial_load_path}. "
            "Model has random weights! Check that the path exists and is valid."
        )
    
    if checkpoint_loaded:
        logger.info("✓ Checkpoint loaded successfully - model has pretrained weights")
    else:
        logger.warning("⚠ No checkpoint loaded - model has randomly initialized weights!")

    # Apply LoRA AFTER checkpoint loading to avoid key mismatch
    if use_lora:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "PEFT is required for LoRA finetuning. Install it with: pip install peft"
            )
        
        lora_target_modules = getattr(job_config.model, 'lora_target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])
        # Handle both string (comma-separated) and list formats
        if isinstance(lora_target_modules, str):
            if lora_target_modules.strip() == "all-linear":
                lora_target_modules = "all-linear"
            else:
                lora_target_modules = [m.strip() for m in lora_target_modules.split(',')]
        
        lora_rank = getattr(job_config.model, 'lora_rank', 8)
        lora_alpha = getattr(job_config.model, 'lora_alpha', 16)
        lora_dropout = getattr(job_config.model, 'lora_dropout', 0.05)
        lora_bias = getattr(job_config.model, 'lora_bias', "none")
        
        logger.info(f"{color.cyan}Applying LoRA to modules: {lora_target_modules}{color.reset}")
        logger.info(f"{color.cyan}  LoRA config: rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout}, bias={lora_bias}{color.reset}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            target_modules=lora_target_modules,
        )
        
        # Apply LoRA to each model part
        new_model_parts = []
        for i, model_part in enumerate(model_parts):
            # Log all Linear modules BEFORE applying LoRA to see what exists
            logger.info(f"{color.cyan}All nn.Linear modules in the model:{color.reset}")
            for name, module in model_part.named_modules():
                if isinstance(module, nn.Linear):
                    logger.info(f"  {name}: {module.in_features} -> {module.out_features}")
            
            peft_model = get_peft_model(model_part, lora_config)
            
            # Explicitly unfreeze structural layers needed for recovery
            # 1. Normalization layers: Crucial for recalibrating feature distributions
            # 2. Convolutions: Structurally modified during pruning
            # 3. Biases: Provides "steering" with minimal overhead (BitFit)
            logger.info(f"{color.cyan}Unfreezing structural layers for training...{color.reset}")
            unfrozen_params = {"conv1d": [], "norm": [], "bias": []}
            
            for name, param in peft_model.named_parameters():
                unfreeze = False
                category = None
                
                if 'conv1d' in name:
                    unfreeze = True
                    category = "conv1d"
                elif 'norm' in name:
                    unfreeze = True
                    category = "norm"
                elif 'bias' in name:
                    unfreeze = True
                    category = "bias"
                
                if unfreeze:
                    param.requires_grad = True
                    if name not in unfrozen_params[category]:
                        unfrozen_params[category].append(name)
            
            # Log results for each category
            for cat, names in unfrozen_params.items():
                if names:
                    logger.info(f"{color.green}✓ Unfrozen {cat} parameters: {len(names)}{color.reset}")
                    for name in names[:2]: # Show first 2 as examples
                        logger.info(f"  - {name}")
                    if len(names) > 2:
                        logger.info(f"  - ... ({len(names)-2} more)")
                else:
                    logger.warning(f"⚠ No {cat} parameters found to unfreeze!")

            peft_model.print_trainable_parameters()
            
            # Log all modules that have LoRA applied (check for lora_A/lora_B attributes)
            logger.info(f"{color.cyan}LoRA applied to the following modules:{color.reset}")
            lora_count = 0
            for name, module in peft_model.named_modules():
                # PEFT wraps linear layers and adds lora_A/lora_B as sub-modules
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    lora_count += 1
                    logger.info(f"  {name}")
            logger.info(f"{color.cyan}Total LoRA-adapted layers: {lora_count}{color.reset}")
            
            new_model_parts.append(peft_model)
        model_parts = new_model_parts
        # Update the model variable for non-PP forward pass
        if not parallel_dims.pp_enabled:
            model = model_parts[0]
        
        # Log LoRA stats
        trainable_params = sum(p.numel() for m in model_parts for p in m.parameters() if p.requires_grad)
        total_params = sum(p.numel() for m in model_parts for p in m.parameters())
        logger.info(f"{color.green}✓ LoRA applied. Trainable: {trainable_params:,} / {total_params:,}{color.reset}. Relative -> {100.0 * trainable_params / total_params:.2f}%")
        
        # Rebuild optimizers with LoRA parameters (only LoRA params have requires_grad=True)
        optimizers = train_spec.build_optimizers_fn(model_parts, job_config, ft_manager)
        lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
        optimizers.register_step_post_hook(
            lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
        )
        
        # Update checkpoint manager with new model_parts and optimizers for future saves
        # The model is wrapped in ModelWrapper and stored in checkpoint.states
        from torchtitan.components.checkpoint import ModelWrapper
        checkpoint.states["model"] = ModelWrapper(model_parts)
        checkpoint.states["optimizer"] = optimizers
        checkpoint.states["lr_scheduler"] = lr_schedulers
        
        # Reset the CPU offload cache since model structure changed (LoRA added new params)
        # This is needed for async checkpointing with pinned memory
        if hasattr(checkpoint, 'cpu_offload_state_dict'):
            checkpoint.cpu_offload_state_dict = None
        
        # For fault tolerance, reinitialize optimizer cache
        if checkpoint.ft_manager is not None:
            optimizers.init_cache_state_dict()

    # Apply torch.compile after checkpoint loading if it was deferred
    #if defer_compile:
    #    from flame.models.parallelize_fla import apply_compile
    #    logger.info("Applying torch.compile after checkpoint loading")
    #    for m in model_parts:
    #        apply_compile(m)
    
    metric_logger = build_metrics_processor(job_config, parallel_dims)
    # Set dependent attributes for metric_logger
    metric_logger.num_flops_per_token = num_flops_per_token
    metric_logger.optimizers = optimizers  # Pass optimizers if needed by logger logic
    metric_logger.lr_schedulers = (
        lr_schedulers  # Pass schedulers if needed by logger logic
    )

    # plot losses loaded from checkpoint (if any) to TensorBoard
    # NOTE: Loss info after the last log step before checkpoint saving will not be ploted.
    #       This can be avoided by setting checkpoint.interval to be a multiple of metrics.log_freq
    if train_state.step > 0 and len(metric_logger.data_loading_times) > 0:
        for idx, step in enumerate(train_state.log_steps):
            metric_logger.log(
                step,
                global_avg_loss=train_state.global_avg_losses[idx],
                global_max_loss=train_state.global_max_losses[idx],
            )

    # Load teacher model if specified
    teacher_model = None
    distillation_loss_weight = getattr(job_config.training, 'distillation_loss_weight', 0.0)
    distillation_temperature = getattr(job_config.training, 'distillation_temperature', 1.0)
    
    if hasattr(job_config.model, 'teacher_model_path') and job_config.model.teacher_model_path:
        logger.info(f"{color.cyan}Loading teacher model for distillation from {job_config.model.teacher_model_path}{color.reset}")
        
        teacher_config = None
        if hasattr(job_config.model, 'teacher_model_config') and job_config.model.teacher_model_config:
            teacher_config = AutoConfig.from_pretrained(job_config.model.teacher_model_config)
            
        with torch.device("meta"):
            # If config is provided, use it
            if teacher_config:
                teacher_model = AutoModelForCausalLM.from_config(teacher_config)
            else:
                 # Otherwise load from path directly (this might be slow if large model)
                 # Better to use from_pretrained with device_map="cpu" then move to device
                 pass
        
        # Load the actual model
        teacher_model = AutoModelForCausalLM.from_pretrained(
            job_config.model.teacher_model_path,
            config=teacher_config,
            torch_dtype=model_config.torch_dtype,
            trust_remote_code=True,
        )
        
        # Move to device and freeze
        teacher_model.to(device)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
            
        logger.info(f"{color.green}✓ Teacher model loaded and frozen.{color.reset}")
        logger.info(f"  Distillation Weight: {distillation_loss_weight}")
        logger.info(f"  Temperature: {distillation_temperature}")

    data_iterator = iter(dataloader)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )
    maybe_enable_amp = dist_utils.maybe_enable_amp(
        parallel_dims,
        job_config.training.mixed_precision_param,
        device_type,
    )

    # variables used to keep info for metrics logging
    device_memory_monitor.reset_peak_stats()

    global_batch_size = (
        job_config.training.batch_size
        * dp_degree
        * job_config.training.gradient_accumulation_steps
    )
    num_tokens_per_step = global_batch_size * job_config.training.seq_len
    # train loop
    logger.info(f"{color.red}***** Running training *****{color.reset}")
    logger.info(f"{color.green}  Training starts at step {train_state.step + 1}")
    logger.info(
        f"{color.green}  Number of tokens per sequence = {job_config.training.seq_len:,}"
    )
    logger.info(
        f"{color.green}  Gradient Accumulation steps = {job_config.training.gradient_accumulation_steps}"
    )
    logger.info(
        f"{color.green}  Instantaneous batch size (per device) = {job_config.training.batch_size:,}"
    )
    logger.info(
        f"{color.green}  Global batch size (w. parallel, distributed & accumulation) = {global_batch_size:,}"
        f" ({num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Total optimization steps = {job_config.training.steps:,} "
        f"({job_config.training.steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Warmup steps = {job_config.lr_scheduler.warmup_steps:,}"
        f" ({job_config.lr_scheduler.warmup_steps * num_tokens_per_step:,} tokens)"
    )
    logger.info(
        f"{color.green}  Number of parameters = {model_param_count:,} {color.reset}"
    )

    with (
        maybe_enable_profiling(
            job_config, global_step=train_state.step
        ) as torch_profiler,
        maybe_enable_memory_snapshot(
            job_config, global_step=train_state.step
        ) as memory_profiler,
    ):
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            optimizers.zero_grad()

            losses = []
            # do gradient accumulation if enabled
            for _ in range(job_config.training.gradient_accumulation_steps):
                # get batch
                data_load_start = time.perf_counter()
                batch = next(data_iterator)
                input_ids, labels = batch["input_ids"], batch["labels"]

                # Update metrics processor state before forward/backward
                metric_logger.ntokens_since_last_log += labels.numel()
                metric_logger.data_loading_times.append(
                    time.perf_counter() - data_load_start
                )

                input_ids = input_ids.to(device_type)

                """
                TODO[flame]: We need to carefully handle the position_ids for TP/CP
                Depending on the Models'PE, the position_ids might be different.

                e.g. for TP
                    For RoPE, all ranks have the same position_ids. [FOR HF model]
                    For sinusoidal, each rank has the coresponding chunked  position_ids. [FOR HF model]

                e.g. for CP, [optional_context_parallel_ctx shoudl automatically distbute the position_ids]
                    Each rank has the coresponding chunked position_ids. [FOR All model]

                """
                labels = labels.to(device_type)
                cu_seqlens = (
                    batch["cu_seqlens"].to(device_type)
                    if "cu_seqlens" in batch
                    else None
                )
                if cu_seqlens is not None:
                    position_ids = prepare_position_ids(cu_seqlens).to(torch.int32)
                else:
                    position_ids = (
                        torch.arange(0, input_ids.shape[1], device=device_type)
                        .repeat(input_ids.shape[0], 1)
                        .to(torch.int32)
                    )
                # apply context parallelism if cp is enabled
                # ensure CP handles the separate freqs_cis buffer for each pp stage
                optional_context_parallel_ctx = (
                    dist_utils.create_context_parallel_ctx(
                        cp_mesh=world_mesh["cp"],
                        cp_buffers=[input_ids, labels, position_ids],
                        cp_seq_dims=[1, 1, 1],
                        cp_no_restore_buffers={input_ids, labels, position_ids},
                        cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                    )
                    if parallel_dims.cp_enabled
                    else None
                )

                # #! TODO[flame], we should distribute the position_ids as well with CP
                if parallel_dims.pp_enabled:
                    raise NotImplementedError(
                        "Pipeline parallelism is not supported in this version"
                    )
                    # Pipeline Parallel forward / backward inside step() call
                    with train_context(optional_context_parallel_ctx):
                        targets, losses = (
                            (labels, []) if has_last_stage else (None, None)
                        )

                        if has_first_stage:
                            pp_schedule.step(input_ids, target=targets, losses=losses)
                        else:
                            pp_schedule.step(target=targets, losses=losses)

                    # accumulate losses across pipeline microbatches
                    # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                    loss = (
                        torch.mean(torch.stack(losses)).to(device)
                        if has_last_stage
                        else torch.tensor([-1.0], device=device)
                    )
                else:
                    # Non-PP forward / backward
                    with train_context(optional_context_parallel_ctx):
                        with maybe_enable_amp:
                            output = model(
                                input_ids=input_ids,
                                labels=labels,
                                position_ids=position_ids,
                                cu_seqlens=cu_seqlens,
                        )
                        loss = (
                            output.loss
                            / job_config.training.gradient_accumulation_steps
                        )
                        
                        # Add distillation loss if teacher is present
                        if teacher_model is not None and distillation_loss_weight > 0:
                            with torch.no_grad():
                                teacher_output = teacher_model(
                                    input_ids=input_ids,
                                    position_ids=position_ids,
                                    cu_seqlens=cu_seqlens,
                                )

                            # Standard KD loss: KL(softmax(student/T), softmax(teacher/T)) * T^2
                            # We compute LogSoftmax for student and Softmax for teacher
                            student_logits = output.logits
                            teacher_logits = teacher_output.logits
                            
                            # Flatten to [B*L, V] so 'batchmean' averages over all tokens
                            # otherwise it only divides by Batch size (B), resulting in sum over Length (L)
                            student_logits_flat = student_logits.view(-1, student_logits.size(-1))
                            teacher_logits_flat = teacher_logits.view(-1, teacher_logits.size(-1))
                            
                            # KL Divergence expects log_target=False by default (target is probabilities)
                            # But here we use log_probs for input
                            
                            kd_loss = F.kl_div(
                                F.log_softmax(student_logits_flat / distillation_temperature, dim=-1),
                                F.softmax(teacher_logits_flat / distillation_temperature, dim=-1),
                                reduction='batchmean',
                                log_target=False
                            ) * (distillation_temperature ** 2)
                            
                            loss += (kd_loss * distillation_loss_weight) / job_config.training.gradient_accumulation_steps

                        loss.backward()

                losses.append(loss)
            loss = sum(losses)

            # clip gradients
            grad_norm = dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters() if p.requires_grad],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            if job_config.training.skip_nan_inf and (
                grad_norm.isnan() or grad_norm.isinf()
            ):
                logger.warning(
                    f"Skipping optimizer step - detected invalid gradient norm: {grad_norm:.4f}"
                )
                optimizers.zero_grad()
                train_state.skipped_step += 1
            else:
                optimizers.step()
            lr_schedulers.step()

            # log metrics - Use MetricsProcessor
            if metric_logger.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    # Use dist_mean/max on the accumulated loss for the step
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                        dist_utils.dist_max(
                            loss,
                            world_mesh["dp_cp"],
                        ),
                    )
                else:
                    # Scale back the loss before logging
                    global_avg_loss = global_max_loss = loss.item()

                # Update train state tokens and elapsed time
                time_now = time.perf_counter()
                time_delta = (
                    time_now - metric_logger.time_last_log
                )  # Use metric_logger's time
                train_state.token += (
                    metric_logger.ntokens_since_last_log  # Use tokens tracked by metric_logger
                    * parallel_dims.world_size
                    / parallel_dims.non_data_parallel_size
                )
                train_state.elapsed += timedelta(seconds=time_delta)
                train_state.log_steps.append(train_state.step)
                train_state.global_avg_losses.append(global_avg_loss)
                train_state.global_max_losses.append(global_max_loss)

                # Log using the metric processor
                last_lr = lr_schedulers.schedulers[0].get_last_lr()[0]
                eta = (
                    train_state.elapsed
                    * (job_config.training.steps - train_state.step)
                    / train_state.step
                )
                metric_logger.log(
                    train_state.step,
                    global_avg_loss,
                    global_max_loss,
                    extra_metrics={
                        "optimizer/lr": last_lr,
                        "optimizer/grad_norm": grad_norm.item(),
                        "optimizer/skipped_step": train_state.skipped_step,
                    },
                )

                logger.info(
                    f"{color.blue}lr: {last_lr:.4e} gnorm: {grad_norm:5.2f} "
                    f"{color.magenta}[{str(train_state.elapsed).split('.')[0]:>8}<{str(eta).split('.')[0]:>8}]{color.reset}"
                )

            checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    # Merge LoRA weights back into base model and save final merged checkpoint
    if use_lora:
        logger.info("Merging LoRA weights into base model...")
        merged_model_path = os.path.join(job_config.job.dump_folder, "merged_model")
        
        for i, model_part in enumerate(model_parts):
            # Merge LoRA weights into base model
            merged_model = model_part.merge_and_unload()
            
            if torch.distributed.get_rank() == 0:
                # Save the merged model (only rank 0 saves to avoid conflicts)
                merged_model.save_pretrained(merged_model_path)
                tokenizer.save_pretrained(merged_model_path)
                logger.info(f"Merged model saved to: {merged_model_path}")

    metric_logger.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()
