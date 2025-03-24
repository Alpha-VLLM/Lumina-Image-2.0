# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for Lumina-T2I using PyTorch FSDP.
"""
import argparse
from collections import OrderedDict, defaultdict
import contextlib
from copy import deepcopy
from datetime import datetime
import functools
from functools import lru_cache, partial
import json
import logging
import os
import random
import socket
import threading
from time import time
import warnings
import requests
import torch.nn.functional as F

from PIL import Image
# import cairosvg
from diffusers import AutoencoderKL
import fairscale.nn.model_parallel.initialize as fs_init
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import trange
import wandb
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
from liger_kernel.transformers import AutoLigerKernelForCausalLM

from data import MyDataset
from data.data_reader import WorldModelItemProcessor
from imgproc import generate_crop_size_list, var_center_crop
import models
from parallel import distributed_init, get_intra_node_process_group
from transport import create_transport
from util.misc import SmoothedValue
from vae import LatentVideoVAEArgs, build_vae


#############################################################################
#                           Training Helper Functions                       #
#############################################################################

def apply_average_pool(latent, factor):
    """
    Apply average pooling to downsample the latent.

    Args:
        latent (torch.Tensor): Latent tensor with shape (1, C, H, W).
        factor (int): Downsampling factor.

    Returns:
        torch.Tensor: Downsampled latent tensor.
    """
    return F.avg_pool2d(latent, kernel_size=factor, stride=factor)

def dataloader_collate_fn(samples):
    image = [x[0] for x in samples]
    caps = [x[1] for x in samples]
    return image, caps


def get_train_sampler(dataset, rank, world_size, global_batch_size, max_steps, resume_step, seed):
    sample_indices = torch.empty([max_steps * global_batch_size // world_size], dtype=torch.long)
    epoch_id, fill_ptr, offs = 0, 0, 0
    while fill_ptr < sample_indices.size(0):
        g = torch.Generator()
        g.manual_seed(seed + epoch_id)
        epoch_sample_indices = torch.randperm(len(dataset), generator=g)
        epoch_id += 1
        epoch_sample_indices = epoch_sample_indices[(rank + offs) % world_size :: world_size]
        offs = (offs + world_size - len(dataset) % world_size) % world_size
        epoch_sample_indices = epoch_sample_indices[: sample_indices.size(0) - fill_ptr]
        sample_indices[fill_ptr : fill_ptr + epoch_sample_indices.size(0)] = epoch_sample_indices
        fill_ptr += epoch_sample_indices.size(0)
    return sample_indices[resume_step * global_batch_size // world_size :].tolist()


@torch.no_grad()
def update_ema(ema_model, model, decay=0.95):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())
    assert set(ema_params.keys()) == set(model_params.keys())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def setup_lm_fsdp_sync(model: nn.Module) -> FSDP:
    # LM FSDP always use FULL_SHARD among the node.
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in list(model.layers),
        ),
        process_group=get_intra_node_process_group(),
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=MixedPrecision(
            param_dtype=next(model.parameters()).dtype,
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()
    return model


def setup_fsdp_sync(model: nn.Module, args: argparse.Namespace) -> FSDP:
    model = FSDP(
        model,
        auto_wrap_policy=functools.partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda m: m in model.get_fsdp_wrap_module_list(),
        ),
        process_group=fs_init.get_data_parallel_group(),
        sharding_strategy={
            "fsdp": ShardingStrategy.FULL_SHARD,
            "sdp": ShardingStrategy.SHARD_GRAD_OP,
        }[args.data_parallel],
        mixed_precision=MixedPrecision(
            param_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.precision],
            reduce_dtype={
                "fp32": torch.float,
                "tf32": torch.float,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[args.grad_precision or args.precision],
        ),
        device_id=torch.cuda.current_device(),
        sync_module_states=True,
        limit_all_gathers=True,
        use_orig_params=True,
    )
    torch.cuda.synchronize()

    return model


def setup_mixed_precision(args):
    if args.precision == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif args.precision in ["bf16", "fp16", "fp32"]:
        pass
    else:
        raise NotImplementedError(f"Unknown precision: {args.precision}")


# Adapted from pipelines.StableDiffusionXLPipeline.encode_prompt
def encode_prompt(prompt_batch, text_encoder, tokenizer, proportion_empty_prompts, is_train=True):
    captions = []
    for caption in prompt_batch:
        if random.random() < proportion_empty_prompts:
            captions.append("")
        elif isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])

    with torch.no_grad():
        text_inputs = tokenizer(
            captions,
            padding=True,
            pad_to_multiple_of=8,
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )

        text_input_ids = text_inputs.input_ids
        prompt_masks = text_inputs.attention_mask

        prompt_embeds = text_encoder(
            input_ids=text_input_ids.to(text_encoder.device),
            attention_mask=prompt_masks.to(text_encoder.device),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks


#############################################################################
#                                Training Loop                              #
#############################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    distributed_init(args)

    dp_world_size = fs_init.get_data_parallel_world_size()
    dp_rank = fs_init.get_data_parallel_rank()

    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    setup_mixed_precision(args)
    print(f"Starting rank={rank}, seed={seed}, "
          f"world_size={dist.get_world_size()}.")

    # Setup an experiment folder:
    os.makedirs(args.results_dir, exist_ok=True)
    checkpoint_dir = os.path.join(args.results_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if rank == 0:
        logger = create_logger(args.results_dir)
        logger.info(f"Experiment directory: {args.results_dir}")
        
        # Initialize wandb if enabled
        if args.use_wandb:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name or f"lumina-t2i-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=args.__dict__,
                resume="allow" if args.resume else None,
                id=args.wandb_id,
            )
            logger.info(f"Initialized wandb run: {wandb_run.name} (ID: {wandb_run.id})")
        else:
            wandb_run = None
    else:
        logger = create_logger(None)
        wandb_run = None

    logger.info("Training arguments: " + json.dumps(args.__dict__, indent=2))

    logger.info(f"Setting-up language model: google/gemma-2-2b")

    # create tokenizers
    # Load the tokenizers
    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    tokenizer.padding_side = "right"

    # create text encoders
    # text_encoder
    if args.text_encoder == "Qwen/Qwen2.5-VL-3B-Instruct":
        text_encoder = AutoModel.from_pretrained(
            args.text_encoder,
            torch_dtype=torch.bfloat16,
        ).cuda()
    else:
        text_encoder =  AutoLigerKernelForCausalLM.from_pretrained(
            args.text_encoder,
            torch_dtype=torch.bfloat16,
        )
        text_encoder = text_encoder.model.cuda()

    text_encoder = setup_lm_fsdp_sync(text_encoder)
    logger.info(f"text encoder: {type(text_encoder)}")
    cap_feat_dim = text_encoder.config.hidden_size

    # Create model:
    model = models.__dict__[args.model](
        in_channels=16,
        qk_norm=args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    logger.info(f"DiT Parameters: {model.parameter_count():,}")
    model_patch_size = model.patch_size

    if args.auto_resume and args.resume is None:
        try:
            existing_checkpoints = os.listdir(checkpoint_dir)
            if len(existing_checkpoints) > 0:
                existing_checkpoints.sort()
                args.resume = os.path.join(checkpoint_dir, existing_checkpoints[-1])
        except Exception:
            pass
        if args.resume is not None:
            logger.info(f"Auto resuming from: {args.resume}")

    # Note that parameter initialization is done within the DiT constructor
    model_ema = deepcopy(model)
    if args.resume:
        if dp_rank == 0:  # other ranks receive weights in setup_fsdp_sync
            logger.info(f"Resuming model weights from: {args.resume}")
            model.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
            logger.info(f"Resuming ema weights from: {args.resume}")
            model_ema.load_state_dict(
                torch.load(
                    os.path.join(
                        args.resume,
                        f"consolidated_ema.{0:02d}-of-{1:02d}.pth",
                    ),
                    map_location="cpu",
                ),
                strict=True,
            )
    elif args.init_from:
        if dp_rank == 0:
            logger.info(f"Initializing model weights from: {args.init_from}")
            state_dict = torch.load(
                os.path.join(
                    args.init_from,
                    f"consolidated.{0:02d}-of-{1:02d}.pth",
                ),
                map_location="cpu",
            )

            size_mismatch_keys = []
            model_state_dict = model.state_dict()
            for k, v in state_dict.items():
                if k in model_state_dict and model_state_dict[k].shape != v.shape:
                    size_mismatch_keys.append(k)
            for k in size_mismatch_keys:
                del state_dict[k]
            del model_state_dict

            missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
            missing_keys_ema, unexpected_keys_ema = model_ema.load_state_dict(state_dict, strict=False)
            del state_dict
            assert set(missing_keys) == set(missing_keys_ema)
            assert set(unexpected_keys) == set(unexpected_keys_ema)
            logger.info("Model initialization result:")
            logger.info(f"  Size mismatch keys: {size_mismatch_keys}")
            logger.info(f"  Missing keys: {missing_keys}")
            logger.info(f"  Unexpeected keys: {unexpected_keys}")
    dist.barrier()

    # checkpointing (part1, should be called before FSDP wrapping)
    if args.checkpointing:
        checkpointing_list = list(model.get_checkpointing_wrap_module_list())
        checkpointing_list_ema = list(model_ema.get_checkpointing_wrap_module_list())
    else:
        checkpointing_list = []
        checkpointing_list_ema = []

    model = setup_fsdp_sync(model, args)
    model_ema = setup_fsdp_sync(model_ema, args)

    # checkpointing (part2, after FSDP wrapping)
    if args.checkpointing:
        logger.info("apply gradient checkpointing")
        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list,
        )
        apply_activation_checkpointing(
            model_ema,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: submodule in checkpointing_list_ema,
        )

    logger.info(f"model:\n{model}\n")
    
    vae_args = LatentVideoVAEArgs(
        model_name="COSMOS-CI",
        pretrained_model_name_or_path="/mnt/pollux/checkpoints/Cosmos/Cosmos-Tokenize1-CI8x8-360p",
        model_dtype="bfloat16",
    )
    vae = build_vae(vae_args).to(device)

    logger.info("AdamW eps 1e-15 betas (0.9, 0.95)")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd, eps=1e-15, betas=(0.9, 0.95))
    if args.resume:
        opt_state_world_size = len(
            [x for x in os.listdir(args.resume) if x.startswith("optimizer.") and x.endswith(".pth")]
        )
        assert opt_state_world_size == dist.get_world_size(), (
            f"Resuming from a checkpoint with unmatched world size "
            f"({dist.get_world_size()} vs. {opt_state_world_size}) "
            f"is currently not supported."
        )
        logger.info(f"Resuming optimizer states from: {args.resume}")
        opt.load_state_dict(
            torch.load(
                os.path.join(
                    args.resume,
                    f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth",
                ),
                map_location="cpu",
            )
        )
        for param_group in opt.param_groups:
            param_group["lr"] = args.lr
            param_group["weight_decay"] = args.wd

        with open(os.path.join(args.resume, "resume_step.txt")) as f:
            resume_step = int(f.read().strip())
    else:
        resume_step = 0

    # Setup data:
    data_collection = {}
    for train_res in [args.resolution]:
        logger.info(f"Creating data for resolution {train_res}")

        global_bsz = getattr(args, f"global_bsz")
        local_bsz = global_bsz // dp_world_size  # todo caution for sequence parallel
        micro_bsz = getattr(args, f"micro_bsz")
        assert global_bsz % dp_world_size == 0, "Batch size must be divisible by data parallel world size."
        logger.info(f"Global bsz: {global_bsz} Local bsz: {local_bsz} Micro bsz: {micro_bsz}")

        patch_size = 8 * model_patch_size
        logger.info(f"patch size: {patch_size}")
        max_num_patches = round((train_res / patch_size) ** 2)
        logger.info(f"Limiting number of patches to {max_num_patches}.")
        crop_size_list = generate_crop_size_list(max_num_patches, patch_size, max_ratio=2.0)
        logger.info("List of crop sizes:")
        for i in range(0, len(crop_size_list), 6):
            logger.info(" " + "".join([f"{f'{w} x {h}':14s}" for w, h in crop_size_list[i : i + 6]]))
        image_transform = transforms.Compose(
            [
                transforms.Lambda(functools.partial(var_center_crop, crop_size_list=crop_size_list, random_top_k=1)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        dataset = MyDataset(
            args.data_path,
            item_processor=WorldModelItemProcessor(image_transform),
            use_cache=args.cache_data_on_disk,
        )
        num_samples = global_bsz * args.max_steps
        logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
        logger.info(f"Total # samples to consume: {num_samples:,} " f"({num_samples / len(dataset):.2f} epochs)")
        sampler = get_train_sampler(
            dataset,
            dp_rank,
            dp_world_size,
            global_bsz,
            args.max_steps,
            resume_step,
            args.global_seed + train_res * 100,  # avoid same sampling for different resolutions
        )
        loader = DataLoader(
            dataset,
            batch_size=local_bsz,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=dataloader_collate_fn,
            prefetch_factor=2,
        )

        # default: 1000 steps, linear noise schedule
        transport = create_transport(
            "Linear",
            "velocity",
            None,
            None,
            None,
            snr_type=args.snr_type,
            do_shift=not args.no_shift,
            seq_len=(train_res // 16) ** 2,
        )  # default: velocity;

        data_collection[train_res] = {
            "loader": loader,
            "loader_iter": iter(loader),
            "global_bsz": global_bsz,
            "local_bsz": local_bsz,
            "micro_bsz": micro_bsz,
            "metrics": defaultdict(lambda: SmoothedValue(args.log_every)),
            "transport": transport,
        }

    # Prepare models for training:
    model.train()

    # Variables for monitoring/logging purposes:

    logger.info(f"Training for {args.max_steps:,} steps...")

    for step in trange(resume_step, args.max_steps):
        start_time = time()
        for train_res, data_pack in data_collection.items():
            x, caps = next(data_pack["loader_iter"])
            x = [img.to(device, non_blocking=True) for img in x]

            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                for i, img in enumerate(x):
                    x[i] = vae.encode(img[None])[0]

            with torch.no_grad():
                cap_feats, cap_mask = encode_prompt(caps, text_encoder, tokenizer, args.caption_dropout_prob)

            loss_item = 0.0

            opt.zero_grad()

            # Number of bins, for loss recording
            n_loss_bins = 20
            # Create bins for t
            loss_bins = torch.linspace(0.0, 1.0, n_loss_bins + 1, device="cuda")
            # Initialize occurrence and sum tensors
            bin_occurrence = torch.zeros(n_loss_bins, device="cuda")
            bin_sum_loss = torch.zeros(n_loss_bins, device="cuda")

            for mb_idx in range((data_pack["local_bsz"] - 1) // data_pack["micro_bsz"] + 1):
                mb_st = mb_idx * data_pack["micro_bsz"]
                mb_ed = min((mb_idx + 1) * data_pack["micro_bsz"], data_pack["local_bsz"])
                last_mb = mb_ed == data_pack["local_bsz"]

                x_mb = x[mb_st:mb_ed]

                cap_feats_mb = cap_feats[mb_st:mb_ed]
                cap_mask_mb = cap_mask[mb_st:mb_ed]

                model_kwargs = dict(cap_feats=cap_feats_mb, cap_mask=cap_mask_mb)
                with {
                    "bf16": torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16),
                    "fp16": torch.amp.autocast(device_type="cuda", dtype=torch.float16),
                    "fp32": contextlib.nullcontext(),
                    "tf32": contextlib.nullcontext(),
                }[args.precision]:
                    loss_dict = data_pack["transport"].training_losses(model, x_mb, model_kwargs)
                loss = loss_dict["loss"].sum() / data_pack["local_bsz"]
                loss_item += loss.item()
                with model.no_sync() if args.data_parallel in ["sdp"] and not last_mb else contextlib.nullcontext():
                    loss.backward()

                # for bin-wise loss recording
                # Digitize t values to find which bin they belong to
                bin_indices = torch.bucketize(loss_dict["t"].cuda(), loss_bins, right=True) - 1
                detached_loss = loss_dict["loss"].detach()

                # Iterate through each bin index to update occurrence and sum
                for i in range(n_loss_bins):
                    mask = bin_indices == i  # Mask for elements in the i-th bin
                    bin_occurrence[i] = bin_occurrence[i] + mask.sum()  # Count occurrences in the i-th bin
                    bin_sum_loss[i] = bin_sum_loss[i] + detached_loss[mask].sum()  # Sum loss values in the i-th bin

            grad_norm = model.clip_grad_norm_(max_norm=args.grad_clip)

            dist.all_reduce(bin_occurrence)
            dist.all_reduce(bin_sum_loss)
            
            end_time = time()
            # Log metrics to wandb if enabled
            if rank == 0 and args.use_wandb and wandb_run is not None:
                log_dict = {
                    f"{train_res}/loss": loss_item,
                    f"{train_res}/grad_norm": grad_norm,
                    f"{train_res}/lr": opt.param_groups[0]["lr"],
                    f"{train_res}/imgs_per_sec": data_pack["global_bsz"] / (end_time - start_time),
                }
                
                # Add bin-wise loss metrics
                for i in range(n_loss_bins):
                    if bin_occurrence[i] > 0:
                        bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                        log_dict[f"{train_res}/loss-bin{i+1}-{n_loss_bins}"] = bin_avg_loss
                
                wandb.log(log_dict, step=step)

            opt.step()

            # Log loss values:
            metrics = data_pack["metrics"]
            metrics["loss"].update(loss_item)
            metrics["grad_norm"].update(grad_norm)
            metrics["Secs/Step"].update(end_time - start_time)
            metrics["Imgs/Sec"].update(data_pack["global_bsz"] / (end_time - start_time))
            metrics["grad_norm"].update(grad_norm)
            for i in range(n_loss_bins):
                if bin_occurrence[i] > 0:
                    bin_avg_loss = (bin_sum_loss[i] / bin_occurrence[i]).item()
                    metrics[f"bin_{i + 1:02}-{n_loss_bins}"].update(bin_avg_loss, int(bin_occurrence[i].item()))
            if (step + 1) % args.log_every == 0:
                # Measure training speed:
                torch.cuda.synchronize()
                logger.info(
                    f"Res{train_res}: (step{step + 1:07d}) "
                    + f"lr{opt.param_groups[0]['lr']:.6f} "
                    + " ".join([f"{key}:{str(metrics[key])}" for key in sorted(metrics.keys())])
                )

            start_time = time()

        update_ema(model_ema, model)

        # Save DiT checkpoint:
        if step == 0 or (step + 1) % args.ckpt_every == 0 or (step + 1) == args.max_steps:
            checkpoint_path = f"{checkpoint_dir}/{step + 1:07d}"
            os.makedirs(checkpoint_path, exist_ok=True)

            with FSDP.state_dict_type(
                model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_model_state_dict = model.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_fn = (
                        "consolidated."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_model_state_dict,
                        os.path.join(checkpoint_path, consolidated_fn),
                    )
            dist.barrier()
            del consolidated_model_state_dict
            logger.info(f"Saved consolidated to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model_ema,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            ):
                consolidated_ema_state_dict = model_ema.state_dict()
                if fs_init.get_data_parallel_rank() == 0:
                    consolidated_ema_fn = (
                        "consolidated_ema."
                        f"{fs_init.get_model_parallel_rank():02d}-of-"
                        f"{fs_init.get_model_parallel_world_size():02d}"
                        ".pth"
                    )
                    torch.save(
                        consolidated_ema_state_dict,
                        os.path.join(checkpoint_path, consolidated_ema_fn),
                    )
            dist.barrier()
            del consolidated_ema_state_dict
            logger.info(f"Saved consolidated_ema to {checkpoint_path}.")

            with FSDP.state_dict_type(
                model,
                StateDictType.LOCAL_STATE_DICT,
            ):
                opt_state_fn = f"optimizer.{dist.get_rank():05d}-of-" f"{dist.get_world_size():05d}.pth"
                torch.save(opt.state_dict(), os.path.join(checkpoint_path, opt_state_fn))
            dist.barrier()
            logger.info(f"Saved optimizer to {checkpoint_path}.")

            if dist.get_rank() == 0:
                torch.save(args, os.path.join(checkpoint_path, "model_args.pth"))
                with open(os.path.join(checkpoint_path, "resume_step.txt"), "w") as f:
                    print(step + 1, file=f)
            dist.barrier()
            logger.info(f"Saved training arguments to {checkpoint_path}.")

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...

    logger.info("Done!")

    # Finish wandb run if it was started
    if rank == 0 and args.use_wandb and wandb_run is not None:
        wandb.finish()

    cleanup()


if __name__ == "__main__":
    # Default args here will train DiT_Llama2_7B_patch2 with the
    # hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--cache_data_on_disk", default=False, action="store_true")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--model", type=str, default="DiT_Llama2_7B_patch2")
    parser.add_argument("--max_steps", type=int, default=100_000, help="Number of training steps.")
    parser.add_argument("--global_bsz", type=int, default=512)
    parser.add_argument("--micro_bsz", type=int, default=16)
    parser.add_argument("--global_seed", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--ckpt_every", type=int, default=50_000)
    parser.add_argument("--master_port", type=int, default=18181)
    parser.add_argument("--model_parallel_size", type=int, default=1)
    parser.add_argument("--data_parallel", type=str, choices=["sdp", "fsdp"], default="fsdp")
    parser.add_argument("--checkpointing", action="store_true")
    parser.add_argument("--precision", choices=["fp32", "tf32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--grad_precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--no_auto_resume",
        action="store_false",
        dest="auto_resume",
        help="Do NOT auto resume from the last checkpoint in --results_dir.",
    )
    parser.add_argument("--resume", type=str, help="Resume training from a checkpoint folder.")
    parser.add_argument(
        "--init_from",
        type=str,
        help="Initialize the model weights from a checkpoint folder. "
        "Compared to --resume, this loads neither the optimizer states "
        "nor the data loader states.",
    )
    parser.add_argument(
        "--grad_clip", type=float, default=2.0, help="Clip the L2 norm of the gradients to the given value."
    )
    parser.add_argument(
        "--wd",
        type=float,
        default=0.0,
        help="Weight decay for the optimizer.",
    )
    parser.add_argument(
        "--qk_norm",
        action="store_true",
    )
    parser.add_argument(
        "--caption_dropout_prob",
        type=float,
        default=0.1,
        help="Randomly change the caption of a sample to a blank string with the given probability.",
    )
    parser.add_argument("--snr_type", type=str, default="uniform")
    parser.add_argument(
        "--no_shift",
        action="store_true",
    )

    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--text_encoder", default="google/gemma-2-2b")

    # Weights & Biases arguments api key 98391050637a5b309639f110183d032837f535f7
    parser.add_argument("--use_wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb_project", type=str, default="speedruns", help="Weights & Biases project name")
    parser.add_argument("--wandb_entity", type=str, default="metauto", help="Weights & Biases entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Weights & Biases run name")
    parser.add_argument("--wandb_id", type=str, default=None, help="Weights & Biases run ID for resuming")
    
    args = parser.parse_args()

    main(args)
