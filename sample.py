import argparse
import json
import math
import os
import random
import socket
import time

from diffusers.models import AutoencoderKL
import numpy as np
import torch
import torch.distributed as dist
from torchvision.transforms.functional import to_pil_image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import models
from transport import Sampler, create_transport

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
            input_ids=text_input_ids.cuda(),
            attention_mask=prompt_masks.cuda(),
            output_hidden_states=True,
        ).hidden_states[-2]

    return prompt_embeds, prompt_masks

def parse_transport_args(parser):
    group = parser.add_argument_group("Transport arguments")
    group.add_argument(
        "--path-type",
        type=str,
        default="Linear",
        choices=["Linear", "GVP", "VP"],
        help="the type of path for transport: 'Linear', 'GVP' (Geodesic Vector Pursuit), or 'VP' (Vector Pursuit).",
    )
    group.add_argument(
        "--prediction",
        type=str,
        default="velocity",
        choices=["velocity", "score", "noise"],
        help="the prediction model for the transport dynamics.",
    )
    group.add_argument(
        "--loss-weight",
        type=none_or_str,
        default=None,
        choices=[None, "velocity", "likelihood"],
        help="the weighting of different components in the loss function, can be 'velocity' for dynamic modeling, 'likelihood' for statistical consistency, or None for no weighting.",
    )
    group.add_argument("--sample-eps", type=float, help="sampling in the transport model.")
    group.add_argument("--train-eps", type=float, help="training to stabilize the learning process.")


def parse_ode_args(parser):
    group = parser.add_argument_group("ODE arguments")
    group.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for the ODE solver.",
    )
    group.add_argument(
        "--rtol",
        type=float,
        default=1e-3,
        help="Relative tolerance for the ODE solver.",
    )
    group.add_argument("--reverse", action="store_true", help="run the ODE solver in reverse.")
    group.add_argument(
        "--likelihood",
        action="store_true",
        help="Enable calculation of likelihood during the ODE solving process.",
    )



def none_or_str(value):
    if value == "None":
        return None
    return value


def main(args, rank, master_port):
    # Setup PyTorch:
    torch.set_grad_enabled(False)

    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(args.num_gpus)
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    dist.init_process_group("nccl")
    torch.cuda.set_device(rank)
    device = rank % torch.cuda.device_count()
    device_str = f"cuda:{device}"
    
    train_args = torch.load(os.path.join(args.ckpt, "model_args.pth"))
    if dist.get_rank() == 0:
        print("Loaded model arguments:", json.dumps(train_args.__dict__, indent=2))

    if dist.get_rank() == 0:
        print(f"Creating text encoder: {args.text_encoder}")

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b", token=args.hf_token)
    tokenizer.padding_side = "right"

    text_encoder = AutoModel.from_pretrained(
        "google/gemma-2-2b", torch_dtype=dtype, device_map="cuda", token=args.hf_token
    ).eval()
    cap_feat_dim = text_encoder.config.hidden_size
    
    if args.vae == "flux":
        vae = AutoencoderKL.from_pretrained(f"black-forest-labs/FLUX.1-dev", subfolder="vae", torch_dtype=dtype).to(device_str)
    elif args.vae != "sdxl":
        vae = AutoencoderKL.from_pretrained(
            f"stabilityai/sd-vae-ft-{args.vae}"
            if args.local_diffusers_model_root is None
            else os.path.join(args.local_diffusers_model_root, f"stabilityai/sd-vae-ft-{args.vae}")
        ).to(device)
    else:
        vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae").to(device)
    vae.requires_grad_(False)

    if dist.get_rank() == 0:
        print(f"Creating DiT: {train_args.model}")
    model = models.__dict__[train_args.model](
        in_channels=16,
        qk_norm=train_args.qk_norm,
        cap_feat_dim=cap_feat_dim,
    )
    model.eval().to("cuda", dtype=dtype)
    
    if args.debug == False:
        # assert train_args.model_parallel_size == args.num_gpus
        if args.ema:
            print("Loading ema model.")
        ckpt = torch.load(
            os.path.join(
                args.ckpt,
                f"consolidated.{rank:02d}-of-{args.num_gpus:02d}.pth",
            )
        )
        model.load_state_dict(ckpt, strict=True)
    
    # begin sampler``
    if args.solver == "dpm":
        transport = create_transport(
            "Linear",
            "velocity",
        )
        sampler = Sampler(transport)
    else:
        transport = create_transport(
            args.path_type,
            args.prediction,
            args.loss_weight,
            args.train_eps,
            args.sample_eps,
        )
        sampler = Sampler(transport)

    sample_folder_dir = args.image_save_path

    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        os.makedirs(os.path.join(sample_folder_dir, "images"), exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    info_path = os.path.join(args.image_save_path, "data.json")
    if os.path.exists(info_path):
        with open(info_path, "r") as f:
            info = json.loads(f.read())
        collected_id = []
    else:
        info = []
        collected_id = []

    if args.caption_path.endswith("json"):
        with open(args.caption_path, "r", encoding="utf-8") as file:
            data = json.load(file)
    elif args.caption_path.endswith("jsonl"):
        data = []
        with open(args.caption_path, "r", encoding="utf-8") as file:
            for line in file:
                data.append(json.loads(line))
    else:
        data = []
        with open(args.caption_path, "r", encoding="utf-8") as file:
            for line in file:
                text = line.strip()
                if text:
                    data.append(line.strip())
    neg_cap = ""
    total = len(info)
    resolution = args.resolution

    with torch.autocast("cuda", dtype):
        for res in resolution:
            for idx, item in tqdm(enumerate(data)):

                if int(args.seed) != 0:
                    torch.random.manual_seed(int(args.seed))

                sample_id = f'{idx}_{res.split(":")[-1]}'
                if isinstance(item, str):
                    cap = item
                else:
                    if "refined_prompt" in item:
                        cap = item["refined_prompt"]
                        print("refined_prompt")
                    elif "prompt" in item:
                        cap = item["prompt"]
                        print("prompt")
                    else:
                        cap = item["gpt_4_caption"]

                if args.system_type == "align":
                    system_prompt = "You are an assistant designed to generate high-quality images with the highest degree of image-text alignment based on textual prompts. <Prompt Start> "  # noqa
                elif args.system_type == "base":
                    system_prompt = "You are an assistant designed to generate high-quality images based on user prompts. <Prompt Start> "  # noqa
                elif args.system_type == "aesthetics":
                    system_prompt = "You are an assistant designed to generate high-quality images with highest degree of aesthetics based on user prompts. <Prompt Start> "  # noqa
                elif args.system_type == "real":
                    system_prompt = "You are an assistant designed to generate superior images with the superior degree of image-text alignment based on textual prompts or user prompts. <Prompt Start> "  # noqa
                elif args.system_type == "4grid":
                    system_prompt = "You are an assistant designed to generate four high-quality images with highest degree of aesthetics arranged in 2x2 grids based on user prompts. <Prompt Start> "  # noqa
                elif args.system_type == "empty":
                    system_prompt = ""
                else:
                    raise ValueError

                cap = system_prompt + cap
                if neg_cap != "":
                    neg_cap = system_prompt + neg_cap

                caps_list = [cap]

                res_cat, resolution = res.split(":")
                res_cat = int(res_cat)
                
                n = len(caps_list)
                w, h = resolution.split("x")
                w, h = int(w), int(h)
                latent_w, latent_h = w // 8, h // 8
                z = torch.randn([1, 16, latent_w, latent_h], device="cuda").to(dtype)
                z = z.repeat(n * 2, 1, 1, 1)

                with torch.no_grad():
                    cap_feats, cap_mask = encode_prompt([cap] + [neg_cap], text_encoder, tokenizer, 0.0)
        
                cap_mask = cap_mask.to(cap_feats.device)
                model_kwargs = dict(
                    cap_feats=cap_feats,
                    cap_mask=cap_mask,
                    cfg_scale=args.cfg_scale,
                )

                start_time = time.perf_counter()
                if args.solver == "dpm":
                    sample_fn = sampler.sample_dpm(
                        model.forward_with_cfg,
                        model_kwargs=model_kwargs,
                    )
                    samples = sample_fn(z, steps=args.num_sampling_steps, order=2, skip_type="time_uniform_flow", method="multistep", flow_shift=args.time_shifting_factor)
                else:
                    sample_fn = sampler.sample_ode(
                        sampling_method=args.solver,
                        num_steps=args.num_sampling_steps,
                        atol=args.atol,
                        rtol=args.rtol,
                        reverse=args.reverse,
                        time_shifting_factor=args.t_shift
                    )
                    samples = sample_fn(z, model.forward_with_cfg, **model_kwargs)[-1]
                end_time = time.perf_counter()
                samples = samples[:1]
                samples = vae.decode(samples / vae.config.scaling_factor + vae.config.shift_factor)[0]
                samples = (samples + 1.0) / 2.0
                samples.clamp_(0.0, 1.0)
                print("sample times:", end_time-start_time)

                # Save samples to disk as individual .png files
                for i, (sample, cap) in enumerate(zip(samples, caps_list)):
                    img = to_pil_image(sample.float())
                    save_path = f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png"
                    img.save(save_path)
                    info.append(
                        {
                            "caption": cap,
                            "image_url": f"{args.image_save_path}/images/{args.solver}_{args.num_sampling_steps}_{sample_id}.png",
                            "resolution": f"res: {resolution}\ntime_shift: {args.time_shifting_factor}",
                            "solver": args.solver,
                            "num_sampling_steps": args.num_sampling_steps,
                        }
                    )

                with open(info_path, "w") as f:
                    f.write(json.dumps(info))

                total += len(samples)
                dist.barrier()
    # end_time = time.time()
    print("sample times:", end_time-start_time)
    dist.barrier()
    dist.barrier()
    dist.destroy_process_group()


def find_free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--solver", type=str, default="euler")
    parser.add_argument("--t_shift", type=int, default=6)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
    )
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--ema", action="store_true", help="Use EMA models.")
    parser.set_defaults(ema=True)
    parser.add_argument(
        "--image_save_path",
        type=str,
        default="samples",
        help="If specified, overrides the default image save path "
        "(sample{_ema}.png in the model checkpoint directory).",
    )
    parser.add_argument(
        "--time_shifting_factor",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--caption_path",
        type=str,
        default="prompts.txt",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="",
        nargs="+",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="",
    )
    parser.add_argument("--proportional_attn", type=bool, default=True)
    parser.add_argument(
        "--scaling_method",
        type=str,
        default="None",
        choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--system_type",
        type=str,
        default="real",
        # choices=["Time-aware", "None"],
    )
    parser.add_argument(
        "--scaling_watershed",
        type=float,
        default=0.3,
    )
    parser.add_argument(
        "--vae", type=str, choices=["ema", "mse", "sdxl", "flux"], default="flux"
    )
    parser.add_argument(
        "--text_encoder", type=str, nargs='+', default=['gemma'], help="List of text encoders to use (e.g., t5, clip, gemma)"
    )
    parser.add_argument(
        "--max_length", type=int, default=256, help="Max length for text encoder."
    )
    parser.add_argument(
        "--use_parallel_attn",
        type=bool,
        default=False,
        help="Use parallel attention in the model.",
    )
    parser.add_argument(
        "--use_flash_attn",
        type=bool,
        default=True,
        help="Use Flash Attention in the model.",
    )
    parser.add_argument("--do_shift", default=False)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--hf_token", type=str, default=None, help="huggingface read token for accessing gated repo.")

    parse_transport_args(parser)
    parse_ode_args(parser)

    args = parser.parse_known_args()[0]

    master_port = find_free_port()
    assert args.num_gpus == 1, "Multi-GPU sampling is currently not supported."

    main(args, 0, master_port)
