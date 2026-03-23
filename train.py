from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import typing

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import wandb
import yaml

from llm_core.model import TransformerLM
from llm_core.optimizer import AdamW, cosine_lr_schedule
from llm_core.nn_functional import cross_entropy, clip_gradient
from llm_core.dataloader import sample_batch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    save_optimizer: bool = True,
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "iteration": iteration,
    }
    if save_optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True)
    torch.save(checkpoint, path)
    with open(os.path.join(parent_dir, "model_config.json"), "w") as f:
        json.dump(model.config, f)


def load_checkpoint(
    path: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    if "optimizer" in checkpoint and optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def get_gpu_stats():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
    )
    lines = result.stdout.decode("utf-8").strip().split("\n")
    return [line.split(", ") for line in lines]


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_worker(
    rank: int = 0,
    world_size: int = 1,
    use_fsdp: bool = False,
    config_path: str = "./configures/sample.yaml",
    use_amp: bool = False,
    use_compile: bool = False,
    use_flash_attn: bool = False,
    use_custom_triton: bool = False,
):
    torch.set_float32_matmul_precision("high")
    config = load_config(config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]
    optim_cfg = config["optimizer"]
    dataset_cfg = config["dataset"]

    use_wandb = train_cfg["wandb"]
    batch_size = train_cfg["batch_size"]
    max_steps = train_cfg["max_steps"]
    val_interval = train_cfg["val_interval"]
    gpu_log_interval = train_cfg["gpu_log_interval"]
    checkpoint_dir = train_cfg["checkpoint_dir"]

    context_length = model_cfg["context_length"]

    lr_max = float(optim_cfg["learning_rate_max"])
    lr_min = float(optim_cfg["learning_rate_min"])
    max_grad_norm = optim_cfg["max_grad_norm"]

    if use_fsdp:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        device = f"cuda:{local_rank}"
        # Only rank 0 logs to wandb
        if rank != 0:
            use_wandb = False
    else:
        device = detect_device() if train_cfg["device"] == "auto" else train_cfg["device"]
    amp_device_type = device.split(":")[0]

    if use_wandb:
        run_name = f"bs{batch_size}_lr{lr_max}_layer{model_cfg['num_layers']}"
        if use_fsdp:
            run_name += f"_fsdp_ws{world_size}"
        wandb.init(
            project=train_cfg["wandb_project"],
            name=run_name,
            config={
                "batch_size": batch_size,
                "learning_rate": lr_max,
                "context_length": context_length,
                "d_model": model_cfg["d_model"],
                "max_steps": max_steps,
                "use_fsdp": use_fsdp,
                "world_size": world_size,
            },
        )

    print(f"[Rank {rank}] device: {device}")

    train_data = np.memmap(dataset_cfg["train_path"], dtype=np.uint32, mode="r")
    val_data = np.memmap(dataset_cfg["val_path"], dtype=np.uint32, mode="r")

    model = TransformerLM(
        d_model=model_cfg["d_model"],
        num_heads=model_cfg["num_heads"],
        d_ff=model_cfg["d_ff"],
        vocab_size=model_cfg["vocab_size"],
        context_length=context_length,
        num_layers=model_cfg["num_layers"],
        rope_theta=model_cfg["rope_theta"],
        use_flash_attn=use_flash_attn,
        use_custom_triton=use_custom_triton,
    ).to(device)

    inner_model = model  # keep reference to the unwrapped TransformerLM for checkpointing

    if use_compile:
        model = torch.compile(model)

    if use_fsdp:
        from llm_systems.parallelism.fsdp_zero3 import FSDP
        model = FSDP(model)

    model.train()
    if use_custom_triton:
        from llm_systems.kernels.triton_adamw import FusedAdamW
        from llm_systems.kernels.triton_cross_entropy import triton_cross_entropy
        OptimizerClass = FusedAdamW
        cross_entropy_fn = triton_cross_entropy
    else:
        OptimizerClass = AdamW
        cross_entropy_fn = cross_entropy
    optimizer = OptimizerClass(
        params=model.parameters(),
        lr=lr_max,
        weight_decay=float(optim_cfg["weight_decay"]),
    )

    # Under FSDP each rank processes a slice of the global batch.
    local_batch_size = batch_size // world_size if use_fsdp else batch_size

    for step in range(max_steps):
        if device == "cuda":
            torch.cuda.synchronize()
        step_start = time.time()

        lr = cosine_lr_schedule(
            step=step,
            max_learning_rate=lr_max,
            min_learning_rate=lr_min,
            warmup_steps=optim_cfg["warmup_steps"],
            decay_steps=optim_cfg["decay_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = sample_batch(
            tokens=train_data,
            batch_size=local_batch_size,
            context_length=context_length,
            device=device,
        )

        # Use model.zero_grad() (not optimizer.zero_grad()) so FSDP's custom
        # zero_grad also releases the unsharded grad references held on each
        # unit.module's parameters — otherwise those stale tensors leak memory.
        model.zero_grad()
        with torch.autocast(device_type=amp_device_type, dtype=torch.bfloat16, enabled=use_amp):
            logits = model(x)
            loss = cross_entropy_fn(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        if use_fsdp:
            model.finish_gradient_synchronization()
        clip_gradient(model.parameters(), max_grad_norm)
        optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
        step_time = time.time() - step_start
        tokens_per_step = batch_size * context_length
        tokens_per_sec = tokens_per_step / step_time
        print(f"[Rank {rank}] Step {step}: loss = {loss.item():.4f}, lr = {lr:.6f}, time = {step_time:.4f}s, tok/s = {tokens_per_sec:.0f}")

        if use_wandb:
            wandb.log({"train/loss": loss.item(), "train/lr": lr, "step": step})

        if step % val_interval == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                x_val, y_val = sample_batch(
                    tokens=val_data,
                    batch_size=local_batch_size,
                    context_length=context_length,
                    device=device,
                )
                val_logits = model(x_val)
                val_loss = cross_entropy_fn(val_logits[:, -1, :], y_val[:, -1])

                if use_wandb:
                    wandb.log({"val/loss": val_loss.item(), "step": step})
                print(f"[Rank {rank}] [Validation] Step {step}: val_loss = {val_loss.item():.4f}")
            model.train()

        if step % val_interval == 0 and step > 0 and not use_fsdp:
            save_checkpoint(
                model=inner_model,
                optimizer=optimizer,
                iteration=step,
                path=os.path.join(checkpoint_dir, f"ckpt_{step}.pt"),
            )

        if use_wandb and step % gpu_log_interval == 0:
            allocated_memory = int(torch.cuda.memory_allocated() / 1024**2)
            cached_memory = int(torch.cuda.memory_reserved() / 1024**2)
            wandb.log({"allocated_memory": allocated_memory, "cached_memory": cached_memory})
            gpu_stats = get_gpu_stats()
            for idx, stat in enumerate(gpu_stats):
                utilization, memory_used, temperature = map(float, stat)
                wandb.log({
                    f"gpu_{idx}_utilization": utilization,
                    f"gpu_{idx}_memory_used": memory_used,
                    f"gpu_{idx}_temperature": temperature,
                })

    if use_fsdp:
        # Gather full params onto every rank so rank 0 can write a complete state dict.
        for unit in model.fsdp_units:
            unit.all_gather_params(async_op=False)
        if rank == 0:
            save_checkpoint(
                model=inner_model,
                optimizer=optimizer,
                iteration=max_steps,
                path=os.path.join(checkpoint_dir, "ckpt_final.pt"),
                save_optimizer=False,
            )
        for unit in model.fsdp_units:
            unit.discard_full_params()
        dist.destroy_process_group()
    else:
        save_checkpoint(
            model=inner_model,
            optimizer=optimizer,
            iteration=max_steps,
            path=os.path.join(checkpoint_dir, "ckpt_final.pt"),
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configures/sample.yaml", help="Path to training config YAML")
    parser.add_argument("--amp", action="store_true", help="Enable mixed-precision training with BF16")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile for faster training")
    parser.add_argument("--flash-attn", action="store_true", help="Use PyTorch's scaled_dot_product_attention (FlashAttention2)")
    parser.add_argument("--custom-triton", action="store_true", help="Use custom Triton kernels in place of the Pytorch implementations")
    parser.add_argument("--fsdp", action="store_true", help="Enable FSDP ZeRO-3 distributed training")
    parser.add_argument("--world-size", type=int, default=2, help="Number of GPUs for FSDP (default: 2)")
    args = parser.parse_args()

    if args.fsdp:
        mp.set_start_method("spawn", force=True)
        mp.spawn(
            train_worker,
            args=(args.world_size, True, args.config, args.amp, args.compile, args.flash_attn, args.custom_triton),
            nprocs=args.world_size,
            join=True,
        )
    else:
        train_worker(
            rank=0,
            world_size=1,
            use_fsdp=False,
            config_path=args.config,
            use_amp=args.amp,
            use_compile=args.compile,
            use_flash_attn=args.flash_attn,
            use_custom_triton=args.custom_triton,
        )


if __name__ == "__main__":
    main()
