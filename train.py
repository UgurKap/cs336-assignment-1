from cs336_basics import transformer_modules, training_utils
from pathlib import Path
import wandb
from dotenv import load_dotenv
import sys
import json
import torch
import numpy as np
import uuid
import shutil

load_dotenv()

models = {"standard-rope": transformer_modules.TransformerLM}
optimizers = {"adamw": training_utils.AdamW, "sgd": training_utils.SGD}
lr_schedules = {
    "lr_cosine_schedule": training_utils.lr_cosine_schedule,
    "lr_cosine_schedule_sine_warmup": training_utils.lr_cosine_schedule_sine_warmup,
}


def log_gradients(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            wandb.log(
                {
                    f"gradients/{name}.norm": param.grad.norm().item(),
                    f"gradients/{name}.mean": param.grad.mean().item(),
                },
                step=step,
            )


def train(cfg):
    wandb.login()
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    is_smoke_test = cfg.get("smoke_test", False)

    model = models[cfg["model"]["type"]](**cfg["model"]["params"])
    model.to(device)
    model = torch.compile(model)

    if "muon" not in (optimizer_name := cfg["optimizer"]["type"]):
        optimizer = optimizers[optimizer_name](model.parameters(), **cfg["optimizer"]["params"])

    gradient_clip_norm = cfg["optimizer"].get("gradient_clipping", None)
    lr_schedule_fn = (
        lr_schedules.get(cfg["optimizer"]["lr_schedule"]["type"], None) if "lr_schedule" in cfg["optimizer"] else None
    )

    train_data = np.load(cfg["data"]["train_path"], mmap_mode="r")
    val_data = np.load(cfg["data"]["val_path"], mmap_mode="r")

    context_length = cfg["model"]["params"]["context_length"]
    batch_size = cfg["training"]["batch_size"]
    iterations = cfg["training"]["iterations"]
    eval_steps = cfg["training"]["eval_steps"]
    grad_log_steps = cfg["training"]["grad_log_steps"]

    run_id = str(uuid.uuid4())
    checkpoint_freq = cfg["checkpoints"].get("freq", None)
    checkpoint_dir = Path(cfg["checkpoints"].get("save_dir", "models")) / run_id
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    start_iter = 0
    if load_from := cfg["checkpoints"].get("load_from", None):
        start_iter = training_utils.load_checkpoint(load_from, model, optimizer)
        print(f"Loaded checkpoint from {load_from}, starting at iteration {start_iter}")

    if len(sys.argv) > 1 and Path(sys.argv[-1]).exists():
        shutil.copy(sys.argv[-1], checkpoint_dir / "config.json")
        print(f"Copied config to {checkpoint_dir / 'config.json'}")

    precision = cfg.get("precision", "fp32")
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16 if precision == "fp16" else torch.float32

    cfg_with_uuid = {**cfg, "model_uuid": run_id}
    run = wandb.init(project="llm-from-scratch", config=cfg_with_uuid, mode="disabled" if is_smoke_test else "online")

    if is_smoke_test:
        smoke_inputs, smoke_targets = training_utils.get_batch(train_data, batch_size, context_length, device, seed=0)
        print("Smoke test")

    print(model)
    print(optimizer)
    print("\n========== Training Start ==============\n")

    for step in range(start_iter, iterations):
        model.train()

        if lr_schedule_fn:
            lr = lr_schedule_fn(step, **cfg["optimizer"]["lr_schedule"]["params"])
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

        if is_smoke_test:
            inputs, targets = smoke_inputs, smoke_targets
        else:
            inputs, targets = training_utils.get_batch(train_data, batch_size, context_length, device)

        with torch.autocast(device_type=device, dtype=dtype, enabled=(precision != "fp32")):
            logits = model(inputs)
            loss = training_utils.cross_entropy_loss(logits.view(-1, logits.size(-1)), targets.reshape(-1))

        loss.backward()

        if step % grad_log_steps == 0:
            log_gradients(model, step)

        if gradient_clip_norm:
            training_utils.gradient_clipping(model.parameters(), **gradient_clip_norm)

        optimizer.step()
        optimizer.zero_grad()

        wandb.log({"train/loss": loss.item(), "train/lr": optimizer.param_groups[0]["lr"]}, step=step)

        if (step % eval_steps == 0) or (step == iterations - 1):
            model.eval()
            with torch.no_grad():
                if is_smoke_test:
                    val_inputs, val_targets = smoke_inputs, smoke_targets
                else:
                    val_inputs, val_targets = training_utils.get_batch(val_data, batch_size, context_length, device)
                with torch.autocast(device_type=device, dtype=dtype, enabled=(precision != "fp32")):
                    val_logits = model(val_inputs)
                    val_loss = training_utils.cross_entropy_loss(
                        val_logits.view(-1, val_logits.size(-1)), val_targets.reshape(-1)
                    )
                wandb.log({"val/loss": val_loss.item()}, step=step)
                wandb.log({"val/perplexity": torch.exp(val_loss).item()}, step=step)
                print(
                    f"Step {step}: train_loss={loss.item():.4f}, val_loss={val_loss.item():.4f}, lr={optimizer.param_groups[0]['lr']:.6f}"
                )

        if checkpoint_freq and step % checkpoint_freq == 0 and step > 0:
            checkpoint_path = checkpoint_dir / f"{cfg['model']['type']}_checkpoint_step_{step}.pt"
            training_utils.save_checkpoint(model, optimizer, step, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    final_checkpoint = checkpoint_dir / f"{cfg['model']['type']}_checkpoint_final.pt"
    training_utils.save_checkpoint(model, optimizer, iterations, final_checkpoint)
    print(f"Training complete. Saved final checkpoint to {final_checkpoint}")

    run.finish()


if __name__ == "__main__":
    with open(sys.argv[-1]) as f:
        cfg = json.load(f)
    train(cfg)
