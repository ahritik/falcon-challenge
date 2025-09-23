"""
Entry point to train/evaluate the Hritik H2 decoder.

Examples (PowerShell):
  # Train on held-in calib and save checkpoint
  python decoder_demos/hritik_sample.py train --training_dir data/h2/held_in_calib --save-path local_data/hritik_h2

  # Local minival evaluation (must have model arrays saved)
  python decoder_demos/hritik_sample.py eval --model-path local_data/hritik_h2/model --phase minival --split h2
"""
from __future__ import annotations

import argparse
from pathlib import Path

from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

def cmd_train(args):
    # Set JAX platform before importing JAX-dependent modules
    import os
    if getattr(args, "device", None):
        os.environ["JAX_PLATFORMS"] = args.device
        # Optional: avoid large upfront allocations
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    from decoder_demos.hritik_decoder import train_hritik, HritikConfig, TrainConfig
    import jax
    print(f"JAX devices: {jax.devices()} | backend: {jax.default_backend()}")
    save_root = Path(args.save_path)
    save_root.mkdir(parents=True, exist_ok=True)
    save_base = save_root / "model"
    cfg = HritikConfig()
    tcfg = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr)
    train_hritik(Path(args.training_dir), Path(args.calibration_dir) if args.calibration_dir else None, save_base, cfg, tcfg)


def cmd_eval(args):
    import os
    if getattr(args, "device", None):
        os.environ["JAX_PLATFORMS"] = args.device
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    task = FalconTask.h2
    cfg = FalconConfig(task=task)
    from decoder_demos.hritik_decoder import HritikDecoder
    import jax
    print(f"JAX devices: {jax.devices()} | backend: {jax.default_backend()}")
    dec = HritikDecoder(task_config=cfg, model_path=args.model_path, batch_size=1)
    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split="h2")
    res = evaluator.evaluate(decoder=dec, phase=args.phase)
    print(res)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--training_dir", type=str, required=True)
    pt.add_argument("--calibration_dir", type=str, default=None)
    pt.add_argument("--save-path", type=str, default="local_data/hritik_h2")
    pt.add_argument("--epochs", type=int, default=3)
    pt.add_argument("--batch-size", type=int, default=8)
    pt.add_argument("--lr", type=float, default=1e-3)
    pt.add_argument("--device", choices=["cpu", "cuda"], default=None)

    pe = sub.add_parser("eval")
    pe.add_argument("--evaluation", choices=["local", "remote"], default="local")
    pe.add_argument("--model-path", type=str, required=True, help="Base path of saved model (without suffix)")
    pe.add_argument("--phase", choices=["minival", "test"], default="minival")
    pe.add_argument("--split", choices=["h2"], default="h2")
    pe.add_argument("--device", choices=["cpu", "cuda"], default=None)

    args = p.parse_args()
    if args.cmd == "train":
        cmd_train(args)
    else:
        cmd_eval(args)


if __name__ == "__main__":
    main()
