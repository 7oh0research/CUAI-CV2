"""Environment and GPU-state capture so every record can be audited (versions, clocks, throttling)."""

from __future__ import annotations

import importlib.metadata as md
import subprocess

import torch


def _ver(pkg: str):
    try:
        return md.version(pkg)
    except md.PackageNotFoundError:
        return None


def _git_commit() -> str | None:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or None
    except Exception:
        return None


def collect_env(device: torch.device) -> dict:
    cuda = device.type == "cuda"
    return {"device": torch.cuda.get_device_name(device) if cuda else "cpu",
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "transformers_version": _ver("transformers"), "flash_attn_version": _ver("flash-attn"),
            "git_commit": _git_commit()}


def gpu_state() -> dict:
    """SM clock / temperature / power right now (nvidia-smi). Empty on machines without it."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        sm, temp, power = [float(x) for x in out.strip().splitlines()[0].split(",")]
        return {"sm_clock_mhz": sm, "temp_c": temp, "power_w": power}
    except Exception:
        return {}
