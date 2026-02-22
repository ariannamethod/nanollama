"""
Common utilities for nanollama.
Adapted from nanochat with modifications for Llama 3 architecture.
"""

import os
import re
import logging
import urllib.request
import torch
import torch.distributed as dist
from filelock import FileLock

class ColoredFormatter(logging.Formatter):
    """Custom formatter that adds colors to log messages."""
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'

    def format(self, record):
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        message = super().format(record)
        if levelname == 'INFO':
            message = re.sub(r'(\d+\.?\d*\s*(?:GB|MB|%|docs))', rf'{self.BOLD}\1{self.RESET}', message)
            message = re.sub(r'(Shard \d+)', rf'{self.COLORS["INFO"]}{self.BOLD}\1{self.RESET}', message)
        return message


def setup_default_logging():
    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logging.basicConfig(
        level=logging.INFO,
        handlers=[handler]
    )


setup_default_logging()
logger = logging.getLogger(__name__)


def get_base_dir():
    """Get the base directory for nanollama data and checkpoints."""
    if os.environ.get("NANOLLAMA_BASE_DIR"):
        nanollama_dir = os.environ.get("NANOLLAMA_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanollama_dir = os.path.join(cache_dir, "nanollama")
    os.makedirs(nanollama_dir, exist_ok=True)
    return nanollama_dir


def download_file_with_lock(url, filename, postprocess_fn=None):
    """
    Downloads a file from a URL to a local path in the base directory.
    Uses a lock file to prevent concurrent downloads among multiple ranks.
    """
    base_dir = get_base_dir()
    file_path = os.path.join(base_dir, filename)
    lock_path = file_path + ".lock"

    if os.path.exists(file_path):
        return file_path

    with FileLock(lock_path):
        if os.path.exists(file_path):
            return file_path

        print(f"Downloading {url}...")
        with urllib.request.urlopen(url) as response:
            content = response.read()

        with open(file_path, 'wb') as f:
            f.write(content)
        print(f"Downloaded to {file_path}")

        if postprocess_fn is not None:
            postprocess_fn(file_path)

    return file_path


def print0(s="", **kwargs):
    """Print only on rank 0."""
    ddp_rank = int(os.environ.get('RANK', 0))
    if ddp_rank == 0:
        print(s, **kwargs)


def print_banner():
    """Print ASCII art banner."""
    banner = """
                                                       ████  ████
                                                      ░░███ ░░███
     ████████    ██████   ████████    ██████   █████   ░███  ░███   ██████   █████████████    ██████
    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███  ░███  ░░░░░███ ░░███░░███░░███  ░░░░░███
     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███  ░███   ███████  ░███ ░███ ░███   ███████
     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███  ░███  ███░░███  ░███ ░███ ░███  ███░░███
     ████ █████░░████████ ████ █████░░██████ ░░██████  █████ █████░░████████ █████░███ █████░░████████
    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░░ ░░░░░  ░░░░░░░░ ░░░░░ ░░░ ░░░░░  ░░░░░░░░

                        by Arianna Method • Llama 3 from scratch • Fork of nanochat
    """
    print0(banner)


def is_ddp_requested() -> bool:
    """True if launched by torchrun (env present), even before init."""
    return all(k in os.environ for k in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))


def is_ddp_initialized() -> bool:
    """True if torch.distributed is available and the process group is initialized."""
    return dist.is_available() and dist.is_initialized()


def get_dist_info():
    """Get distributed training info."""
    if is_ddp_requested():
        assert all(var in os.environ for var in ['RANK', 'LOCAL_RANK', 'WORLD_SIZE'])
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        return True, ddp_rank, ddp_local_rank, ddp_world_size
    else:
        return False, 0, 0, 1


def autodetect_device_type():
    """Detect available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device_type = "cuda"
    elif torch.backends.mps.is_available():
        device_type = "mps"
    else:
        device_type = "cpu"
    print0(f"Autodetected device type: {device_type}")
    return device_type


def compute_init(device_type="cuda"):
    """Initialize compute environment (device, DDP, precision)."""
    assert device_type in ["cuda", "mps", "cpu"], "Invalid device type"
    if device_type == "cuda":
        assert torch.cuda.is_available(), "CUDA not available"
    if device_type == "mps":
        assert torch.backends.mps.is_available(), "MPS not available"

    # Reproducibility
    torch.manual_seed(42)
    if device_type == "cuda":
        torch.cuda.manual_seed(42)

    # Precision
    if device_type == "cuda":
        torch.set_float32_matmul_precision("high")

    # Distributed setup
    is_ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    if is_ddp and device_type == "cuda":
        device = torch.device("cuda", ddp_local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    else:
        device = torch.device(device_type)

    if ddp_rank == 0:
        logger.info(f"Distributed world size: {ddp_world_size}")

    return is_ddp, ddp_rank, ddp_local_rank, ddp_world_size, device


def compute_cleanup():
    """Clean up distributed training."""
    if is_ddp_initialized():
        dist.destroy_process_group()


class DummyWandb:
    """Dummy wandb for when we don't want to use it."""
    def __init__(self):
        pass
    def log(self, *args, **kwargs):
        pass
    def finish(self):
        pass


def get_peak_flops(device_name: str) -> float:
    """Get peak BF16 FLOPs for various GPUs."""
    name = device_name.lower()

    _PEAK_FLOPS_TABLE = (
        # NVIDIA Blackwell
        (["gb200"], 2.5e15),
        (["grace blackwell"], 2.5e15),
        (["b200"], 2.25e15),
        (["b100"], 1.8e15),
        # NVIDIA Hopper
        (["h200", "nvl"], 836e12),
        (["h200", "pcie"], 836e12),
        (["h200"], 989e12),
        (["h100", "nvl"], 835e12),
        (["h100", "pcie"], 756e12),
        (["h100"], 989e12),
        (["h800", "nvl"], 989e12),
        (["h800"], 756e12),
        # NVIDIA Ampere
        (["a100"], 312e12),
        (["a800"], 312e12),
        (["a40"], 149.7e12),
        (["a30"], 165e12),
        # NVIDIA Ada
        (["l40s"], 362e12),
        (["l40-s"], 362e12),
        (["l40 s"], 362e12),
        (["l4"], 121e12),
        # AMD
        (["mi355"], 2.5e15),
        (["mi325"], 1.3074e15),
        (["mi300x"], 1.3074e15),
        (["mi300a"], 980.6e12),
        (["mi250x"], 383e12),
        (["mi250"], 362.1e12),
        # Consumer
        (["5090"], 209.5e12),
        (["4090"], 165.2e12),
        (["3090"], 71e12),
    )

    for patterns, flops in _PEAK_FLOPS_TABLE:
        if all(p in name for p in patterns):
            return flops

    logger.warning(f"Peak flops undefined for: {device_name}, MFU will show as 0%")
    return float('inf')
