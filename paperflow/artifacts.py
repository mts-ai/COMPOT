"""Check saved artifacts before reusing a completed pipeline stage."""

import json
import hashlib
from pathlib import Path
from safetensors import SafetensorError, safe_open


def model_identifier(name):
    """Keep local-model output names short while distinguishing equal basenames."""
    path = Path(str(name))
    if path.is_absolute() or path.is_dir():
        suffix = hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:8]
        return path.name[:64] + '_' + suffix
    return str(name).replace('/', '_')


def checkpoint_complete(root):
    root = Path(root)
    try:
        config = json.loads((root / "config.json").read_text())
        indices = [root / name for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json")]
        index = next((p for p in indices if p.is_file()), None)
        if index:
            files = set(json.loads(index.read_text())["weight_map"].values())
        else:
            files = {p.name for pattern in ("model*.safetensors", "pytorch_model*.bin") for p in root.glob(pattern)}
        if not files:
            return False
        for name in files:
            path = root / name
            if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size == 0:
                return False
            if path.suffix == '.safetensors':
                with safe_open(path, framework='pt', device='cpu') as tensors:
                    if not list(tensors.keys()):
                        return False
        if config.get("compot_checkpoint_format"):
            manifest = json.loads((root / "compression_manifest.json").read_text())
            if manifest["achieved_core_cr"] + 1e-12 < manifest["target_core_cr"]:
                return False
        for target in config.get("auto_map", {}).values():
            targets = target if isinstance(target, list) else [target]
            for entry in targets:
                if entry and "--" not in entry and not (root / (entry.rsplit(".", 1)[0] + ".py")).is_file():
                    return False
        return True
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, SafetensorError):
        return False


def compression_signature(settings):
    """Numerical inputs only; evaluation settings do not invalidate compression."""
    defaults = dict(allocation="uniform", allocation_min_cr=.05, allocation_max_cr=.8,
                    allocation_norm="fro", checkpoint_format="deployable_v1", layer_idx="")
    keys = ("model_name", "method", "compression_ratio", "ks_ratio",
            "ksvd_iterations", "power_iterations")
    result = {key: settings[key] for key in keys}
    result.update({key: settings.get(key, value) for key, value in defaults.items()})
    result["calib_data"] = str(Path(settings["calib_data"]).resolve())
    result["seed"] = 42
    return result
