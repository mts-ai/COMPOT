"""Run COMPOT, CoSpaDi and SVD-LLM compression workers."""

import sys
from pathlib import Path
from .utils import run_checked
from .artifacts import checkpoint_complete, compression_signature, model_identifier


def _extras(args):
    result = []
    for key, value in args.items():
        flag = "--" + key
        if isinstance(value, bool):
            if value:
                result.append(flag)
        elif value is not None:
            if isinstance(value, list):
                result.extend([flag, ",".join(map(str, value))])
            else:
                result.extend([flag, str(value)])
    return result


class CoSpaDiBackend:
    """Execute the selected compression method in a separate worker process."""

    def build_command(self, ctx):
        cfg, method = ctx["cfg"], ctx["method"]
        if method.name not in ("ksvd", "compot", "svdllm"):
            raise ValueError("Compression backend only supports ksvd, compot or svdllm.")
        return [
            sys.executable,
            "-m",
            "paperflow.impl.compress",
            "--method",
            method.name,
            "--compression-ratio",
            str(cfg.compression_ratio),
            "--model-name",
            str(cfg.model.name),
            "--calib-data",
            str(ctx["calib_root"]),
            "--save-path",
            str(ctx["checkpoint_root"]),
            "--ks-ratio",
            str(method.ks_ratio),
            "--ksvd-iterations",
            str(method.iterations),
            "--power-iterations",
            str(method.power_iterations),
        ] + _extras(method.extra_args)

    def checkpoint_path(self, ctx):
        """Return the checkpoint path determined by the compression settings."""
        cfg, method = ctx["cfg"], ctx["method"]
        model_id = model_identifier(cfg.model.name)
        folder = (
            f"{model_id}_cr_{cfg.compression_ratio}_{method.name}_"
            f"ks_{method.ks_ratio}_ksvd_it_{method.iterations}_"
            f"power_it_{method.power_iterations}"
        )
        folder += "_alloc_" + str(method.extra_args.get("allocation", "uniform"))
        checkpoint_format = str(method.extra_args.get("checkpoint-format", "deployable_v1"))
        folder += "_format_" + checkpoint_format
        if checkpoint_format == "deployable_v1":
            return Path(ctx["checkpoint_root"]) / folder / "transformers"
        return Path(ctx["checkpoint_root"]) / folder

    def signature(self, ctx):
        cfg, method = ctx["cfg"], ctx["method"]
        settings = dict(model_name=str(cfg.model.name), method=method.name,
                        compression_ratio=float(cfg.compression_ratio),
                        ks_ratio=method.ks_ratio, ksvd_iterations=method.iterations,
                        power_iterations=method.power_iterations,
                        calib_data=str(ctx["calib_root"]))
        settings.update({key.replace("-", "_"): value for key, value in method.extra_args.items()})
        return compression_signature(settings)

    def reusable_checkpoint(self, ctx):
        import json
        path = self.checkpoint_path(ctx)
        if not checkpoint_complete(path):
            return None
        try:
            if json.loads((path / "compression_recipe.json").read_text()) == self.signature(ctx):
                return path
        except (OSError, ValueError):
            pass
        return None

    def run(self, ctx):
        cmd = self.build_command(ctx)
        if not 0.0 <= float(ctx["cfg"].compression_ratio) < 1.0:
            raise ValueError("compression_ratio must be in [0, 1)")
        if not ctx["dry_run"] and not Path(ctx["calib_root"]).is_dir():
            raise FileNotFoundError(
                f"Calibration directory not found: {ctx['calib_root']}. "
                "Set calib_root or collect statistics with stages.collect=true."
            )
        run_checked(
            cmd,
            cwd=ctx["repo_root"],
            dry_run=ctx["dry_run"],
            log_path=ctx["run_dir"] / "compress.log",
        )
        if ctx["dry_run"]:
            return {
                "command": cmd,
                "model_path": None,
                "backend": ctx["method"].name,
                "status": "planned",
            }
        path = self.checkpoint_path(ctx)
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Compression did not produce the expected checkpoint: {path}")
        return {"command": cmd, "model_path": str(path), "backend": ctx["method"].name}
