"""Export sparse or dense U/V factors as a deployable Transformers model.

Supports legacy ``*_separate`` HF checkpoints and per-projection ``*_u.pth`` /
``*_v.pth`` factors beside a fused HF checkpoint. No factorization is inferred
from fused weights: original factors must be present.
"""

import argparse
import json
import math
import re
import shutil
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from huggingface_hub import split_torch_state_dict_into_shards
from .modelings.compot_sparse import pack_weight

PROJECTION = re.compile(
    r"^(model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:up|gate|down)_proj))\.weight$"
)
FACTOR = re.compile(
    r"^(model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:up|gate|down)_proj))_([uv])\.weight$"
)


class TensorReader:
    def __init__(self, root):
        self.root = Path(root)
        self.paths = {}
        self.bins = {}
        for file in sorted(self.root.glob("*.safetensors")):
            with safe_open(file, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key in self.paths:
                        raise ValueError(f"Duplicate tensor {key}")
                    self.paths[key] = file
        if not self.paths:
            for file in self.root.glob("pytorch_model*.bin"):
                data = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
                self.bins.update(data)
        self.keys = set(self.paths) | set(self.bins)
        if not self.keys:
            raise ValueError("No Hugging Face tensor checkpoint found")

    def get(self, key):
        if key in self.bins:
            return self.bins[key]
        with safe_open(self.paths[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def convert_checkpoint(source, output, target_cr, factors=None, state_dict=None,
                       method=None, ks_ratio=None):
    source, output = Path(source), Path(output)
    factors = Path(factors) if factors else source
    if output.resolve() == source.resolve() or output.resolve() == factors.resolve():
        raise ValueError("Output must be separate from the source checkpoints")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose an empty output directory: {output}")
    config = json.loads((source / "config.json").read_text())
    family = config["model_type"]
    if family not in ("llama", "qwen3"):
        raise ValueError(f"Unsupported architecture: {family}")
    if state_dict is None:
        reader = TensorReader(source)
    else:

        class StateReader:
            keys = set(state_dict)

            def get(self, key):
                return state_dict[key].detach().cpu()

        reader = StateReader()
    pairs = {}
    for key in reader.keys:
        match = FACTOR.fullmatch(key)
        if match:
            pairs.setdefault(match[1], {})[match[2]] = ("hf", key)
    for file in factors.glob("model.layers.*_u.pth"):
        name = file.name[: -len("_u.pth")]
        if not PROJECTION.fullmatch(name + ".weight"):
            continue
        pairs[name] = {"u": ("pth", file), "v": ("pth", file.with_name(name + "_v.pth"))}
    if not pairs:
        raise ValueError("Original U/V factors are required; fused weights cannot recover them")
    for name, pair in pairs.items():
        if set(pair) != {"u", "v"}:
            raise ValueError(f"Incomplete factor pair: {name}")
    dense_keys = {key for key in reader.keys if PROJECTION.fullmatch(key)}
    projection_names = {key[:-7] for key in dense_keys} | set(pairs)
    expected_count = int(config["num_hidden_layers"]) * 7
    if len(projection_names) != expected_count:
        raise ValueError(
            f"Expected all {expected_count} core projections, found {len(projection_names)}"
        )
    state, specs, rows = {}, {}, []
    original_projection_bytes = packed_projection_bytes = 0
    for name in sorted(projection_names):
        key = name + ".weight"
        if name not in pairs:
            weight = reader.get(key).to(torch.bfloat16).contiguous()
            state[key] = weight
            original_projection_bytes += tensor_bytes(weight)
            packed_projection_bytes += tensor_bytes(weight)
            rows.append(
                dict(
                    path=name,
                    storage="dense",
                    dense_bytes=tensor_bytes(weight),
                    stored_bytes=tensor_bytes(weight),
                )
            )
            continue
        pair = pairs[name]
        values = {}
        for side in ("u", "v"):
            kind, location = pair[side]
            tensor = (
                reader.get(location)
                if kind == "hf"
                else torch.load(location, map_location="cpu", weights_only=True, mmap=True).T
            )
            values[side] = tensor.to(torch.bfloat16).contiguous()
        u, v = values["u"], values["v"]
        if u.ndim != 2 or v.ndim != 2 or v.shape[1] != u.shape[0]:
            raise ValueError(f"Invalid factor shapes for {name}: {u.shape}, {v.shape}")
        if not torch.isfinite(u).all() or not torch.isfinite(v).all():
            raise ValueError(f"Non-finite factor values for {name}")
        k, in_features = u.shape
        out_features = v.shape[0]
        mask, nonzeros = pack_weight(v)
        sparse_bytes = tensor_bytes(mask) + tensor_bytes(nonzeros)
        dense_v = tensor_bytes(v) <= sparse_bytes
        storage = "dense_uv_v1" if dense_v else "packed_mask_v1"
        dense_bytes = in_features * out_features * 2
        stored_bytes = tensor_bytes(u) + (tensor_bytes(v) if dense_v else sparse_bytes)
        original_projection_bytes += dense_bytes
        if stored_bytes >= dense_bytes:
            # Dense (zero-allocation) projections should not grow from identity factors.
            weight = reader.get(key).to(torch.bfloat16) if key in reader.keys else v @ u
            state[key] = weight.contiguous()
            packed_projection_bytes += dense_bytes
            rows.append(
                dict(path=name, storage="dense", dense_bytes=dense_bytes, stored_bytes=dense_bytes)
            )
            continue
        state[name + "_u.weight"] = u
        if dense_v:
            state[name + "_v.weight"] = v
        else:
            state[name + "_v.packed_mask"] = mask
            state[name + "_v.values"] = nonzeros
        specs[name] = dict(
            k=k,
            in_features=in_features,
            out_features=out_features,
            nnz=nonzeros.numel(),
            storage=storage,
        )
        packed_projection_bytes += stored_bytes
        rows.append(
            dict(
                path=name,
                storage=storage,
                dense_bytes=dense_bytes,
                stored_bytes=stored_bytes,
                rank=k,
                nnz=nonzeros.numel(),
            )
        )
    for key in sorted(reader.keys):
        if PROJECTION.fullmatch(key) or FACTOR.fullmatch(key):
            continue
        renamed = key
        if key.endswith(".bias") and key[:-5] in specs:
            renamed = key[:-5] + "_v.bias"
        tensor = reader.get(key)
        state[renamed] = (
            tensor.to(torch.bfloat16).contiguous()
            if tensor.is_floating_point()
            else tensor.contiguous()
        )
    if (
        config.get("tie_word_embeddings")
        and "lm_head.weight" in state
        and "model.embed_tokens.weight" in state
    ):
        if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
            raise ValueError("Config requests tied embeddings, but lm_head differs from embeddings")
        del state["lm_head.weight"]
    projection_tensor_keys = {
        name + suffix
        for name in projection_names
        for suffix in ('.weight', '_u.weight', '_v.weight', '_v.packed_mask', '_v.values')
    }
    unchanged_core_bytes = sum(
        tensor_bytes(t)
        for key, t in state.items()
        if key.startswith("model.layers.")
        and key not in projection_tensor_keys
    )
    dense_core_bytes = original_projection_bytes + unchanged_core_bytes
    packed_core_bytes = packed_projection_bytes + unchanged_core_bytes
    achieved = 1 - packed_core_bytes / dense_core_bytes
    if packed_core_bytes > math.floor((1 - float(target_cr)) * dense_core_bytes):
        raise ValueError(
            f"Core storage reduction {achieved:.6%} misses requested {target_cr:.6%}; no checkpoint exported"
        )
    checkpoint_format = (
        "factorized_v1" if any(s["storage"] == "dense_uv_v1" for s in specs.values())
        else "packed_mask_v1"
    )
    config.update(
        compot_checkpoint_format=checkpoint_format,
        factor_shapes=specs,
        torch_dtype="bfloat16",
        auto_map={
            "AutoModelForCausalLM": f"modeling_{family}_svd."
            + ("Llama" if family == "llama" else "Qwen3")
            + "ForCausalLM"
        },
    )
    if method is not None:
        config["compression_method"] = method
    if ks_ratio is not None:
        config["ks_ratio"] = float(ks_ratio)
    output.mkdir(parents=True, exist_ok=True)
    plan = split_torch_state_dict_into_shards(
        state, filename_pattern="model{suffix}.safetensors", max_shard_size="2GB"
    )
    for filename, keys in plan.filename_to_tensors.items():
        save_file({key: state[key] for key in keys}, output / filename, metadata={"format": "pt"})
    if plan.is_sharded:
        (output / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": plan.metadata, "weight_map": plan.tensor_to_filename}, indent=2)
        )
    (output / "config.json").write_text(json.dumps(config, indent=2))
    for file in source.iterdir():
        if file.is_file() and (
            file.name.startswith(
                (
                    "tokenizer",
                    "special_tokens",
                    "added_tokens",
                    "vocab",
                    "merges",
                    "generation_config",
                    "chat_template",
                )
            )
            or file.name == "tokenizer.model"
        ):
            shutil.copy2(file, output / file.name)
    modeling_root = Path(__file__).parent / "modelings"
    for name in (f"modeling_{family}_svd.py", "compot_sparse.py"):
        shutil.copy2(modeling_root / name, output / name)
    report = dict(
        format=checkpoint_format,
        method=method,
        ks_ratio=ks_ratio,
        source=str(source),
        factors=str(factors),
        target_core_cr=target_cr,
        achieved_core_cr=achieved,
        dense_core_bytes=dense_core_bytes,
        packed_core_bytes=packed_core_bytes,
        projection_cr=1 - packed_projection_bytes / original_projection_bytes,
        total_tensor_payload_bytes=sum(tensor_bytes(t) for t in state.values()),
        actual_safetensors_file_bytes=sum(f.stat().st_size for f in output.glob("*.safetensors")),
        scope="all decoder-layer parameters; embeddings and lm_head excluded from core CR; norms/biases included",
        runtime="Dense U/V factors or packed sparse coefficients remain persistent; packed coefficients use a forward temporary; no fused weight cache",
        factorized_projections=len(specs),
        projections=rows,
    )
    (output / "compression_manifest.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--factors")
    parser.add_argument("--output", required=True)
    parser.add_argument("--target-cr", type=float, required=True)
    args = parser.parse_args()
    report = convert_checkpoint(args.source, args.output, args.target_cr, args.factors)
    print(json.dumps({k: v for k, v in report.items() if k != "projections"}, indent=2))


if __name__ == "__main__":
    main()
