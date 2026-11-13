"""Calibration-whitened truncated SVD with balanced dense U/V factors."""

import torch
from .calibration import Calib


def rank_from_compression_ratio(shape, compression_ratio):
    d1, d2 = shape
    if compression_ratio == 0:
        return min(d1, d2)
    rank = int(d1 * d2 * (1 - compression_ratio) / (d1 + d2))
    return max(1, min(rank, d1, d2))


@torch.no_grad()
def svdllm(weight, name="self_attn.q_proj", index=0, compression_ratio=0.2,
           calib_data=None, device=None):
    """Factor W.T as U @ V using the calibration-weighted SVD-LLM objective.

    The singular-value square root is split between U and V. Whitening uses
    the same Cholesky factor and inverse as COMPOT and CoSpaDi.
    """
    if not 0 <= compression_ratio < 1:
        raise ValueError("compression_ratio must be in [0, 1)")
    if compression_ratio == 0:
        return weight, torch.eye(weight.shape[1], device=weight.device, dtype=weight.dtype)
    group = index if isinstance(index, list) else [index]
    target = torch.device(device or weight.device)
    if calib_data is None:
        scaling = inverse = torch.eye(weight.shape[0], device=target, dtype=torch.float64)
    else:
        scaling, inverse = Calib.get_s_inv_s(group, name, "llama3", calib_data)
        scaling = scaling.to(device=target, dtype=torch.float64)
        inverse = inverse.to(device=target, dtype=torch.float64)
    whitened = scaling @ weight.to(device=target, dtype=torch.float64)
    u, sigma, vh = torch.linalg.svd(whitened, full_matrices=False)
    rank = rank_from_compression_ratio(weight.shape, compression_ratio)
    root_sigma = sigma[:rank].sqrt()
    u = (inverse @ u[:, :rank]) * root_sigma.unsqueeze(0)
    v = root_sigma.unsqueeze(1) * vh[:rank, :]
    return u.cpu(), v.cpu()
