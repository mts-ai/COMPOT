"""Allocate per-projection ranks under a global compression budget.

Truncate singular values across matrices subject to rank bounds, retain
dense layers when factorization saves no parameters, and search the global
truncation budget to match the requested compression ratio.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import logging

import torch
from torch import nn

from .compot import get_k_and_sparsity

def truncate_global_smallest_svals_constrained(
    weights: Optional[List[torch.Tensor]] = None,
    svals: Optional[List[torch.Tensor]] = None,
    k: int = 0,
    *,
    min_rank_per_matrix: Optional[List[int]] = None,
    min_trunc_per_matrix: Optional[List[int]] = None,  # Minimum truncations per matrix
    return_details: bool = True,
) -> Dict[str, Any]:
    """
    Select the global k smallest singular values across multiple matrices while
    respecting minimum rank and minimum truncation constraints per matrix.

    Args:
        weights: Optional list of 2D tensors. Used only if svals is None.
        svals: Optional list of 1D tensors, singular values per matrix.
        k: Number of singular values to truncate globally (k smallest).
        min_rank_per_matrix: List[int] specifying minimum retained rank (max_cr guard)
        min_trunc_per_matrix: List[int] specifying minimum truncations (min_cr guard)
        return_details: If True, also returns per-matrix indices and values.

    Returns:
        Dict with:
          - counts_per_matrix: List[int] truncation count per matrix
          - selected_values: (k_eff,) tensor of selected singular values
          - selected_global_indices: (k_eff,) tensor into concatenated vector
          - selected_matrix_ids: (k_eff,) tensor of matrix id for each value
          - selected_local_indices: List[LongTensor] local indices per matrix
          - constraint_satisfied: bool indicating if all constraints met
    """
    if k < 0:
        raise ValueError("k must be >= 0")

    if svals is None:
        if weights is None:
            raise ValueError("Provide either svals or weights.")
        svals = [torch.linalg.svdvals(W) for W in weights]

    n = len(svals)
    if n == 0:
        return {
            "counts_per_matrix": [],
            "selected_values": torch.empty(0),
            "selected_global_indices": torch.empty(0, dtype=torch.long),
            "selected_matrix_ids": torch.empty(0, dtype=torch.long),
            "selected_local_indices": [],
            "constraint_satisfied": True,
        }

    device = svals[0].device
    svals_1d = [sv.reshape(-1).to(device) for sv in svals]
    lengths = torch.tensor([sv.numel() for sv in svals_1d], device=device, dtype=torch.long)
    total = int(lengths.sum().item())

    # Setup constraints
    if min_rank_per_matrix is not None:
        if len(min_rank_per_matrix) != n:
            raise ValueError(f"min_rank_per_matrix length {len(min_rank_per_matrix)} != {n}")
        max_truncate_per_matrix = [
            max(0, int(lengths[i].item()) - int(min_rank_per_matrix[i]))
            for i in range(n)
        ]
    else:
        max_truncate_per_matrix = [int(L.item()) for L in lengths]
    
    # Minimum truncation constraints for min_cr
    if min_trunc_per_matrix is not None:
        if len(min_trunc_per_matrix) != n:
            raise ValueError(f"min_trunc_per_matrix length {len(min_trunc_per_matrix)} != {n}")
        # Ensure min_trunc doesn't exceed max_trunc
        for i in range(n):
            if min_trunc_per_matrix[i] > max_truncate_per_matrix[i]:
                raise ValueError(f"min_trunc_per_matrix[{i}]={min_trunc_per_matrix[i]} > max_truncate={max_truncate_per_matrix[i]}")
    else:
        min_trunc_per_matrix = [0] * n

    k_eff = min(int(k), total)

    # Check if k meets minimum truncation requirements
    total_min_trunc = sum(min_trunc_per_matrix)
    if k_eff < total_min_trunc:
        raise ValueError(f"k_eff={k_eff} < total_min_trunc={total_min_trunc}. Increase k or reduce min_trunc constraints.")

    if k_eff == 0:
        empty_long = torch.empty(0, device=device, dtype=torch.long)
        return {
            "counts_per_matrix": [0] * n,
            "selected_values": torch.empty(0, device=device),
            "selected_global_indices": empty_long,
            "selected_matrix_ids": empty_long,
            "selected_local_indices": [empty_long.clone() for _ in range(n)],
            "constraint_satisfied": True,
        }

    # Flatten and map positions
    flat = torch.cat(svals_1d, dim=0)
    matrix_ids = torch.repeat_interleave(torch.arange(n, device=device), lengths)
    local_ids = torch.cat([torch.arange(int(L.item()), device=device) for L in lengths], dim=0)

    # Pre-select mandatory truncations (min_trunc_per_matrix)
    mandatory_indices = []
    truncate_counts = [0] * n
    remaining_budget = k_eff - total_min_trunc
    
    # First pass: select the smallest min_trunc_per_matrix[i] SVs for each matrix
    for i in range(n):
        if min_trunc_per_matrix[i] == 0:
            continue
            
        # Get indices for this matrix
        mask = (matrix_ids == i)
        mat_svs = flat[mask]
        mat_local_ids = local_ids[mask]
        
        # Get smallest min_trunc_per_matrix[i] values
        _, smallest_indices = torch.topk(mat_svs, min_trunc_per_matrix[i], largest=False)
        global_indices = torch.where(mask)[0][smallest_indices]
        
        mandatory_indices.extend(global_indices.tolist())
        truncate_counts[i] = min_trunc_per_matrix[i]
    
    # Second pass: select remaining from non-mandatory SVs
    if remaining_budget > 0:
        # Create mask for non-mandatory SVs
        non_mandatory_mask = torch.ones_like(flat, dtype=torch.bool)
        non_mandatory_mask[mandatory_indices] = False
        
        # Filter remaining values
        remaining_svs = flat[non_mandatory_mask]
        remaining_matrix_ids = matrix_ids[non_mandatory_mask]
        remaining_local_ids = local_ids[non_mandatory_mask]
        remaining_global_indices = torch.arange(total, device=device)[non_mandatory_mask]
        
        # Sort remaining SVs (smallest first)
        sorted_indices = torch.argsort(remaining_svs, descending=False)
        
        # Select up to remaining_budget items respecting max_trunc constraints
        selected_indices = []
        for idx in sorted_indices[:remaining_budget].tolist():
            mat_id = int(remaining_matrix_ids[idx].item())
            current_trunc = truncate_counts[mat_id]
            max_allowed = max_truncate_per_matrix[mat_id]
            
            if current_trunc < max_allowed:
                selected_indices.append(int(remaining_global_indices[idx].item()))
                truncate_counts[mat_id] += 1
            # Else: skip due to max_trunc constraint

        # Combine mandatory and selected indices
        selected_global_idx_list = mandatory_indices + selected_indices
    else:
        selected_global_idx_list = mandatory_indices

    # Prepare outputs
    selected_global_idx = torch.tensor(selected_global_idx_list, device=device, dtype=torch.long)
    selected_values = flat[selected_global_idx] if selected_global_idx.numel() else torch.empty(0, device=device)
    selected_matrix_ids = matrix_ids[selected_global_idx] if selected_global_idx.numel() else torch.empty(0, device=device, dtype=torch.long)
    selected_local_idx = local_ids[selected_global_idx] if selected_global_idx.numel() else torch.empty(0, device=device, dtype=torch.long)

    # Verify constraints
    constraint_violated = False
    for i in range(n):
        if truncate_counts[i] < min_trunc_per_matrix[i]:
            constraint_violated = True
        if truncate_counts[i] > max_truncate_per_matrix[i]:
            constraint_violated = True

    out: Dict[str, Any] = {
        "counts_per_matrix": truncate_counts,
        "constraint_satisfied": not constraint_violated,
    }

    if return_details:
        per_matrix_local: List[torch.Tensor] = []
        for i in range(n):
            mask = (selected_matrix_ids == i)
            per_matrix_local.append(selected_local_idx[mask])

        out.update(
            selected_values=selected_values,
            selected_global_indices=selected_global_idx,
            selected_matrix_ids=selected_matrix_ids,
            selected_local_indices=per_matrix_local,
        )

    return out


def _layer_params_and_cr(m: int, n: int, r: int, *, force_dense: bool = False) -> Tuple[int, float, bool]:
    """
    Decide whether to keep a layer dense or factorized at rank r.

    Dense params: m*n
    Factorized params (SVD-style): r*(m+n)

    Returns:
        params_used, cr_i, is_dense
    """
    dense_params = m * n
    fact_params = r * (m + n)

    if force_dense or (fact_params >= dense_params):
        return dense_params, 0.0, True

    cr_i = 1.0 - (fact_params / dense_params)
    return fact_params, float(cr_i), False


def allocate_counts_with_dense_loop(
    *,
    svals_list: List[torch.Tensor],
    shapes: List[Tuple[int, int]],
    target_cr: float,
    min_rank_per_matrix: Optional[List[int]] = None,
    max_rank_per_matrix: Optional[List[int]] = None,  # Maximum rank constraints
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Iteratively:
      - binary search k to match global parameter target
      - if any layer ends up non-beneficial for factorization (r*(m+n) >= m*n),
        mark dense and repeat
      - respect both min_rank (max_cr) and max_rank (min_cr) constraints

    Returns:
      counts_per_matrix, dense_mask, cr_per_matrix, achieved_cr, params_compressed, params_vanilla
    """
    if not (0.0 <= target_cr < 1.0):
        raise ValueError("target_cr must be in [0, 1).")

    n = len(shapes)
    if len(svals_list) != n:
        raise ValueError(f"svals_list length {len(svals_list)} != shapes length {n}")

    Ls = [min(m, n) for (m, n) in shapes]
    params_vanilla = sum(m * n for (m, n) in shapes)
    target_params = int(round((1.0 - target_cr) * params_vanilla))

    # Setup min-rank constraint (max_cr guard)
    if min_rank_per_matrix is None:
        min_rank_per_matrix = [0] * n
    if len(min_rank_per_matrix) != n:
        raise ValueError(f"min_rank_per_matrix length {len(min_rank_per_matrix)} != {n}")

    # Maximum rank constraint for min_cr
    if max_rank_per_matrix is not None:
        if len(max_rank_per_matrix) != n:
            raise ValueError(f"max_rank_per_matrix length {len(max_rank_per_matrix)} != {n}")
        # Convert max_rank to min_truncation requirements
        min_trunc_per_matrix = [
            max(0, Ls[i] - int(max_rank_per_matrix[i]))
            for i in range(n)
        ]
    else:
        min_trunc_per_matrix = [0] * n

    # max truncation from min rank
    max_trunc = [max(0, Ls[i] - int(min_rank_per_matrix[i])) for i in range(n)]

    # Validate constraints
    for i in range(n):
        if min_trunc_per_matrix[i] > max_trunc[i]:
            raise ValueError(f"Layer {i}: min_trunc({min_trunc_per_matrix[i]}) > max_trunc({max_trunc[i]})")

    dense_mask = [False] * n
    counts = [0] * n

    # Outer loop: each iteration can only add dense layers -> <= n iterations
    for outer in range(n + 1):
        active = [i for i in range(n) if not dense_mask[i]]
        if verbose:
            logging.info(f"[densify iter {outer}] active={len(active)} dense={n-len(active)}")

        if len(active) == 0:
            break  # nothing left to compress

        # Pre-densify layers that are never beneficial even at minimum allowed rank
        changed = False
        for i in active:
            m, nn = shapes[i]
            r_min = max(0, int(min_rank_per_matrix[i]))
            _, _, is_dense_at_rmin = _layer_params_and_cr(m, nn, r_min, force_dense=False)
            if is_dense_at_rmin:
                dense_mask[i] = True
                counts[i] = 0
                changed = True
        if changed:
            continue

        active = [i for i in range(n) if not dense_mask[i]]
        if len(active) == 0:
            break

        # Binary search k over ACTIVE layers (k == total truncations across active)
        lo = sum(min_trunc_per_matrix[i] for i in active)  # Respect min_trunc constraints
        hi = sum(max_trunc[i] for i in active)

        if lo > hi:
            raise ValueError(f"Binary search bounds invalid: lo={lo} > hi={hi}. Constraint conflict.")

        best_counts_active: Optional[List[int]] = None
        best_params: Optional[int] = None

        while lo < hi:
            mid = (lo + hi) // 2

            svals_active = [svals_list[i] for i in active]
            min_rank_active = [min_rank_per_matrix[i] for i in active]
            min_trunc_active = [min_trunc_per_matrix[i] for i in active]

            try:
                res = truncate_global_smallest_svals_constrained(
                    svals=svals_active,
                    k=mid,
                    min_rank_per_matrix=min_rank_active,
                    min_trunc_per_matrix=min_trunc_active,
                    return_details=False,
                )
                counts_active = res["counts_per_matrix"]
                constraint_satisfied = res["constraint_satisfied"]
            except ValueError as e:
                # Constraint violation - need more truncations
                lo = mid + 1
                continue

            if not constraint_satisfied:
                # Not enough truncations to satisfy min_trunc constraints
                lo = mid + 1
                continue

            # Map to full counts
            tmp_counts = counts[:]
            for j, i in enumerate(active):
                tmp_counts[i] = counts_active[j]

            # Evaluate total parameters and detect newly non-beneficial layers
            params_total = 0
            newly_dense: List[int] = []
            for i in range(n):
                m, nn = shapes[i]
                L = Ls[i]
                r = L - tmp_counts[i]
                params_i, _, is_dense = _layer_params_and_cr(m, nn, r, force_dense=dense_mask[i])
                params_total += params_i
                if (not dense_mask[i]) and is_dense:
                    newly_dense.append(i)

            if newly_dense:
                for i in newly_dense:
                    dense_mask[i] = True
                    counts[i] = 0
                best_counts_active = None
                best_params = None
                break  # restart outer loop

            # params_total is non-increasing as k increases
            if params_total <= target_params:
                hi = mid
                best_counts_active = counts_active
                best_params = params_total
            else:
                lo = mid + 1

        # If densification triggered mid-search, restart outer loop
        if best_counts_active is None and best_params is None:
            continue

        # Finalize with k = hi
        k_final = hi
        svals_active = [svals_list[i] for i in active]
        min_rank_active = [min_rank_per_matrix[i] for i in active]
        min_trunc_active = [min_trunc_per_matrix[i] for i in active]

        res = truncate_global_smallest_svals_constrained(
            svals=svals_active,
            k=k_final,
            min_rank_per_matrix=min_rank_active,
            min_trunc_per_matrix=min_trunc_active,
            return_details=False,
        )
        counts_active = res["counts_per_matrix"]
        for j, i in enumerate(active):
            counts[i] = counts_active[j]

        # One last check: if final counts make something non-beneficial, densify and rerun
        newly_dense_final: List[int] = []
        for i in active:
            m, nn = shapes[i]
            L = Ls[i]
            r = L - counts[i]
            _, _, is_dense = _layer_params_and_cr(m, nn, r, force_dense=False)
            if is_dense:
                newly_dense_final.append(i)

        if newly_dense_final:
            for i in newly_dense_final:
                dense_mask[i] = True
                counts[i] = 0
            continue

        # Stable solution found
        break

    # Final metrics
    cr_per_matrix: List[float] = []
    params_compressed = 0
    for i in range(n):
        m, nn = shapes[i]
        L = Ls[i]
        r = L - counts[i]
        params_i, cr_i, is_dense = _layer_params_and_cr(m, nn, r, force_dense=dense_mask[i])
        params_compressed += params_i
        cr_per_matrix.append(0.0 if is_dense else cr_i)

    achieved_cr = 1.0 - (params_compressed / params_vanilla) if params_vanilla > 0 else 0.0

    return {
        "counts_per_matrix": counts,
        "dense_mask": dense_mask,
        "cr_per_matrix": cr_per_matrix,
        "achieved_cr": float(achieved_cr),
        "params_compressed": int(params_compressed),
        "params_vanilla": int(params_vanilla),
        "target_cr": float(target_cr),
    }

def get_dynamic_cr_svd_all_enhanced_mincr(
    model: nn.Module,
    module_names: List[str],
    cr: float,
    calib_data: Optional[str] = None,
    use_calib: bool = False,
    normalization: bool = False,
    norm: str = "fro",
    in_importance: bool = False,
    out_importance: bool = False,
    ks_ratio: float = 2.0, # to compute CR with k/s
    *,
    arch: str = "llama3",
    sv_dtype: torch.dtype = torch.double,
    max_cr: Optional[float] = None,
    min_cr: Optional[float] = None,  # Minimum compression ratio per layer
    verbose: bool = True,
) -> Dict[str, float]:
    """Compute per-module compression ratios from constrained spectral allocation.

    Singular values determine a global truncation order. Binary search matches
    the parameter budget implied by ``cr``, while layers with no factorization
    savings remain dense. Optional ``min_cr`` and ``max_cr`` bound each layer.

    Returns:
        Mapping from module paths to allocated compression ratios.
    """
    if not (0.0 <= cr < 1.0):
        raise ValueError("target cr must be in [0, 1).")

    device = "cuda" if torch.cuda.is_available() else next(model.parameters()).device

    # Validate min_cr/max_cr
    if min_cr is not None and max_cr is not None:
        if min_cr > max_cr:
            raise ValueError(f"min_cr ({min_cr}) cannot be greater than max_cr ({max_cr})")

    try:
        if hasattr(model.model, "language_model"):
            n_layers = len(model.model.language_model.layers)  # assumes HF-style llama: model.model.layers
        else:
            n_layers = len(model.model.layers)  # assumes HF-style llama: model.model.layers
    except: # for OPT model
        n_layers = len(model.model.decoder.layers)  # assumes HF-style opt: model.model.layers

    layer_indices = list(range(n_layers))

    cr_allocation: Dict[str, float] = {}
    entries: List[Tuple[str, int, int]] = []
    svals_list: List[torch.Tensor] = []
    shapes: List[Tuple[int, int]] = []

    for module_name in module_names:
        for layer_idx in layer_indices:
            try:
                if hasattr(model.model, "language_model"):
                    full_name = f"model.language_model.layers.{layer_idx}.{module_name}"
                else:
                    full_name = f"model.layers.{layer_idx}.{module_name}"
                W = model.get_submodule(full_name).weight.detach().T.contiguous()
            except: # opt
                full_name = f"model.decoder.layers.{layer_idx}.{module_name}"
                W = model.get_submodule(full_name).weight.detach().T.contiguous()

            if use_calib and calib_data is not None:
                from .calibration import Calib  # type: ignore
                ss, _ = Calib.get_s_inv_s([layer_idx], module_name, arch, calib_data)
                ss = ss.to(device=device, dtype=sv_dtype)
                W = W.to(device=device, dtype=sv_dtype)
                W = ss @ W
            else:
                W = W.to(device=device, dtype=sv_dtype)

            if normalization:
                # normalize on GPU if possible
                W = W.to("cuda") if torch.cuda.is_available() else W
                W = W / torch.linalg.matrix_norm(W, ord=norm)
                W = W.to(device=device, dtype=sv_dtype)

            if in_importance and calib_data is not None:
                from .calibration import Calib  # type: ignore
                ss, _ = Calib.get_s_inv_s([layer_idx], module_name, arch, calib_data)
                ss = ss.to(device=device, dtype=sv_dtype)
                G_x = ss @ ss.t()
                input_importance = torch.sqrt(torch.diag(G_x) + 1e-8)
            else:
                input_importance = None

            if out_importance and calib_data is not None:
                from .calibration import Calib  # type: ignore
                ss, _ = Calib.get_s_inv_s([layer_idx], module_name, arch, calib_data)
                ss = ss.to(device=device, dtype=sv_dtype)
                G_xx = ss @ ss.t()
                G_y = W.t() @ G_xx @ W
                output_importance = torch.sqrt(torch.diag(G_y) + 1e-8)
            else:
                output_importance = None

            m, n = W.shape
            shapes.append((m, n))

            s = torch.linalg.svdvals(W).reshape(-1)
            if input_importance is not None:
                s = s * input_importance
            if output_importance is not None:
                s = s * output_importance

            s = s.to("cpu")
            svals_list.append(s)
            entries.append((full_name, m, n))

    # Build constraints
    min_rank_per_matrix: Optional[List[int]] = None
    max_rank_per_matrix: Optional[List[int]] = None
    
    # min_rank from max_cr (minimum rank to retain)
    if max_cr is not None:
        min_rank_per_matrix = []
        for (m, n) in shapes:
            r_min = int((m * n * (1.0 - max_cr)) / (m + n))
            min_rank_per_matrix.append(max(0, r_min))
    
    # max_rank from min_cr (maximum rank allowed)
    if min_cr is not None:
        max_rank_per_matrix = []
        for (m, n) in shapes:
            # Solve: 1 - [r*(m+n)]/(m*n) >= min_cr
            r_max = int((1.0 - min_cr) * (m * n) / (m + n))
            # Ensure we don't exceed matrix dimensions
            r_max = min(r_max, min(m, n))
            max_rank_per_matrix.append(max(0, r_max))
    
    # Allocate counts with iterative densification and global param matching
    alloc = allocate_counts_with_dense_loop(
        svals_list=svals_list,
        shapes=shapes,
        target_cr=cr,
        min_rank_per_matrix=min_rank_per_matrix,
        max_rank_per_matrix=max_rank_per_matrix,
        verbose=verbose,
    )

    counts = alloc["counts_per_matrix"]
    dense_mask = alloc["dense_mask"]

    # Convert counts -> per-layer CR allocation
    for i, (full_name, m_i, n_i) in enumerate(entries):
        if dense_mask[i]:
            cr_allocation[full_name] = 0.0
            continue

        L_i = min(m_i, n_i)
        r_i = L_i - counts[i]
        _, cr_i, is_dense = _layer_params_and_cr(m_i, n_i, r_i, force_dense=False)
        cr_allocation[full_name] = 0.0 if is_dense else float(cr_i)

    if verbose:
        logging.info(f"Target CR: {cr:.4f} | Achieved CR: {alloc['achieved_cr']:.4f}")

    # Get cr with k/s
    vanilla_params = 0
    compres_params = 0
    for i, (full_name, m_i, n_i) in enumerate(entries):
        vanilla_params += m_i * n_i
        cr_tmp = cr_allocation[full_name]
        if cr_tmp == 0:
            compres_params += m_i * n_i
        elif cr_tmp > 0:
            k, s = get_k_and_sparsity(cr=cr_tmp, d1=m_i, d2=n_i, ks_ratio=ks_ratio)
            compres_params += m_i * k + s * n_i + k * n_i / 16
    cr_ks = 1 - compres_params / vanilla_params
    if verbose:
        logging.info(f"k/s CR: {cr_ks:.4f}")

    assert cr_ks >= cr, f"KS_CR is lower than target CR, something went wrong, consider softening/removing min_cr constraint!!!"

    return cr_allocation