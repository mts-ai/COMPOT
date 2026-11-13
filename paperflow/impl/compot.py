"""COMPOT sparse coding with calibration-weighted orthogonal Procrustes updates."""
import gc, logging
from os.path import isdir, isfile, join
from typing import Optional, Tuple, List
import numpy as np
import torch
from tabulate import tabulate
from .calibration import Calib

def get_k_and_sparsity(cr, d1, d2, ks_ratio, divisibility=1):
    """Choose dictionary rank and per-column sparsity for one projection.

    The storage estimate includes dictionary values, nonzero coefficients and
    a one-bit coefficient mask. Layer counts are handled by the allocator.
    """
    k = (1-cr) * d1 * d2 / (d1 + d2 / ks_ratio + d2 / 16)
    if divisibility > 1:
        k = (k // divisibility) * divisibility
        sparsity, _ = get_sparsity(cr, d1, d2, k)
    else: # old behaviour
        sparsity = k / ks_ratio
    return int(k), int(sparsity)

def get_sparsity(cr, d1, d2, k):
    """Compute per-column coefficient sparsity for a fixed dictionary rank."""
    s = ((1 - cr) * d1 * d2 - k * (d1 + d2 / 16)) / d2
    ks_ratio = k / int(s)
    return int(s), ks_ratio

def subsample_equal_step_numpy(elements):
    elements = [f'{i:.5f}' for i in elements] 
    n = len(elements)
    if n < 10:
        return list(range(len(elements))), elements
        
    indices_float = np.linspace(0, n-1, 10)
    indices = [int(round(idx)) for idx in indices_float]
    
    indices[0] = 0
    indices[-1] = n - 1
    
    unique_indices = []
    for idx in indices:
        if idx not in unique_indices:
            unique_indices.append(idx)
    
    while len(unique_indices) < 10 and n > 10:
        max_gap = 0
        insert_position = 0
        insert_value = 0
        
        for i in range(1, len(unique_indices)):
            gap = unique_indices[i] - unique_indices[i-1]
            if gap > max_gap:
                max_gap = gap
                insert_position = i
                insert_value = (unique_indices[i-1] + unique_indices[i]) // 2
        
        if insert_value not in unique_indices:
            unique_indices.insert(insert_position, insert_value)
    
    return unique_indices, [elements[i] for i in unique_indices[:10]]

def _normalize_columns_(D: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    nrm = torch.linalg.norm(D, dim=0, keepdim=True).clamp_min(eps)
    D /= nrm
    return D

def _maybe_enable_fast_kernels():
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True


def extrapolate_dictionary(current, previous, iteration, iterations, mode, device):
    if mode != 'default':
        raise ValueError('Only the standard COMPOT update is supported')
    return current, None

def sparse_code_orthonormal(
    W: torch.Tensor,
    D: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Analytic sparse coding for an orthonormal dictionary.

    Solves   min_S ||W - D @ S||_F^2   subject to   ||s_j||_0 = k  for every column s_j,
    assuming D.T @ D = I (columns of D are orthonormal).

    Parameters
    ----------
    W : torch.Tensor, shape (n, p)
        Data matrix: n features, p samples (columns).
    D : torch.Tensor, shape (n, m)
        Dictionary with orthonormal columns (D.T @ D == I).
    k : int
        Desired number of non‑zero entries per column of S (0 < k ≤ m).

    Returns
    -------
    S : torch.Tensor, shape (m, p)
        Sparse coefficient matrix. Exactly k entries per column are non‑zero
        (unless some of the top‑k magnitudes are exactly zero).

    Raises
    ------
    ValueError
        If input shapes are incompatible or k is out of range.
    TypeError
        If W or D are not PyTorch tensors.
    """
    # ----- input validation -------------------------------------------------
    if not isinstance(W, torch.Tensor) or not isinstance(D, torch.Tensor):
        raise TypeError("W and D must be torch.Tensor instances.")
    if W.dim() != 2 or D.dim() != 2:
        raise ValueError("W and D must be 2‑dimensional tensors.")
    if W.shape[0] != D.shape[0]:
        raise ValueError("First dimension of W and D must match (n).")
    n, p = W.shape
    n2, m = D.shape
    if n != n2:
        raise ValueError("Dimension mismatch between W and D.")
    if not (0 <= k <= m):
        raise ValueError(f"k must satisfy 0 ≤ k ≤ m (got k={k}, m={m}).")

    # ----- trivial cases ----------------------------------------------------
    if k == 0:
        return torch.zeros(m, p, dtype=W.dtype, device=W.device)
    if k == m:
        # keep everything – the solution is simply the orthogonal projection
        return D.t() @ W

    # ----- optimal coefficients in the orthonormal basis --------------------
    # A_j = D.T @ w_j  for all columns simultaneously
    A = D.t() @ W                     # shape (m, p)

    # ----- hard‑threshold each column to exactly k non‑zeros ----------------
    # Find indices of the k largest absolute values per column (dim=0)
    # topk returns (values, indices) of shape (k, p)
    _, topk_idx = torch.topk(torch.abs(A), k, dim=0, largest=True, sorted=False)

    # Prepare output tensor filled with zeros
    S = torch.zeros_like(A)

    # Column indices broadcasted to shape (k, p)
    col_idx = torch.arange(p, device=W.device).unsqueeze(0).expand(k, p)

    # Copy the selected coefficients into S
    S[topk_idx, col_idx] = A[topk_idx, col_idx]

    return S

@torch.no_grad()
def ksvd_opp(
    W: torch.Tensor,                # (d1, d2) weight matrix (columns are signals)
    k: int,                         # K
    n_iter: int = 10,
    s: int = 8,
    D_init: Optional[torch.Tensor] = None,  # (m, K)
    C_init: Optional[torch.Tensor] = None,  # (K, N)
    batch_size: int = 8192,
    ridge: float = 1e-6,
    svd_solver: str = "svd",
    verbose: bool = True,
    mode: str = "default",           # "linear" | "exponential" | "muon" | "default"
    early_stopping_patience: int = 0,  # Zero disables early stopping
    early_stopping_tol: float = 1e-2,  # Relative improvement tolerance
    support_stats: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
    assert svd_solver == "svd", f"[!] Do not support other solvers than SVD! You passed: {svd_solver}"

    # We are already working on whitened weight matrix, L@W of shape d1xd2
    _maybe_enable_fast_kernels()
    device, dtype = W.device, W.dtype
    d1, d2 = W.shape

    # Initialize dictionary
    if D_init is not None:
        D = D_init.to(device=device, dtype=dtype).clone()
        _normalize_columns_(D)
        if D.shape != (d1, k):
            raise ValueError(f"D_init has wrong shape. Expected [{d1}, {k}], recveived [{D.shape[0]}, {D.shape[1]}]")
    else:
        if d2 >= k:
            perm = torch.randint(0, d2, (k,), device=device)
            D = W[:, perm].clone()
        else:
            D = torch.randn((d1, k), device=device, dtype=dtype)
        _normalize_columns_(D)

    # Initialize coefficients for an empty iteration sequence
    C = torch.zeros((k, d2), device=device, dtype=dtype)
    rel_errs: List[float] = []
    abs_errs: List[float] = []
    D_prev = None 
    transposed = None

    # Support diagnostics are accumulated online. Keep only the previous
    # boolean support mask instead of storing one K x N mask per iteration.
    prev_support = None
    if support_stats is not None:
        support_stats.clear()
        support_stats["hamming_fraction"] = []
        support_stats["turnover"] = []
        support_stats["jaccard"] = []
        support_stats["nnz"] = []

    def _record_support(C_current: torch.Tensor) -> None:
        nonlocal prev_support
        if support_stats is None:
            return

        current_support = C_current.ne(0)
        curr_nnz = int(current_support.sum().item())
        support_stats["nnz"].append(curr_nnz)

        if prev_support is not None:
            changed = int(torch.logical_xor(prev_support, current_support).sum().item())
            prev_nnz = int(prev_support.sum().item())
            total = int(current_support.numel())

            support_stats["hamming_fraction"].append(
                changed / float(total) if total > 0 else 0.0
            )

            # For equal-size supports this is the fraction of selected
            # coefficients that were replaced between sparse-coding steps.
            denom = prev_nnz + curr_nnz
            support_stats["turnover"].append(
                changed / float(denom) if denom > 0 else 0.0
            )

            intersection = (prev_nnz + curr_nnz - changed) // 2
            union = prev_nnz + curr_nnz - intersection
            support_stats["jaccard"].append(
                intersection / float(union) if union > 0 else 1.0
            )

        prev_support = current_support


    for it in range(n_iter):
        # Step 1. Fix D and solve sparse coding problem using OMP
        if C_init is not None and it == 0: # in case we performed some rough estimation of coefficients
            C = C_init
        else:
            C = sparse_code_orthonormal(W, D, k=s)

        _record_support(C)

        # Step 2. Fix C and use orthogonal Procrustes Problem to update all at once
        M = W @ C.T # d1 x k
        U, _, Vh = torch.linalg.svd(M, full_matrices=False)
            
        D_new = U @ Vh
        D, D_prev = extrapolate_dictionary(D_new, D_prev, it, n_iter, mode, device)

        # Step 3. Compute the residual for Early Stopping and Error Tracking
        R = W - D @ C  # (d1, d2)
        abs_err = float(torch.linalg.norm(R).item())
        abs_errs.append(abs_err)
        rel_err = abs_err / float(torch.linalg.norm(W).item())
        rel_errs.append(rel_err)
        
        if early_stopping_patience > 0 and it >= early_stopping_patience:
            no_improve = True
            # Check last `early_stopping_patience` intervals (most recent first)
            for j in range(it, it - early_stopping_patience, -1):
                prev_err = rel_errs[j-1]
                curr_err = rel_errs[j]
                # Calculate relative improvement (handle zero division)
                rel_improve = (prev_err - curr_err) / prev_err if prev_err > 0 else 0.0
                
                # If significant improvement found, break early
                if rel_improve >= early_stopping_tol:
                    no_improve = False
                    break
            
            if no_improve:
                if verbose:
                    logging.info(f"Early stopping triggered at iteration {it+1} "
                          f"(patience={early_stopping_patience}, tol={early_stopping_tol:.1e})")
                break

    # Final sparse-code update with the final dictionary. This C is returned,
    # so include its support as the last support transition.
    C = sparse_code_orthonormal(W, D, k=s)
    _record_support(C)
    R = W - D @ C
    final_abs_err = float(torch.linalg.norm(R).item())
    abs_errs.append(final_abs_err)
    rel_errs.append(final_abs_err / float(torch.linalg.norm(W).item()))
    return D, C, rel_errs, abs_errs

def augment_columns_random_mix(x: torch.Tensor, n_mix: int, generator=None) -> torch.Tensor:
    """
    Create `n_mix` random convex combinations of columns of `x`,
    and concatenate them to the original tensor along dim=1.

    Each new column = sum_i (w_i * x[:, i]), where w_i ≥ 0, sum_i w_i = 1.

    Args:
        x: [B, C] 2D tensor
        n_mix: number of new columns to generate
        generator: optional torch.Generator for reproducibility

    Returns:
        [B, C + n_mix] tensor
    """
    assert x.ndim == 2, "Input must be 2D"
    B, C = x.shape

    # Sample random weights from Dirichlet(1,...,1) = uniform on simplex
    # Shape: [n_mix, C]
    weights = torch.rand(n_mix, C, generator=generator, dtype=x.dtype, device=x.device)
    weights = weights / weights.sum(dim=1, keepdim=True)  # normalize to sum to 1

    # Compute new columns: [B, C] @ [C, n_mix] → [B, n_mix]
    new_cols = x @ weights.T

    # Concatenate original and new columns
    return torch.cat([x, new_cols], dim=1)

def ksvd_opp_wrapper(w1, name="self_attn.q_proj", index=0, compression_ratio=0.2, calib_data=None, ks_ratio=2, whitening=True, svd_solver="svd", n_iter=100, d_init_method="", c_init_method="", mode="default", early_stopping_patience = 0, early_stopping_tol = 0, verbose = False, ensure_undercomplete = False, divisibility = 1, device=None, return_support_stats=False):
    # Implementation of k-svd with undercomplete dictionary, using orthogonal procrustes problem to update all dictionary at once
    if not isinstance(index, list):
        index = [index]
    if whitening:
        ss, inv_s = Calib.get_s_inv_s(index, name, "llama3", calib_data)
        ss = ss.to(device=device or "cuda", dtype=torch.float64)
        inv_s = inv_s.to(device=device or "cuda", dtype=torch.float64)
    with torch.no_grad():  # Prevent autograd from building a computation graph
        d1, d2 = w1.shape
        if whitening:
            x = ss @ w1.to(device=device or "cuda", dtype=torch.float64)
        else:
            x = w1.to(device=device or "cuda", dtype=torch.float64)
        k_est, sparsity_est = get_k_and_sparsity(compression_ratio, d1, d2, ks_ratio, divisibility)
        _, ks_ratio_adj = get_sparsity(cr = compression_ratio, d1=d1, d2=d2, k=k_est)
        
        if ensure_undercomplete and k_est > d1:
            k_est = d1
            sparsity_est, ks_ratio_adj = get_sparsity(cr = compression_ratio, d1=d1, d2=d2, k=k_est)
            logging.warning(f"[!] Dictionary [{d1} {k_est}] NOT undercomplete for k/s={ks_ratio}! Recalibrating k/s={ks_ratio_adj:.4f} for K=d1")

        # Initialize the dictionary if necessary
        if isdir(d_init_method):
            logging.info(f"Initialized dict with folder: {d_init_method}")
            full_name_u = f"model.layers.{index[0]}.{name}_u.pth"
            assert isfile(join(d_init_method, full_name_u)), f"Can't find {full_name_u}"
            D_init = torch.load(join(d_init_method, full_name_u), map_location="cuda")
        elif d_init_method == "svd":
            logging.info("Initialized dict with SVD")
            D_init = torch.linalg.svd(x, full_matrices=True)[0] # use only U matrix
            D_init = D_init[:, :k_est]
            if D_init.shape[1] < k_est: # add mixture of columns
                D_init = augment_columns_random_mix(D_init, k_est - D_init.shape[1])
            assert D_init.shape[1] == k_est, f"D_init has shape {D_init.shape}, while k_est={k_est}, cr={compression_ratio}"
        elif d_init_method == "rand_qr":
            logging.info("Initialized dict with random QR-orthogonalized columns")
    
            # Randomly select k_est unique columns from w1 (if possible)
            if k_est <= d2:
                # Use randperm for unique indices when we have enough columns
                perm = torch.randperm(d2, device=x.device)[:k_est]
            else:
                # Fallback: sample with replacement if k_est > d2 (unusual case)
                perm = torch.randint(0, d2, (k_est,), device=x.device)
                logging.warning(f"k_est ({k_est}) > d2 ({d2}), sampling columns with replacement")
            
            # Extract selected columns and clone to avoid in-place modification issues
            D_init = x[:, perm].clone()  # Shape: (d1, k_est)
            
            # Orthogonalize via reduced QR decomposition
            # Q will have orthonormal columns with shape (d1, k_est) when k_est <= d1
            Q, R = torch.linalg.qr(D_init, mode='reduced')
            D_init = Q
            
            # Safety check for shape consistency
            assert D_init.shape == (d1, k_est), \
                f"D_init has shape {D_init.shape}, expected ({d1}, {k_est})"
        else:
            logging.info(f"Initialized dict with None")
            D_init = None

        if isdir(c_init_method):
            logging.info(f"Initialized coeff with folder: {c_init_method}")
            full_name_v = f"model.layers.{index[0]}.{name}_v.pth"
            assert isfile(join(c_init_method, full_name_v)), f"Can't find {full_name_v}"
            C_init = torch.load(join(c_init_method, full_name_v), map_location="cuda")
        else:
            logging.info(f"Initialized coeff with None")
            C_init = None
        
        support_stats = {} if return_support_stats else None
        D, C, rel_errs, abs_errs = ksvd_opp(
                            W=x,
                            k=k_est,
                            s=sparsity_est,
                            n_iter=n_iter,
                            batch_size=16,
                            svd_solver=svd_solver,
                            verbose=verbose,
                            D_init=D_init,
                            C_init=C_init,
                            mode=mode,
                            early_stopping_patience=early_stopping_patience,
                            early_stopping_tol = early_stopping_tol,
                            support_stats=support_stats,
                            )
        if verbose:
            iterations, rel_errs_strided = subsample_equal_step_numpy(rel_errs)
            headers = ["Iterations"] + [str(i) for i in iterations]
            data = [["Rel. error"] + rel_errs_strided]
            table = tabulate(data, headers=headers, tablefmt="pipe")
        
            log_message = f"""
            ------------------------------------------------------------------------------------------------
            Layer: {index} Proj: {name} CR: {compression_ratio}
            \t[kSVD-OPP] Acceleration mode: {mode}
            \t[kSVD-OPP] Total max iterations: {n_iter}
            \t[kSVD-OPP] Set K/S ratio: {ks_ratio:.4f} Actual K/S ratio: {ks_ratio_adj:.4f}
            \t[kSVD-OPP] Number of iterations {len(rel_errs)}
            \t[kSVD-OPP] Early Stop? {len(rel_errs) < n_iter}
            \t[kSVD-OPP] Early stopping patience: {early_stopping_patience}
            \t[kSVD-OPP] Early stopping tol: {early_stopping_tol}
            \t[kSVD-OPP] W: {x.shape} D: {D.shape} C: {C.shape} Undercomplete: {D.shape[1] <= D.shape[0]}
            \n{table}
            """
            logging.info(log_message)
            
        if whitening:
            D = inv_s @ D 
            del ss, inv_s
    gc.collect()
    torch.cuda.empty_cache()  # Try to free CUDA memory immediately
    if not torch.isnan(D).any() and not torch.isnan(C).any():
        if return_support_stats:
            return D, C, rel_errs, abs_errs, support_stats
        return D, C, rel_errs, abs_errs
    raise ValueError("D or C contain Nan values!")
