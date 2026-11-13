import pickle
import torch
from paperflow.impl.svdllm import svdllm, rank_from_compression_ratio


def test_svdllm_matches_truncated_svd_in_calibration_space(tmp_path):
    torch.manual_seed(19)
    weight = torch.randn(12, 9, dtype=torch.float64)
    activation = torch.randn(40, 12, dtype=torch.float64)
    gram = activation.T @ activation
    directory = tmp_path / 'self_attn.k_proj'
    directory.mkdir()
    with (directory / '0.pkl').open('wb') as stream:
        pickle.dump(gram, stream)
    u, v = svdllm(weight, name='self_attn.q_proj', calib_data=tmp_path, device='cpu')
    scaling = torch.linalg.cholesky(gram).T
    left, sigma, right = torch.linalg.svd(scaling @ weight, full_matrices=False)
    rank = rank_from_compression_ratio(weight.shape, .2)
    expected = (left[:, :rank] * sigma[:rank]) @ right[:rank]
    torch.testing.assert_close(scaling @ u @ v, expected, rtol=1e-12, atol=1e-12)
    assert u.shape == (12, rank) and v.shape == (rank, 9)
    torch.testing.assert_close(v @ v.T, torch.diag(sigma[:rank]), rtol=1e-12, atol=1e-12)


def test_svdllm_zero_reduction_preserves_weight():
    weight = torch.randn(8, 6)
    u, v = svdllm(weight, compression_ratio=0)
    torch.testing.assert_close(u @ v, weight, rtol=0, atol=0)
