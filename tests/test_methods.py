import pytest
from paperflow.methods import MethodSpec


def test_ksvd_is_supported():
    assert MethodSpec('ksvd', 60).name == 'ksvd'


def test_svdllm_requires_dense_factor_ratio():
    assert MethodSpec('svdllm', 1, ks_ratio=1).name == 'svdllm'
    with pytest.raises(ValueError, match='requires ks_ratio=1'):
        MethodSpec('svdllm', 1)


@pytest.mark.parametrize('method', ('ksvd_opp', 'ksvd_opp_weighted', 'svd', 'invalid'))
def test_unknown_methods_are_rejected(method):
    with pytest.raises(ValueError):
        MethodSpec(method, 60)


@pytest.mark.parametrize('iterations,power,ratio', [(0, 8, 2), (60, 0, 2), (60, 8, 0)])
def test_invalid_solver_parameters(iterations, power, ratio):
    with pytest.raises(ValueError):
        MethodSpec('ksvd', iterations, power_iterations=power, ks_ratio=ratio)
