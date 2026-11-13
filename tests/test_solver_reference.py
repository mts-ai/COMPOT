import sys
from pathlib import Path
import pytest
torch=pytest.importorskip('torch')
ROOT=Path(__file__).resolve().parents[1]

def test_ksvd_shape_sparsity_finite():
    from paperflow.impl.ksvd import ksvd,batch_threshold_code
    torch.manual_seed(111)
    w=torch.randn((7,11),dtype=torch.float64)
    d,c,errs=ksvd(w,5,sparsity=2,n_iter=2,svd_solver='svd',verbose=False)
    assert d.shape==(7,5) and c.shape==(5,11)
    assert (c!=0).sum(0).max()<=2
    assert len(errs)==2
    assert torch.isfinite(d).all() and torch.isfinite(c).all()
    assert torch.isfinite(torch.linalg.norm(w-d@c))
    test=batch_threshold_code(w,d,s=2)
    assert test.shape==c.shape

