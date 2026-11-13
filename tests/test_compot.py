import torch
from paperflow.impl.compot import ksvd_opp, sparse_code_orthonormal

def test_compot_orthogonal_dictionary_and_sparse_codes():
    torch.manual_seed(42)
    weight=torch.randn(12,16,dtype=torch.float64)
    initial=torch.linalg.svd(weight,full_matrices=False)[0][:,:6]
    d,c,relative,absolute=ksvd_opp(weight,6,n_iter=3,s=2,D_init=initial,mode='default',verbose=False)
    torch.testing.assert_close(d.T@d,torch.eye(6,dtype=d.dtype),atol=1e-10,rtol=1e-10)
    assert torch.all((c!=0).sum(0)<=2)
    assert torch.isfinite(d@c).all()
    torch.testing.assert_close(c,sparse_code_orthonormal(weight,d,k=2))
