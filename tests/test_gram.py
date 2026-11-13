import pickle
import pytest
torch=pytest.importorskip('torch')
from paperflow.statistics import GramAccumulator,sample_token_blocks
from paperflow.statistics import ActivationGramCollector, CANONICAL_PROJECTIONS
from types import SimpleNamespace
import json
def test_gram_matches_explicit_xtx(tmp_path):
    x=torch.tensor([[[1.,2.],[3.,4.]]]);acc=GramAccumulator();acc.add(x)
    acc.add(x)
    torch.testing.assert_close(acc.value,2*(x.reshape(-1,2).T@x.reshape(-1,2)))
    assert acc.tokens==4
    p=tmp_path/'m.pkl';acc.save(p)
    with p.open('rb') as f:torch.testing.assert_close(pickle.load(f),acc.value)
def test_selection_deterministic():
    data=[[i]*12 for i in range(6)]
    assert sample_token_blocks(data,length=4,samples=3,seed=123)==sample_token_blocks(
        data,length=4,samples=3,seed=123)
def test_selection_errors():
    with pytest.raises(ValueError):sample_token_blocks([[1,2]],length=4,samples=1,seed=42)


def test_downloaded_calibration_is_reused_only_when_complete(tmp_path):
    rows = []
    for module in CANONICAL_PROJECTIONS:
        directory = tmp_path/module
        directory.mkdir()
        for layer in range(2):
            path = directory/f'{layer}.pkl'
            path.write_bytes(b'calibration')
            rows.append(dict(file=str(path.relative_to(tmp_path)), bytes=path.stat().st_size))
    (tmp_path/'CALIBRATION_MANIFEST.json').write_text(json.dumps(dict(model='test/model', files=rows)))
    ctx = dict(calib_root=tmp_path, cfg=SimpleNamespace(model=SimpleNamespace(name='test/model')))
    collector = ActivationGramCollector()
    assert collector.is_complete(ctx)
    (tmp_path/CANONICAL_PROJECTIONS[0]/'1.pkl').unlink()
    assert not collector.is_complete(ctx)
