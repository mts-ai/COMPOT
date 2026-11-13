from pathlib import Path
import ast
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]

def test_release_has_only_paper_sources():
    for dirname in ('legacy', 'examples', 'repro', 'configs/allocation'):
        assert not (ROOT / dirname).exists()
    shell_scripts = [p for p in ROOT.glob("*.sh") if p.name != "setup.sh"]

    for source_dir in ("configs", "paperflow", "tests", ".github"):
        shell_scripts.extend((ROOT / source_dir).rglob("*.sh"))

    assert not shell_scripts, f"Unexpected release scripts: {shell_scripts}"
    assert not (ROOT/'paperflow/allocation.py').exists()
    assert 'dynamic:' not in '\n'.join(x.read_text() for x in (ROOT/'configs').rglob('*.yaml'))
    assert (ROOT/'paperflow/impl/ksvd.py').exists()
    assert (ROOT/'paperflow/impl/evaluate.py').exists()

def test_experiment_names_and_model_roots():
    experiments = ROOT / 'configs' / 'experiments'
    names = sorted(x.stem for x in experiments.glob('*.yaml'))
    assert len(names) == 36
    assert 'qwen3_06b_compot_cr02' in names
    for model in ('qwen3_06b', 'qwen3_8b', 'llama32_1b', 'llama3_8b'):
        for cr in ('cr02','cr03','cr04'):
            assert f'{model}_cospadi_{cr}' in names
            compot = OmegaConf.load(experiments / f'{model}_compot_{cr}.yaml')
            assert compot.compression_ratio == int(cr[-1]) / 10
            assert compot.project == 'COMPOT'
            assert {'override /method': 'compot'} in compot.defaults
            doc = OmegaConf.load(experiments / f'{model}_cospadi_{cr}.yaml')
            assert doc.stages.evaluate
            assert doc.stages.compress
            assert not doc.stages.collect

def test_exact_solver_numeric_core_functions_retained():
    # Verify API and no use of orthogonal-Procrustes/alternative solvers.
    impl = ast.parse((ROOT/'paperflow/impl/ksvd.py').read_text())
    funcs = {x.name for x in impl.body if isinstance(x, ast.FunctionDef)}
    assert {'ksvd','batch_threshold_code','_top_rank1'} <= funcs

@pytest.mark.parametrize('n', (1, 2, 7))
def test_bf16_truncation_matches_bitwise_reference(n):
    torch = pytest.importorskip('torch')
    from paperflow.impl.evaluate import bf16_truncate_frac_bits
    x = torch.tensor([1.0, 1.234, -0.1, 0.0, float('inf'), float('nan')], dtype=torch.bfloat16)
    y = bf16_truncate_frac_bits(x, n)
    bits = x.view(torch.int16).to(torch.int32) & 0xffff
    exp = (bits >> 7) & 0xff
    expected_bits = torch.where(exp == 255, bits, bits & (~((1 << n) - 1) & 0xffff))
    assert torch.equal(y.view(torch.int16), expected_bits.to(torch.int16))

def test_eval_kernel_preserves_nan_inf():
    torch = pytest.importorskip('torch')
    from paperflow.impl.evaluate import bf16_truncate_frac_bits
    x = torch.tensor([float('inf'), float('-inf'), float('nan')], dtype=torch.bfloat16)
    y = bf16_truncate_frac_bits(x, 2)
    assert torch.equal(x.view(torch.int16), y.view(torch.int16))
