import torch
import pytest
from paperflow.impl.allocation import allocate_counts_with_dense_loop
from paperflow.impl.allocation_cache import load_or_allocate
from paperflow.methods import MethodSpec
from paperflow.compression import CoSpaDiBackend

def test_allocator_preserves_parameter_budget_and_constraints():
    spectra=[torch.arange(32,0,-1,dtype=torch.float64),torch.arange(16,0,-1,dtype=torch.float64)]
    result=allocate_counts_with_dense_loop(svals_list=spectra,shapes=[(32,64),(16,32)],
        target_cr=.5,min_rank_per_matrix=[4,2],max_rank_per_matrix=[20,10],verbose=False)
    assert abs(result['achieved_cr']-.5)<.06
    for length,count,low,high in zip([32,16],result['counts_per_matrix'],[4,2],[20,10]):
        assert low <= length-count <= high

def test_compot_and_uniform_factors_have_distinct_paths(fake_context):
    ctx=fake_context
    ctx['method']=MethodSpec('compot',20,extra_args={'allocation':'original'})
    original=CoSpaDiBackend().checkpoint_path(ctx)
    assert '--allocation' in CoSpaDiBackend().build_command(ctx)
    ctx['method']=MethodSpec('compot',20)
    assert original!=CoSpaDiBackend().checkpoint_path(ctx)

def test_compot_recipe_uses_direct_full_lm_eval():
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    root=Path(__file__).resolve().parents[1]
    with initialize_config_dir(version_base=None,config_dir=str(root/'configs')):
        cfg=compose(config_name='config',overrides=['experiments=qwen3_06b_compot_cr02'])
        evaluation=instantiate(cfg.evaluation)
        command=evaluation.build_command({'model_path':'/fused/checkpoint','run_dir':Path('/run')})
    assert command[1:3]==['-m','lm_eval']
    assert '--quant_coeffs' not in command and '--limit' not in command
    assert command[command.index('--batch_size')+1]=='auto:1'
