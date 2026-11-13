import pytest
from paperflow.compression import CoSpaDiBackend
from paperflow.evaluation import CoSpaDiEvaluation, LMEvalHarness
from paperflow.methods import MethodSpec


def test_old_method_commands(fake_context):
    args = CoSpaDiBackend().build_command(fake_context)
    assert args[args.index('--ksvd-iterations') + 1] == '60'
    assert args[args.index('--power-iterations') + 1] == '8'
    assert '--dynamic' not in args
    assert '--method' in args and args[args.index('--method') + 1] == 'ksvd'
    assert '--model-name' in args
    assert args[1:4] == ['-m', 'paperflow.impl.compress', '--method']


def test_unsupported_method_rejected(fake_context):
    class InvalidMethod:
        name = 'ksvd_opp'
    fake_context['method'] = InvalidMethod()
    with pytest.raises(ValueError, match='only supports ksvd'):
        CoSpaDiBackend().build_command(fake_context)


def test_coefficient_quantization_eval(fake_context):
    result = CoSpaDiEvaluation(quant_coeffs=14).run(fake_context)
    assert result['command'][-2:] == ['--quant_coeffs', '14']
    assert result['results_root'] == str(fake_context['run_dir']/'evaluation')


def test_lm_eval_dedicated_output(fake_context):
    cmd = LMEvalHarness(tasks=['wikitext']).build_command(fake_context)
    assert 'wikitext' in cmd and '--output_path' in cmd


def test_lm_eval_supports_named_custom_tasks(fake_context):
    cmd = LMEvalHarness(tasks=['local_smoke'], include_path='/custom/tasks').build_command(fake_context)
    assert cmd[cmd.index('--include_path') + 1] == '/custom/tasks'


def test_evaluation_requires_saved_results(fake_context, monkeypatch):
    monkeypatch.setattr('paperflow.evaluation.run_checked', lambda *args, **kwargs: None)
    fake_context['dry_run'] = False
    fake_context['model_path'] = '/checkpoint'
    with pytest.raises(RuntimeError, match='did not write readable results'):
        LMEvalHarness(tasks=['piqa']).run(fake_context)


def test_evaluation_does_not_accept_stale_results(fake_context, monkeypatch):
    import json
    output = fake_context['run_dir']/'eval'
    output.mkdir(parents=True)
    (output/'results_old.json').write_text(json.dumps({'results': {'piqa': {'acc,none': .1}}}))
    monkeypatch.setattr('paperflow.evaluation.run_checked', lambda *args, **kwargs: None)
    fake_context.update(dry_run=False, model_path='/different-checkpoint')
    with pytest.raises(RuntimeError, match='for this run'):
        LMEvalHarness(tasks=['piqa']).run(fake_context)


def test_evaluation_selects_its_own_results(fake_context, monkeypatch):
    import json
    from paperflow.evaluation import collect_eval_metrics
    output = fake_context['run_dir']/'eval'
    output.mkdir(parents=True)
    (output/'results_old.json').write_text(json.dumps({'results': {'piqa': {'acc,none': .1}}}))
    def evaluate(*args, **kwargs):
        (output/'results_new.json').write_text(json.dumps({'results': {'piqa': {'acc,none': .7}}}))
    monkeypatch.setattr('paperflow.evaluation.run_checked', evaluate)
    fake_context.update(dry_run=False, model_path='/different-checkpoint')
    result = LMEvalHarness(tasks=['piqa']).run(fake_context)
    assert collect_eval_metrics(result['results_file']) == {'piqa/acc,none': .7}


def test_local_model_names_are_short_and_unambiguous(tmp_path):
    from paperflow.artifacts import model_identifier
    first = model_identifier(tmp_path/'first'/'base')
    second = model_identifier(tmp_path/'second'/'base')
    assert first != second and len(first) < 80
    assert model_identifier('Qwen/Qwen3-0.6B') == 'Qwen_Qwen3-0.6B'


def test_checkpoint_path_is_specific_to_model_and_settings(fake_context):
    backend = CoSpaDiBackend()
    path = backend.checkpoint_path(fake_context)
    assert path.name == 'transformers'
    assert path.parent.name == 'Qwen_Qwen3-0.6B_cr_0.2_ksvd_ks_2.0_ksvd_it_60_power_it_8_alloc_uniform_format_deployable_v1'
    assert path.parent.parent == fake_context['checkpoint_root']


def test_missing_calibration_fails_early(fake_context):
    fake_context['dry_run'] = False
    with pytest.raises(FileNotFoundError, match='Calibration directory'):
        CoSpaDiBackend().run(fake_context)


def test_invalid_compression_ratio_rejected(fake_context):
    fake_context['cfg'].compression_ratio = 1.1
    with pytest.raises(ValueError, match='compression_ratio'):
        CoSpaDiBackend().run(fake_context)
