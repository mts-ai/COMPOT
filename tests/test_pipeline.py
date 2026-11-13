from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import pytest
from omegaconf import OmegaConf
from paperflow.pipeline import Pipeline
from paperflow.methods import MethodSpec
from paperflow.artifacts import checkpoint_complete
from paperflow.compression import CoSpaDiBackend
import json
import torch
from safetensors.torch import save_file
class Worker:
    def __init__(self,result):self.calls=0;self.result=result
    def run(self,ctx):self.calls+=1;return dict(self.result)
class Reporter:
    def __init__(self):self.calls=0
    def report(self,ctx,payload):self.calls+=1
def config(tmp,force=False):
    return OmegaConf.create({'project':'unit','repo_root':str(tmp),'run_dir':str(tmp/'run'),
      'checkpoint_root':str(tmp/'ckpt'),'calib_root':str(tmp/'calib'),
      'checkpoint_path':None,'dry_run':False,'force':force,
      'stages':{'collect':True,'compress':False,'evaluate':True,'report':True},
      'components':{'statistics':{'x':1},'compression':{'x':2},'evaluation':{'x':3}}})
def test_pipeline_caches_completed_stages(tmp_path):
    a=Worker({'calib_root':str(tmp_path/'calib')});b=Worker({});c=Worker({'quality':0.7});r=Reporter()
    p=Pipeline(a,b,c,r,MethodSpec('ksvd',60))
    p.run(config(tmp_path))
    (tmp_path/'calib').mkdir(exist_ok=True)
    (tmp_path/'calib/calibration_manifest.json').write_text('{}')
    p.run(config(tmp_path))
    assert (a.calls,b.calls,c.calls,r.calls)==(1,0,1,2)
    p.run(config(tmp_path,force=True))
    assert a.calls==2 and c.calls==2
def test_dryrun_no_files(tmp_path):
    a=Worker({});p=Pipeline(a,Worker({}),Worker({}),Reporter(),
                            MethodSpec('ksvd',60))
    cfg=config(tmp_path);cfg.dry_run=True
    p.run(cfg)
    assert not (tmp_path/'run').exists()


def saved_checkpoint(path):
    path.mkdir(parents=True)
    (path/'config.json').write_text('{"model_type":"llama"}')
    save_file({'weight': torch.ones(1)}, path/'model.safetensors')
    return path


def test_checkpoint_path_evaluates_without_calibration_or_compression(tmp_path):
    checkpoint = saved_checkpoint(tmp_path/'downloaded')
    a, b, c = Worker({}), Worker({}), Worker({'quality': .7})
    cfg = config(tmp_path)
    cfg.stages.compress = True
    cfg.checkpoint_path = str(checkpoint)
    result = Pipeline(a, b, c, Reporter(), MethodSpec('compot', 20)).run(cfg)
    assert (a.calls, b.calls, c.calls) == (0, 0, 1)
    assert result['stages']['collect']['status'] == 'reused'
    assert result['model_path'] == str(checkpoint)


def test_matching_compression_output_is_detected_without_receipts(tmp_path):
    cfg = config(tmp_path)
    cfg.model = {'name': 'Qwen/Qwen3-0.6B'}
    cfg.compression_ratio = .2
    cfg.stages.compress = True
    method = MethodSpec('svdllm', 1, ks_ratio=1)
    backend = CoSpaDiBackend()
    ctx = dict(cfg=cfg, method=method, checkpoint_root=tmp_path/'ckpt', calib_root=tmp_path/'calib')
    checkpoint = saved_checkpoint(backend.checkpoint_path(ctx))
    (checkpoint/'compression_recipe.json').write_text(json.dumps(backend.signature(ctx)))
    collect, evaluate = Worker({}), Worker({'quality': .7})
    result = Pipeline(collect, backend, evaluate, Reporter(), method).run(cfg)
    assert collect.calls == 0 and evaluate.calls == 1
    assert result['stages']['compress']['status'] == 'reused'
    assert not (tmp_path/'calib').exists()
    ctx['calib_root'] = tmp_path/'different_calibration'
    assert backend.reusable_checkpoint(ctx) is None


def test_incomplete_checkpoint_is_not_accepted(tmp_path):
    checkpoint = saved_checkpoint(tmp_path/'incomplete')
    (checkpoint/'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'missing.safetensors'}}))
    assert not checkpoint_complete(checkpoint)
    cfg = config(tmp_path)
    cfg.checkpoint_path = str(checkpoint)
    with pytest.raises(FileNotFoundError, match='Incomplete checkpoint'):
        Pipeline(Worker({}), Worker({}), Worker({}), Reporter(), MethodSpec('compot', 20)).run(cfg)
    (checkpoint/'model.safetensors.index.json').unlink()
    (checkpoint/'model.safetensors').write_bytes(b'incomplete tensor data')
    assert not checkpoint_complete(checkpoint)
