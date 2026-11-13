"""Hydra command-line entry point for calibration, compression and evaluation."""
from pathlib import Path
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

_CONFIG_ROOT = Path(__file__).resolve().parents[1] / 'configs'
if not _CONFIG_ROOT.is_dir():
    _CONFIG_ROOT = Path(__file__).resolve().parent / 'configs'


@hydra.main(version_base=None, config_path=str(_CONFIG_ROOT), config_name='config')
def main(cfg: DictConfig):
    runner=instantiate(cfg.pipeline)
    result=runner.run(cfg)
    print('PIPELINE_RESULT:',result['model_path'] or '(no checkpoint)')

if __name__=='__main__':main()
