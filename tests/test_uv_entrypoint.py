"""The documented uv console command must resolve the checkout's Hydra config."""
import subprocess
import sysconfig
from pathlib import Path


def test_console_entrypoint_loads_config():
    command = Path(sysconfig.get_path('scripts')) / 'compot-run'
    result = subprocess.run([str(command), '--cfg', 'job'], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'Qwen/Qwen3-0.6B' in result.stdout
    assert 'sampling: reference' in result.stdout
