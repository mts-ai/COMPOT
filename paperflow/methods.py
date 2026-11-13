"""Compression method configuration."""
from dataclasses import dataclass, field

@dataclass(frozen=True)
class MethodSpec:
    name: str
    iterations: int
    power_iterations: int = 8
    ks_ratio: float = 2.0
    extra_args: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.name not in ('ksvd', 'compot', 'svdllm'):
            raise ValueError('Choose compot, ksvd (CoSpaDi), or svdllm.')
        if self.iterations <= 0 or self.power_iterations <= 0 or self.ks_ratio <= 0:
            raise ValueError('iterations, power_iterations and ks_ratio must be positive')
        if self.name == 'svdllm' and self.ks_ratio != 1.0:
            raise ValueError('SVD-LLM uses dense factors and requires ks_ratio=1.')
