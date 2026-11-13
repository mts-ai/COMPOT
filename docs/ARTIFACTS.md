# Calibration and checkpoints

Download the [calibration data](https://disk.yandex.ru/d/3RwESGstkyY9lA) and
[checkpoints](https://disk.yandex.ru/d/04MqoMUthjrw5Q), or generate them yourself.
Extract the calibration archive for your model before using it.

## Generate calibration and a checkpoint

```bash
uv run --locked --no-sync compot-run experiments=qwen3_06b_compot_cr02 \
  stages.collect=true calib_root="$PWD/calibration/Qwen3-0.6B" \
  checkpoint_root="$PWD/checkpoints"
```

This collects calibration, compresses the model and evaluates it. Existing
complete calibration and matching checkpoints are reused automatically.
Use `stages.evaluate=false` to save a checkpoint without evaluation.

SVD-LLM uses `experiments=qwen3_06b_svdllm_cr02`; CoSpaDi uses
`experiments=qwen3_06b_cospadi_cr02`. All methods save deployable Transformers models.

## Evaluate a checkpoint

```bash
uv run --locked --no-sync compot-run experiments=qwen3_06b_compot_cr02 \
  checkpoint_path=/path/to/downloaded/checkpoint
```

The command detects the complete checkpoint, skips calibration and compression,
and goes directly to evaluation. Calibration data is not needed for this step.
For a model generated locally, use the `model_path` printed by the pipeline
(the `transformers/` directory). Keep all downloaded model files together.
