# Reproducing COMPOT

## Environments

The paper experiments used `nvcr.io/nvidia/pytorch:25.04-py3`
(`nvcr.io#nvidia/pytorch:25.04-py3` in cluster notation). Run `source setup.sh`
inside the container to install the experiment dependencies.

For a standalone installation, use `uv sync --locked --extra eval --extra test`.
The lock selects Python 3.13, PyTorch 2.9.1+cu128, Transformers 4.55.4,
datasets 3.6.0, Accelerate 1.7.0 and lm-eval 0.4.8.

## Calibration

LLM calibration uses FineWeb, 256 sequences of length 1,024, seed 2023. The
provided archives contain the precomputed activation statistics and diagnostic
summaries used by the release, rather than raw FineWeb text. Each archive has
a model-root folder and a manifest. Extract it and pass that folder
as `calib_root`; see [ARTIFACTS.md](ARTIFACTS.md).

To collect new statistics using the configured FineWeb source:

```bash
uv run --locked --no-sync compot-run experiments=qwen3_06b_compot_cr02 \
  stages.collect=true stages.compress=false stages.evaluate=false \
  calib_root="$PWD/calibration/Qwen3-0.6B"
```

This may download a substantial dataset. Access to the base model and dataset
is required. The collector profiles K/O attention projections and up/down MLP
projections, reusing the matching input statistics for projections with shared
inputs. Do not substitute calibration statistics from another model.

## Allocation and compression

All twelve `*_compot_cr0{2,3,4}` recipes use the original `all_mincr` allocator:
unwhitened spectra, Frobenius normalization, minimum per-projection reduction
0.05 and maximum 0.8. The solver uses activation whitening, SVD initialization,
k/s = 2, 20 alternating iterations and seed 42. The
orthogonality condition belongs to the whitened optimization space.

The compressed core covers Q/K/V/O and up/gate/down projections in every
decoder layer. `allocation.json`
records allocation provenance; the outer checkpoint directory retains raw
U/V factors for resuming. `transformers/` is the deployable output, and
`results.json` records the actual path.

```bash
uv run --locked --no-sync compot-run experiments=llama32_1b_compot_cr03 \
  calib_root=/path/to/Llama-3.2-1B checkpoint_root=checkpoints \
  run_dir=runs/llama32_1b_cr03 stages.evaluate=false
```

Use a fresh checkpoint root when changing solver settings or calibration data. Fingerprints guard
resumption, but keeping distinct output directories makes provenance clear.
`method.extra_args.allocation=uniform` selects the static ablation.

## SVD-LLM and CoSpaDi

Use `experiments=qwen3_06b_svdllm_cr02` for SVD-LLM. It applies truncated
SVD to the calibration-whitened weight and splits the square root of the
singular values between U and V. Its dense factors use k/s = 1; this is a
separate solver, not COMPOT with a different sparsity setting. The shared
modeling files use serialized ranks and shapes for all three methods.

CoSpaDi uses `experiments=qwen3_06b_cospadi_cr02`, with K-SVD optimization and sparse
coefficients. Both methods have recipes for all four models at the three CRs.
All recipes export deployable models and evaluate them directly with lm-eval.
The sparse rank budget includes mask storage, and dense SVD factors are stored
without masks.

## Deployable format and MACKO

`packed_mask_v1` stores dense BF16 dictionaries, BF16 nonzero coefficient
values and uint8 arrays containing packed one-bit masks. Dense SVD-LLM
coefficients use `dense_uv_v1` with ordinary BF16 U/V tensors, inside the
`factorized_v1` checkpoint format. Both formats use the same modeling files. Sparse values are
buffers, so parameter count alone understates persistent storage. The exporter
checks actual `state_dict` tensor bytes against the original BF16 core.

Core reduction is `1 - compressed_core_bytes / original_core_bytes`. The core
includes every decoder-layer tensor, including norms, biases and buffers;
embeddings and `lm_head` are excluded. Whole-checkpoint payload is reported
separately and is less compressed because these large tensors remain dense.

For packed sparse coefficients, the portable forward materializes a temporary
coefficient matrix for one projection at a time. Dense factors use two linear
operations directly. It has no persistent dense coefficient cache or fused
projection copy. Its peak inference memory is not the same as persistent
checkpoint storage. MACKO `Sparse*MLP`, asynchronous routines
and `optimize()` provide an optional path for compatible
factorized up/gate and dense down MLPs; a fully factorized checkpoint does not
automatically enable it. No portable-forward speedup is claimed.

For a legacy checkpoint containing separate HF U/V tensors, or saved raw U/V
files alongside a supporting HF checkpoint:

```bash
uv run --locked --no-sync python -m paperflow.deploy \
  --source /path/to/supporting_hf_checkpoint \
  --factors /path/to/original_raw_factors \
  --output /path/to/new_deployable --target-cr 0.2
```

Omit `--factors` when the source already contains separate U/V tensors.
Fused-only weights cannot recover the original factors and are rejected.
Existing output directories are not overwritten.

## Evaluation

Evaluate the full task set at zero shots, with no sample limit:

```bash
export CHECKPOINT=/path/to/deployable_model
uv run --locked --no-sync python -m lm_eval --model hf \
  --model_args pretrained="$CHECKPOINT",dtype=bfloat16,trust_remote_code=True \
  --tasks mmlu,arc_easy,arc_challenge,hellaswag,piqa,lambada_openai,race,wikitext,sciq \
  --batch_size auto:1 --device cuda:0 --num_fewshot 0 --output_path runs/eval
```

For two GPUs, data parallel evaluation assigns one complete model to each GPU
and distributes evaluation requests across ranks:

```bash
uv run --locked --no-sync python -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 --num_machines 1 --mixed_precision no \
  --dynamo_backend no -m lm_eval --model hf \
  --model_args pretrained="$CHECKPOINT",dtype=bfloat16,trust_remote_code=True \
  --tasks mmlu,arc_easy,arc_challenge,hellaswag,piqa,lambada_openai,race,wikitext,sciq \
  --batch_size auto:1 --num_fewshot 0 --output_path runs/eval_2gpu
```

Run one distributed launch per evaluation, not two independent full evaluations.
Do not set model-parallel `parallelize=True` for this data-parallel command.
Increase `--num_processes` to four only with four available GPUs.

Average accuracy is the unweighted mean of MMLU, ARC-easy, ARC-challenge,
HellaSwag, PIQA, LAMBADA, RACE and SciQ. Use `acc_norm` when present and `acc`
otherwise. WikiText `word_perplexity` and LAMBADA `perplexity` are separate
metrics with different tokenization/aggregation protocols. They are not
interchangeable with sliding-window perplexities from another script.

Expected full evaluation counts are PIQA 1,838; HellaSwag 10,042; LAMBADA 5,153;
ARC-easy 2,376; ARC-challenge 1,172; SciQ 1,000; RACE 1,045; WikiText 62; and
MMLU 14,042 across subjects. Retain the harness result JSON, task configuration,
metric standard errors and sample counts. Partial logs are not final results.

## Questions

For questions about evaluation metrics, please open a
[GitHub issue](https://github.com/mts-ai/COMPOT/issues) with your experiment
configuration, dependency versions and lm-eval result JSON.
