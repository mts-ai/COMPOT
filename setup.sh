#!/usr/bin/env bash
# Original paper environment: nvcr.io/nvidia/pytorch:25.04-py3
# Cluster image notation: nvcr.io#nvidia/pytorch:25.04-py3
# Source this script to retain HF_ALLOW_CODE_EVAL in the calling shell.
unset PIP_CONSTRAINT
pip uninstall -y dill
pip install --upgrade dill==0.3.8
pip install transformers==4.55.4 datasets==3.6.0 accelerate==1.7.0 lm_eval==0.4.8.0
pip install timm
pip install 'clearml[s3]'
pip install onnxsim
pip install onnxruntime
pip install omegaconf
pip install tabulate==0.9.0
export HF_ALLOW_CODE_EVAL="1"
