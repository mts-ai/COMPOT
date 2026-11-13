import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config
from paperflow.deploy import convert_checkpoint
from paperflow.modelings.compot_sparse import PackedSparseLinear, pack_weight


def test_sparse_storage_roundtrip_forward_and_no_dense_cache():
    weight = torch.tensor([[1, 0, -2, 0, 3], [0, 0, 4, 0, 0]], dtype=torch.bfloat16)
    mask, values = pack_weight(weight)
    layer = PackedSparseLinear(5, 2, values.numel())
    layer.packed_mask.copy_(mask)
    layer.values.copy_(values)
    torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)
    x = torch.randn(3, 5, dtype=torch.bfloat16)
    torch.testing.assert_close(layer(x), torch.nn.functional.linear(x, weight), rtol=0, atol=0)
    assert set(layer.state_dict()) == {"packed_mask", "values"}
    assert mask.numel() == 2


@pytest.mark.parametrize("family", ["llama", "qwen3"])
@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("dense_factors", [False, True])
@pytest.mark.parametrize("bias", [False, True])
def test_transformers_deploy_strict_load_generate_and_resave(tmp_path, family, tied, dense_factors, bias):
    torch.manual_seed(42)
    cls = LlamaConfig if family == "llama" else Qwen3Config
    cfg = cls(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        tie_word_embeddings=tied,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attention_bias=bias,
        mlp_bias=bias,
    )
    native = (
        AutoModelForCausalLM.from_config(cfg, attn_implementation="eager").to(torch.bfloat16).eval()
    )
    state = {k: v.contiguous() for k, v in native.state_dict().items()}
    name = "model.layers.0.self_attn.q_proj"
    u = torch.randn(4, 16, dtype=torch.bfloat16) / 4
    v = torch.randn(16, 4, dtype=torch.bfloat16) / 4
    if not dense_factors:
        v[:, ::2] = 0
    state[name + "_u.weight"] = u
    state[name + "_v.weight"] = v
    state[name + ".weight"] = (v @ u).contiguous()
    source = tmp_path / "source"
    source.mkdir()
    cfg.save_pretrained(source)
    output = tmp_path / f"{family}_cr_0.2"
    if tied:
        torch.save(u.T.double(), source / (name + "_u.pth"))
        torch.save(v.T.double(), source / (name + "_v.pth"))
        del state[name + "_u.weight"], state[name + "_v.weight"]
        report = convert_checkpoint(source, output, target_cr=0.0, state_dict=state)
    else:
        save_file(state, source / "model.safetensors", metadata={"format": "pt"})
        report = convert_checkpoint(source, output, target_cr=0.0)
    assert report["packed_core_bytes"] < report["dense_core_bytes"]
    expected = "dense_uv_v1" if dense_factors else "packed_mask_v1"
    assert next(row for row in report['projections'] if row['path'] == name)['storage'] == expected
    model, info = AutoModelForCausalLM.from_pretrained(
        output,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert (
        info["missing_keys"] == []
        and info["unexpected_keys"] == []
        and info["mismatched_keys"] == []
    )
    assert not hasattr(model.model.layers[0].self_attn, "q_proj")
    core_bytes = sum(t.numel() * t.element_size() for key, t in model.state_dict().items() if key.startswith('model.layers.'))
    assert core_bytes == report['packed_core_bytes']
    x = torch.tensor([[1, 4, 8, 9]])
    with torch.inference_mode():
        logits = model(x).logits
        assert torch.isfinite(logits).all()
        assert model.generate(x, max_new_tokens=2, do_sample=False).shape[1] >= 5
    # Validate the loaded U/V route against the exact uncompressed U/V route.
    attention = model.model.layers[0].self_attn
    torch.testing.assert_close(attention.q_proj_u.weight, u, rtol=0, atol=0)
    torch.testing.assert_close(attention.q_proj_v.weight, v, rtol=0, atol=0)
    saved = tmp_path / "resaved"
    model.save_pretrained(saved)
    loaded, loading = AutoModelForCausalLM.from_pretrained(
        saved,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        output_loading_info=True,
    )
    assert not loading["missing_keys"] and not loading["unexpected_keys"]
    with torch.inference_mode():
        torch.testing.assert_close(loaded(x).logits, logits, rtol=0, atol=0)
    assert (saved / f"modeling_{family}_svd.py").is_file()


def test_reject_fused_only_and_impossible_budget(tmp_path):
    cfg = LlamaConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    source = tmp_path / "source"
    source.mkdir()
    cfg.save_pretrained(source)
    state = {
        k: v.contiguous() for k, v in AutoModelForCausalLM.from_config(cfg).state_dict().items()
    }
    save_file(state, source / "model.safetensors", metadata={"format": "pt"})
    with pytest.raises(ValueError, match="Original U/V factors"):
        convert_checkpoint(source, tmp_path / "deploy", 0.2)
