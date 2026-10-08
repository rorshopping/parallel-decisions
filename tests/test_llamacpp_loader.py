"""Model-free regression for the native loader's tested numerical setting."""
import sys
from types import SimpleNamespace

import pytest

from parallel_decisions.engine_llamacpp import LlamaCppRuntime


def test_loader_disables_extra_buffer_types_before_model_creation(tmp_path, monkeypatch):
    model_path = tmp_path / "synthetic.gguf"
    model_path.touch()
    params = SimpleNamespace(n_gpu_layers=-1, use_extra_bufts=True)
    seen = []

    class StopAtModel(Exception):
        pass

    def model(**kwargs):
        seen.append(kwargs)
        assert kwargs["params"].n_gpu_layers == 0
        assert kwargs["params"].use_extra_bufts is False
        raise StopAtModel

    native = SimpleNamespace(
        llama_model_default_params=lambda: params,
        llama_backend_init=lambda: None,
        llama_max_parallel_sequences=lambda: 64,
    )
    # `llama_synchronize` is part of the required native set (the prefill path
    # must finish before the prefill clock stops) and is provided by the real
    # 0.3.35 binding, so the stand-in has to list it too.
    for name in ("llama_get_memory", "llama_memory_seq_cp", "llama_memory_seq_rm",
                 "llama_memory_clear", "llama_batch_init", "llama_batch_free",
                 "llama_decode", "llama_get_logits_ith", "llama_n_ctx_seq",
                 "llama_synchronize"):
        setattr(native, name, lambda *args: None)
    fake = SimpleNamespace(__version__="0.3.35", llama_cpp=native,
                           _internals=SimpleNamespace(LlamaModel=model))
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    runtime = LlamaCppRuntime(model_path)
    with pytest.raises(StopAtModel):
        runtime.load()
    assert len(seen) == 1
    assert seen[0]["path_model"] == str(model_path)
    runtime.close()


def test_loader_offloads_every_layer_when_gpu_requested(tmp_path, monkeypatch):
    """n_gpu_layers=-1 must reach the model params (and keep the CPU-path
    invariants: extra buffer types stay disabled, KV/op offload on)."""
    model_path = tmp_path / "synthetic.gguf"
    model_path.touch()
    params = SimpleNamespace(n_gpu_layers=0, use_extra_bufts=True)
    model_kwargs = []
    context_kwargs = []

    def model(**kwargs):
        model_kwargs.append(kwargs)
        assert kwargs["params"].n_gpu_layers == -1
        assert kwargs["params"].use_extra_bufts is False
        return SimpleNamespace(
            close=lambda: None,
            metadata=lambda: {"general.architecture": "qwen2"},
            n_vocab=lambda: 16,
            n_params=lambda: 1,
            model_desc=lambda: "synthetic",
            desc=lambda: "synthetic",
        )

    def context_ctor(**kwargs):
        context_kwargs.append(kwargs)
        assert kwargs["params"].offload_kqv is True
        assert kwargs["params"].op_offload is True
        return SimpleNamespace(close=lambda: None, ctx=object())

    native = SimpleNamespace(
        llama_model_default_params=lambda: params,
        llama_context_default_params=lambda: SimpleNamespace(),
        llama_backend_init=lambda: None,
        llama_max_parallel_sequences=lambda: 64,
        llama_get_memory=lambda ctx: object(),
        llama_n_ctx_seq=lambda ctx: 4096,
        llama_batch_init=lambda n, emb, seq: SimpleNamespace(token=[0] * n),
    )
    for name in ("llama_memory_seq_cp", "llama_memory_seq_rm",
                 "llama_memory_clear", "llama_batch_free",
                 "llama_decode", "llama_get_logits_ith", "llama_synchronize"):
        setattr(native, name, lambda *args: None)
    fake = SimpleNamespace(
        __version__="0.4.2", llama_cpp=native,
        _internals=SimpleNamespace(LlamaModel=model, LlamaContext=context_ctor),
    )
    monkeypatch.setitem(sys.modules, "llama_cpp", fake)
    runtime = LlamaCppRuntime(model_path, n_gpu_layers=-1)
    runtime.load()
    assert model_kwargs and model_kwargs[0]["path_model"] == str(model_path)
    assert context_kwargs and context_kwargs[0]["params"].offload_kqv is True
    assert runtime.metadata["architecture"] == "qwen2"
    runtime.close()
