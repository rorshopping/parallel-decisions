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
