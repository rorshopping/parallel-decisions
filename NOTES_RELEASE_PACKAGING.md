# Release packaging handoff

Branch: `feat/release-packaging`. Scope: `pyproject.toml`, `engine.py`, packaging/default-selection tests, this note. No graph implementation changes, shared environment installs, process termination, pushes or merges.

## Changes and integration documentation

- Base dependencies `mlx>=0.22` and `mlx-lm>=0.21` now apply only when `sys_platform == 'darwin'` and machine is `arm64` or `aarch64`, matching backend auto-selection. Windows/Linux/Intel Mac base installs do not request either MLX package or Torch. Base installs remain useful for schema/config/calibration helpers; inference on those platforms needs the `torch` extra.
- `.[torch]` declares `torch>=2.6`, `transformers>=5.17,<6`, `accelerate>=1.1`. The runtime uses the Transformers 5 cache/dtype APIs; 5.17 is the deliberately conservative API baseline corresponding to the existing implementation. Torch 2.6 is the existing measured release baseline. Accelerate 1.1 matches the Transformers 5.17 Torch-extra minimum and supports the model-loading ecosystem. These bounds are **not** a claim that every allowed release/platform combination has been tested. Transformers 6 is excluded because the cache interface is version-sensitive. A broader old-version matrix was not tested.
- `.[dev]` also declares `packaging>=23` for marker/specifier regression tests.
- Resolve backend before selecting the fallback model. MLX keeps `mlx-community/Qwen2.5-7B-Instruct-4bit`; Torch uses the existing `engine_torch.DEFAULT_TORCH_MODEL`, `Qwen/Qwen2.5-0.5B-Instruct` (no duplicated runtime default).
- `parallel_decisions.DEFAULT_MODEL` and `engine.DEFAULT_MODEL` retain the original MLX ID. Passing that constant explicitly still uses it; it is not silently translated to a Torch model. Explicit/configured incompatible IDs are likewise preserved.
- Normal precedence remains explicit model/backend argument > environment > configuration file > backend-specific default. Passing a `Config` object still means already-resolved configuration and does not overlay environment variables again. Construction/import remains lazy without MLX, Torch or Transformers installed.
- Coordinator should update README/GPU_SETUP/AGENTS/CHANGELOG: remove the unconditional-MLX/no-extra and incompatible-default limitations; retain CUDA wheel/device/dtype and VRAM warnings. Accuracy figures for the old MLX 7B model do **not** apply to the Torch 0.5B default. No versions/changelogs were changed here.

## Exact installation recipes (for documentation; not executed in the shared venv)

Run from the checkout containing this change. Use a fresh environment, not the shared GPU worktree environment. Do not combine the alternatives below in one environment.

### Windows: pip, default package index

```powershell
py -3.12 -m venv .venv
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install -e ".[torch]"
```

### Windows: uv, default package index

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[torch]"
```

For local development use `.[torch,dev]` instead. Omit `[torch]` only for helpers without inference. Plain `.[dev]` on Windows intentionally does not supply Torch. Default-index wheels do not guarantee CUDA support or suitability for a particular GPU. Choose a compatible PyTorch wheel and driver separately for CUDA.

### Windows: historical CUDA 12.4 wheel baseline, with uv

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv\Scripts\python.exe -e ".[torch,dev]" "torch==2.6.0" "transformers==5.17.0" "accelerate==1.15.0"
```

### Windows: same baseline, with pip

```powershell
py -3.12 -m venv .venv
& .venv\Scripts\python.exe -m pip install --upgrade pip
& .venv\Scripts\python.exe -m pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cu124
& .venv\Scripts\python.exe -m pip install -e ".[torch,dev]" "torch==2.6.0" "transformers==5.17.0" "accelerate==1.15.0"
```

The second step uses the normal package index, retaining the previously installed compatible Torch wheel. No `--no-deps` workaround is necessary with the fixed metadata. These are install recipes, not fresh-environment or GPU validation performed by this agent. The hardware/driver caveats and original measurement evidence remain in GPU_SETUP.md. Explicit `torch_device="cuda", torch_dtype="float16"` is still the documented baseline for the Turing card, not an implication that automatic bf16 is fast.

### Apple Silicon: existing default installation

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -e ".[dev]"
```

Apple Silicon base installs retain MLX. Selecting Torch there requires `.[torch]` and `backend="torch"` (installing the extra alone does not change backend auto-selection).

## Verification

All Python tests used `C:\Users\Richard\Documents\Projects\parallel-decisions_wt\gpu\.venv\Scripts\python.exe` (Python 3.12.13), with `PYTHONPATH` pinned to this worktree's `src`; the printed engine import path was verified. Existing installed versions: Torch 2.6.0+cu124, Transformers 5.17.0, Accelerate 1.15.0, setuptools 78.1.0, packaging 26.3. No installation or alteration of that environment was performed. That environment has no `pip` module; version inspection used `importlib.metadata` instead.

- Before the facade fix: `tests/test_model_defaults.py`: **9 failed, 11 passed in 0.15s**, exposing all Torch-default mismatches.
- After the fix: **20 passed in 0.09s** for that same new regression file.
- Focused defaults + packaging + existing configuration tests: **56 passed in 2.22s**.
- Final CPU-only/offline suite: **174 passed, 3 skipped, 4 deselected in 10.94s** (exit 0). The skips are MLX DLL unavailability in shared-prefix/example/memory tests. Four CUDA-dependent graph cases were deliberately deselected; CPU graph/config plumbing remains included.
- `git diff --check`: passed.
- Emitted METADATA tested via setuptools `prepare_metadata_for_build_wheel` on a temporary source copy, evaluating actual `Requires-Dist` markers for Windows/Linux/Intel Mac/Apple Silicon and the Torch extra. This requires no installation/network/model; the normal build dependency is skipped if setuptools is absent. A complete fresh pip/uv dependency resolution and wheel install were not run.
- Public import plus Torch-default construction tested in a subprocess blocking imports of `mlx`, `mlx_lm`, `torch`, `transformers`; all model-default tests also block backend dependency imports.

Final suite command:

```powershell
$env:PYTHONPATH = "C:\Users\Richard\Documents\Projects\parallel-decisions_wt\release-packaging\src"
$env:CUDA_VISIBLE_DEVICES = "-1"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$py = "C:\Users\Richard\Documents\Projects\parallel-decisions_wt\gpu\.venv\Scripts\python.exe"
& $py -c "import parallel_decisions.engine as m; print(m.__file__)"
& $py -m pytest -q -rs --deselect=tests/test_cuda_graph.py::test_graph_replay_fresh_buffers_and_bucket_lengths --deselect=tests/test_cuda_graph.py::test_graph_failure_falls_back_once --deselect=tests/test_cuda_graph.py::test_graph_prefix_collisions_chunks
exit $LASTEXITCODE
```

### Verification deviation (important)

An earlier full run used `$env:CUDA_VISIBLE_DEVICES = ""`, which removes the variable in this Windows PowerShell rather than masking CUDA. Consequently the existing four tiny synthetic CUDA tests inadvertently ran: **178 passed, 3 skipped in 13.06s**. This was contrary to the requested no-GPU-run boundary; it is disclosed rather than described as CPU-only verification. No downloaded/pretrained model or benchmark was run. Once noticed, no further GPU tests were run: the explicit `-1` value and test deselection above were used, with two successful CPU-only runs (12.16s then final 10.94s). There were no process kills or shared-environment installs.

## Known limits

- Metadata simulation is not cross-platform installation testing; no clean resolver/install, Apple Silicon execution, Python-version matrix or minimum-version matrix was performed. Python 3.12 is the documented install recipe; project metadata still permits Python >=3.10, whose existing pd.toml reader requires 3.11+.
- Torch remains optional on Windows/Linux; `Decider()` is lazy, but calling inference without the extra still raises a missing-dependency error. This change does not add dependency diagnostics in engine_torch.py.
- MLX-only `load_tokenizer` behavior and Torch tokenizer-only loading were not changed (Torch currently falls back to loading the model).
- Torch chunking remains fixed-row, not a guaranteed VRAM budget. No 7B fit, default 0.5B accuracy, CUDA performance or graph-compatibility claims were added. Graph behavior and GPU optimization code are unchanged.
