# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

seqax is a small (~500 LOC in `train.py`) LLM pretraining codebase in JAX, targeting research at the ~100-GPU/TPU scale. The defining design value is **explicitness**: the math, memory footprint, partitioning, and inter-chip communication are all visible in source rather than hidden behind abstractions. Preserve this when editing — prefer inline, readable code over new abstractions.

## Code Standards

### Verify loss goes down

Whenever you run an experiment configuration, run it against `local_test_synthetic.yaml` using XLA since we are on a Mac. After any edit to `train.py`, `shardlib/`, or `input_loader.py`, run the local synthetic config and confirm loss decreases across the ~50 steps before declaring done.

## Common commands

### Install

Requires Python >= 3.11. CPU dev:

```
uv sync --extra cpu
```

Also requires system `graphviz` (e.g. `brew install graphviz`). For GPU/TPU use `uv sync --extra gpu` or `uv sync --extra tpu`.

### Local CPU smoke test

Simulates 8 devices via XLA flag:

```
XLA_FLAGS=--xla_force_host_platform_device_count=8 uv run python -m train --config-name=local_test_synthetic +paths.model_name=synthetic_000
```

Each file in `configs/` lists its own intended launch command at the top — copy from there for real runs. `paths.model_name` controls the checkpoint subdirectory under `paths.root_working_dir` (default `/tmp`); change it per run.

### Pre-tokenize data

```
uv run python -m tools.huggingface_to_flat_tokens
```

Uses a config from `tools/configs/`. Install `tools/requirements.txt` first.

### Lint and format

CI enforces this via `.github/workflows`:

```
uvx ruff check
uvx ruff format
```

Ruff config lives in `pyproject.toml` (line-length 120, py310, rules `E4,E7,E9,F,I`).

### Tests

There is no test suite. The local synthetic config above is the smoke test.

## Architecture

Top-level files are flat — there is no package layout.

```
seqax/
├── train.py              # entire training program (model, loss, optimizer, loop)
├── init_seqax.py         # XLA flag setup; must import before jax
├── jax_extra.py          # @explicit_activation_checkpointing, save_for_backward
├── input_loader.py       # FlatTokens + HuggingFace data loaders
├── training_io.py        # checkpoint, profile, HLO-graph dumping
├── shardlib/             # in-house sharding library
│   ├── shardtypes.py     # typed pytree dataclasses + sharding annotations
│   └── shardops.py       # all_gather, psum_scatter, einsum_unreduced, ...
├── configs/              # Hydra configs (base.yaml + per-run inherits)
├── tools/                # offline data-prep scripts
│   ├── huggingface_to_flat_tokens.py
│   ├── flat_tokens.py
│   └── configs/          # data-prep configs (c4_en, tinystories, ...)
├── docs/                 # file format specs (flat-tokens, pytree-zarr-checkpoint)
├── synthetic_dataset.zarr  # tiny dataset for the local CPU smoke test
└── pyproject.toml        # ruff config
```

The important pieces:

- **`train.py`** — entire training program in one file: `Hparams`, `Model`/`TransformerLayer` pytree dataclasses, forward pass + loss, optimizer, FSDP + tensor-parallel partitioning, training loop, checkpointing calls, and the Hydra entrypoint. When changing model math, optimizer, or parallelism, this is the file. Activations saved across the backward pass and tensors that hit the checkpoint are intentionally explicit here — keep them so.
- **`shardlib/`** — the in-house sharding library (`shardtypes.py`, `shardops.py`). Tensor types carry sharding in their annotations (e.g. `f32[b"d_model/t/d"]` means the `d_model` axis is sharded across mesh axes `t` and `d`). `typed_shard_map` infers `in_specs`/`out_specs` from these annotations. `shardops` provides `all_gather`, `einsum_unreduced`, etc.; `einsum_unreduced` is **chip-local** — the caller is responsible for any cross-chip reduction implied by the einsum spec. Call `shardtypes.register_with_typeguard()` once at import time so annotations are runtime-checked.
- **`jax_extra.py`** — `@explicit_activation_checkpointing` + `save_for_backward`. By default JAX saves all intermediates and lets XLA prune; this decorator flips the policy so **only** function arguments and explicitly marked values survive into the backward pass. Use `save_for_backward(...)` deliberately — it is the memory knob.
- **`input_loader.py`** — two data paths behind `get_loader`: `FlatTokensParams` (zarr files in the [flat-tokens format](docs/flat-tokens.md), supports exact resume from checkpoint, sequence packing) and `HuggingFaceDataParams` (streaming, convenient for experiments but lossy at batch boundaries and not resumable).
- **`training_io.py`** — checkpoint read/write in the [pytree-zarr format](docs/pytree-zarr-checkpoint.md), plus profile/HLO-graph dumping. Each run writes an XLA Perfetto trace and an optimized-HLO SVG into the model dir.
- **`init_seqax.py`** — must be imported **before** `jax`/`jax.numpy` because it sets environment variables (NCCL tuning, libtpu args). `train.py` imports it first; preserve that ordering in any new entrypoint.
- **`configs/`** — Hydra configs. `base.yaml` is the schema; the rest inherit via `defaults: [base, _self_]`. `mesh.d` is the FSDP/data axis, `mesh.t` is the tensor-parallel axis; their product must equal total devices.
- **`tools/`** — offline data-prep scripts (separate `requirements.txt`).

## Coding patterns in `train.py`

Follow these when editing.

### One file, no helpers

Model, loss, optimizer, training loop, and Hydra entrypoint all live in `train.py`. Resist the urge to extract "utility" modules; the explicitness goal depends on a reader being able to see the whole step linearly.

### Two dataclass flavors

- **`@pytree_dataclass`** (from `shardlib.shardtypes`) for anything containing tensors — every field must carry a sharding-annotated type like `f32[b"d_model/d d_ff/t"]`.
- **`@dataclass(frozen=True)`** for plain hparam/config records (`Hparams`, `TrainingHparams`, `Config`, `Paths`, `MeshConfig`).

Cross-field validation goes in `__post_init__` as `assert`s with messages.

### Annotate, then `@typechecked`

Public methods on tensor-carrying classes (`Model.init`, `forward_pass`, `loss`, `rms_norm`) get `@typechecked` from `typeguard` so the bytes-string sharding annotations are checked at runtime. Don't drop these — they catch shape/sharding mistakes before XLA does.

### Communicate at point of use

`all_gather` a weight on the line just before the einsum that consumes it; `psum_scatter` immediately after the einsum that produced an unreduced output. Don't hoist gathers to the top of a function — locality is what makes the communication pattern legible.

### Dtype discipline

Weights are stored `f32`; cast to `bf16` inline (`jnp.bfloat16(layer_weights.w_q)`) right at the gather. Norm statistics, RoPE tables, softmax accumulators, and the loss stay in `f32` (use `preferred_element_type=jnp.float32` on einsums whose output feeds a reduction). Re-cast back to `bf16` at the end of each block.

### Memory is a manual knob

Inside any `@explicit_activation_checkpointing` function (e.g. the `loop_body` passed to `lax.scan`), wrap any intermediate you want kept for the backward pass in `save_for_backward(...)`. Everything else will be recomputed. When adding new ops, decide explicitly whether each large intermediate is saved.

### `lax.scan` over layers

Use the `Transformer` pytree as the scanned input. New per-layer state must be added as a `TransformerLayer` field, not as a Python loop.

### Reductions outside autodiff

Loss is `psum`'d across `("d", "t")` _after_ `value_and_grad` returns — reducing inside would double-count. Gradient global-norm is also reduced explicitly. Preserve this pattern; don't move reductions into the loss function. The grad of `all_gather` is `psum_scatter` (and vice versa), which is why no extra grad communication is needed.

### Named RNG splits

Always derive subkeys via `jax_extra.fold_in_str(rng, "name")` rather than `jax.random.split` index juggling — names make initialization reproducible and greppable.

### Optimizer step uses tree-leaf zip

AdamW iterates `tree_leaves(weights)`, `grad_leaves`, `tree_leaves(adam_mu/nu)`, and `tree_leaves(make_partition_specs(State))` together, asserting `is_fully_sharded(spec)` per leaf. New optimizer state should be added as a `State` field and threaded through the same zip.

### Donate the state buffer

`@partial(jax.jit, donate_argnums=(0,))` on `training_step` — the previous `state` buffer is donated so XLA can reuse it. Honor this (don't read `state` after the call) when extending the step.

### Comments explain _why_, not _what_

See the notes on the loss-reduction ordering, the `endpoint=False` RoPE quirk, the `check_rep=False` workaround, and `completed_steps = step + 1` for Adam bias correction. Match this style: leave a comment only when the next reader would otherwise be confused or repeat a known mistake.

## Conventions worth preserving

- Sharding annotations use bytes-strings (`b"..."`) inside `pytree_dataclass` fields so they survive `typeguard`.
- Mesh axis names are conventionally `d` (data/FSDP) and `t` (tensor parallel). New axes should be added consciously since they propagate through every annotation.
- Pipeline parallelism is **not** implemented — README explicitly notes this is the scaling ceiling.
- ClearML is imported unconditionally in `train.py` for experiment tracking.
