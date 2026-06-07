"""One-off diagnostic for the fp8 cuBLASLt failure on the RTX 4090.

Prints the GPU/JAX/CUDA environment, then runs each of seqax's six projection einsums in fp8 (e4m3,
f32 accumulate) in isolation and reports OK/FAIL per shape. This separates "fp8 is broken in this
env" (all FAIL) from "a specific matmul shape/layout is unsupported by this cuBLASLt" (some FAIL).

Run on the GPU box:  uv run --extra gpu python -m tools.fp8_diag
"""

import subprocess

import jax
import jax.numpy as jnp

print("=== nvidia-smi ===")
try:
    print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout)
except Exception as e:
    print("nvidia-smi failed:", e)

print("=== jax.print_environment_info() ===")
jax.print_environment_info()
print("devices:", jax.devices())

F8 = jnp.float8_e4m3fn
# 31m config sizes (mesh d=1/t=1, so full, unsharded). B reduced to keep it light.
B, L, M, Q, K, D, F = 2, 1024, 256, 1, 16, 64, 1024


def probe(name, spec, xshape, yshape):
    x = jnp.ones(xshape, dtype=F8)
    y = jnp.ones(yshape, dtype=F8)
    fn = jax.jit(lambda a, b: jnp.einsum(spec, a, b, preferred_element_type=jnp.float32))
    try:
        r = fn(x, y)
        r.block_until_ready()
        print(f"OK   {name:7} {spec:22} {tuple(xshape)} x {tuple(yshape)} -> {r.shape} {r.dtype}")
    except Exception as e:
        print(f"FAIL {name:7} {spec:22} {type(e).__name__}: {str(e)[:260]}")


# Mirror the six fp8 projection einsums (single-letter equivalents of the model's specs).
probe("w_gate", "blm,mf->blf", (B, L, M), (M, F))
probe("w_up", "blm,mf->blf", (B, L, M), (M, F))
probe("w_down", "blf,mf->blm", (B, L, F), (M, F))
probe("w_q", "blm,mqkd->blqkd", (B, L, M), (M, Q, K, D))
probe("w_kv", "blm,vmkd->vblkd", (B, L, M), (2, M, K, D))
probe("w_o", "bsqkd,mqkd->bsm", (B, L, Q, K, D), (M, Q, K, D))
print("=== done ===")
