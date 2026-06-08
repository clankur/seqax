"""Low-precision (quantized) matmul primitive.

This is a *precision* concern, orthogonal to the *sharding* concern in shardops.py. The quantized
matmul is chip-local: it operates on a prebuilt single-letter einsum spec produced by
shardops.einsum_unreduced, so it needs no sharding/shardtypes knowledge and does no collectives.
"""

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


@dataclass(frozen=True)
class QuantFormat:
    dtype: Any  # storage dtype: jnp.float8_e4m3fn / jnp.float8_e5m2 / jnp.int8 / jnp.int4
    maxval: float  # max representable magnitude, used for amax scaling
    integer: bool  # int formats need an explicit round before the cast; float casts round themselves


FP8_E4M3 = QuantFormat(jnp.float8_e4m3fn, 448.0, integer=False)
FP8_E5M2 = QuantFormat(jnp.float8_e5m2, 57344.0, integer=False)
INT8 = QuantFormat(jnp.int8, 127.0, integer=True)
INT4 = QuantFormat(jnp.int4, 7.0, integer=True)


def _quantize(x: jax.Array, fmt: QuantFormat) -> tuple[jax.Array, jax.Array]:
    """Per-tensor "current" amax scaling into `fmt`. Returns (low-precision x, f32 scale)."""
    scale = jnp.maximum(jnp.max(jnp.abs(jnp.float32(x))) / fmt.maxval, 1e-12)
    scaled = jnp.clip(jnp.float32(x) / scale, -fmt.maxval, fmt.maxval)  # clamp keeps values in range (no NaN/wrap)
    return (jnp.round(scaled) if fmt.integer else scaled).astype(fmt.dtype), scale


def _lowp_matmul(
    spec: str, a: jax.Array, a_scale: jax.Array, b: jax.Array, b_scale: jax.Array, *, integer: bool
) -> jax.Array:
    """Chip-local GEMM on two already-quantized operands, rescaled back to bf16 by their per-tensor
    scales. A pure integer GEMM accumulates in int32; anything with an fp8 operand accumulates in f32."""
    acc = jnp.einsum(spec, a, b, preferred_element_type=jnp.int32 if integer else jnp.float32)
    return jnp.bfloat16(jnp.float32(acc) * a_scale * b_scale)


# custom_vjp, not the `x + stop_gradient(x_dq - x)` STE used elsewhere in shardlib: float add isn't
# associative, so the STE wouldn't let XLA drop a redundant bf16 forward GEMM, negating the speedup.
# fp8 operands + f32 accumulate let XLA fuse convert(f8)->f32 into a cublasLt fp8 matmul on Ada/Hopper
# (int / unsupported formats emulate -- correct numerics, no speedup).
@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def quantized_einsum(spec: str, fmt: QuantFormat, x: jax.Array, y: jax.Array) -> jax.Array:
    return _quantized_einsum_fwd(spec, fmt, x, y)[0]


def _quantized_einsum_fwd(
    spec: str, fmt: QuantFormat, x: jax.Array, y: jax.Array
) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
    x_q, x_scale = _quantize(x, fmt)
    y_q, y_scale = _quantize(y, fmt)
    # Save the quantized operands, not the bf16 inputs -- the backward reuses them, halving saved bytes.
    return _lowp_matmul(spec, x_q, x_scale, y_q, y_scale, integer=fmt.integer), (x_q, x_scale, y_q, y_scale)


def _quantized_einsum_bwd(
    spec: str, fmt: QuantFormat, res: tuple[jax.Array, jax.Array, jax.Array, jax.Array], g: jax.Array
) -> tuple[jax.Array, jax.Array]:
    # vjp of "lhs,rhs->out": dx = einsum("out,rhs->lhs", g, y), dy = einsum("lhs,out->rhs", x, g).
    x_q, x_scale, y_q, y_scale = res
    lhs, rhs_out = spec.split(",")
    rhs, out = rhs_out.split("->")
    # gradients need range over precision -> e5m2; operands stay e4m3. The standard Transformer Engine split.
    g_q, g_scale = _quantize(g, FP8_E5M2)
    dx = _lowp_matmul(f"{out},{rhs}->{lhs}", g_q, g_scale, y_q, y_scale, integer=False)
    dy = _lowp_matmul(f"{lhs},{out}->{rhs}", x_q, x_scale, g_q, g_scale, integer=False)
    return dx, dy


quantized_einsum.defvjp(_quantized_einsum_fwd, _quantized_einsum_bwd)
