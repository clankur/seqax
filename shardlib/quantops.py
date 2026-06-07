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
    dtype: Any  # storage dtype: jnp.float8_e4m3fn / jnp.int8 / jnp.int4
    maxval: float  # max representable magnitude, used for amax scaling
    accum: Any  # preferred_element_type for the matmul: jnp.float32 (float) or jnp.int32 (int)
    integer: bool  # int formats need an explicit round before the cast; float casts round themselves


FP8_E4M3 = QuantFormat(jnp.float8_e4m3fn, 448.0, jnp.float32, integer=False)
INT8 = QuantFormat(jnp.int8, 127.0, jnp.int32, integer=True)
INT4 = QuantFormat(jnp.int4, 7.0, jnp.int32, integer=True)


def _quantize(x, fmt: QuantFormat):
    """Per-tensor "current" amax scaling into `fmt`. Returns (low-precision x, f32 scale)."""
    scale = jnp.maximum(jnp.max(jnp.abs(jnp.float32(x))) / fmt.maxval, 1e-12)
    z = jnp.clip(jnp.float32(x) / scale, -fmt.maxval, fmt.maxval)  # clamp keeps values in range (no NaN/wrap)
    return (jnp.round(z) if fmt.integer else z).astype(fmt.dtype), scale


# Real low-precision matmul: low-precision forward GEMM, bf16 straight-through backward. This needs
# custom_vjp rather than the `x + stop_gradient(x_dq - x)` STE used in shardlib: float addition isn't
# associative, so `bf16_out + stop_gradient(lowp_out - bf16_out)` won't let XLA drop the redundant bf16
# forward GEMM, which would negate the speedup. With fp8 operands + f32 accumulation, XLA fuses the
# convert(f8)->f32 dot into a cublasLt fp8 matmul on Ada/Hopper; int formats (and any format the GPU
# lacks) run emulated -- correct numerics, no speedup. The backward is format-independent (bf16
# straight-through), so only the forward reads `fmt`.
@partial(jax.custom_vjp, nondiff_argnums=(0, 1))
def quantized_einsum(jaxspec: str, fmt: QuantFormat, x, y):
    return _quantized_einsum_fwd(jaxspec, fmt, x, y)[0]


def _quantized_einsum_fwd(jaxspec, fmt, x, y):
    xq, sx = _quantize(x, fmt)
    yq, sy = _quantize(y, fmt)
    acc = jnp.einsum(jaxspec, xq, yq, preferred_element_type=fmt.accum)  # f32/int32 accumulate
    return jnp.bfloat16(jnp.float32(acc) * sx * sy), (x, y)


def _quantized_einsum_bwd(jaxspec, fmt, res, g):
    # Straight-through bf16. The vjp of an einsum "lhs,rhs->out" is two more einsums with the index
    # letters rearranged: dx = einsum("out,rhs->lhs", g, y), dy = einsum("lhs,out->rhs", x, g).
    x, y = res
    lhs, rhs_out = jaxspec.split(",")
    rhs, out = rhs_out.split("->")
    g = jnp.bfloat16(g)
    dx = jnp.einsum(f"{out},{rhs}->{lhs}", g, y)
    dy = jnp.einsum(f"{lhs},{out}->{rhs}", x, g)
    return jnp.bfloat16(dx), jnp.bfloat16(dy)


quantized_einsum.defvjp(_quantized_einsum_fwd, _quantized_einsum_bwd)
