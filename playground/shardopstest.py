# %%
import os
import jax.experimental
import jax.experimental.multihost_utils

# %%
os.environ.update({
  "NCCL_LL128_BUFFSIZE": "-2",
  "NCCL_LL_BUFFSIZE": "-2",
   "NCCL_PROTO": "SIMPLE,LL,LL128",
 })

UCX_NET_DEVICES = 'mlx5_0:1'
NCCL_IB_HCA = 'mlx5_5,mlx5_6,mlx5_7,mlx5_8,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_14,mlx5_15,mlx5_16,mlx5_17,mlx5_9,mlx5_10,mlx5_11,mlx5_12'
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8' # Use 8 CPU devices

# Environment variables setup
# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_IB_SL'] = '0'
os.environ['NCCL_IB_TC'] = '41'
os.environ['NCCL_IB_QPS_PER_CONNECTION'] = '4'
os.environ['UCX_TLS'] = 'ud,self,sm'
os.environ['UCX_NET_DEVICES'] = UCX_NET_DEVICES
os.environ['HCOLL_ENABLE_MCAST_ALL'] = '0'
os.environ['coll_hcoll_enable'] = '0'
os.environ['NCCL_IB_GID_INDEX'] = '3'
os.environ['NCCL_ALGO'] = 'Ring'
os.environ['NCCL_IB_HCA'] = NCCL_IB_HCA
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

# %%
import shardlib.shardtypes as shardtypes
shardtypes.register_with_typeguard()
from functools import cached_property, partial
from typing import Any, Optional, Tuple
from typeguard import typechecked
from dataclasses import dataclass
import operator
import jax
from jax import lax
from jax.sharding import PartitionSpec
import jax.numpy as jnp
from clearml import Task

from input_loader import FlatTokensParams, ShufflingLoader, TokenBatch, TokenBatchParams
from shardlib.shardtypes import bf16, bool_, f32, pytree_dataclass, u32, make_shardings
import shardlib.shardops as shardops
P = PartitionSpec
from jax_extra import fold_in_str #set_mesh, with_sharding, get_mesh, stop_gradient_pmax
from hydra.core.config_store import ConfigStore
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.tree_util import tree_leaves
from jax.experimental.shard_map import shard_map
from shardlib.shardtypes import f32, typed_shard_map
import argparse
import numpy as np

# %%
jax.config.update("jax_disable_jit", True)
print(jax.devices())

# %%
@pytree_dataclass
class Weights:
  w1: f32['in/d hidden1/t']
  w2: f32['hidden1/t hidden2/d']
  w3: f32['hidden2/d']

with Mesh(np.reshape(jax.devices(), (4, 2)), ('d', 't')) as mesh:
  # Create dummy weights.
  w = Weights(
    w1= jnp.arange(64).astype(jnp.float32).reshape(8, 8),
    w2=jnp.ones((8, 8), dtype=jnp.float32),
    w3=jnp.ones((8,), dtype=jnp.float32),
  )
  w = jax.tree.map(jax.device_put, w, make_shardings(Weights))
 
  from functools import partial
  
  # @typed_shard_map
  @partial(typed_shard_map, check_rep=False)
  def forward_pass(x: f32[b'batch/d in'], w: Weights) -> f32[b'batch/d hidden2']:
    print(f'{x=}')
    w1 = shardops.all_gather('in/d hidden1/t -> in hidden1/t', w.w1)
    # print(f'{w1=}')
    print(f"{w1.shape=}")
    
    y = jax.nn.relu(shardops.einsum_unreduced('batch/d in, in hidden1/t -> batch/d hidden1/t', x, w1))
    
    w2 = shardops.all_gather('hidden1/t hidden2/d -> hidden1/t hidden2', w.w2)
    z = jax.nn.relu(shardops.einsum_unreduced('batch/d hidden1/t, hidden1/t hidden2 -> batch/d hidden2', y, w2))

    print(f'{z.shape=}')
    print(f'{z=}')
    z1 = shardops.psum_scatter('batch/d hidden2 -> batch/d hidden2/t', z)
    print(f'{z1.shape=}')
    z1 = shardops.all_gather('batch/d hidden2/t -> batch/d hidden2', z1)
    z2 = jax.lax.psum(z, 't')
    print(f'{z1.shape=}, {z2.shape=}')
    print(f'{z1=}')
    print(f'{z2=}')
    # w3 = shardops.all_gather('hidden2/d -> hidden2', w.w3)
    # out = shardops.einsum_unreduced('batch/d hidden2, hidden2 -> batch/d', z, w3)
    # print(f'{out=}') 
    return z2
    # return jax.lax.psum(out, 't')

  b = 4 
  x = forward_pass(jnp.arange(b*8).reshape(b, 8).astype(jnp.float32), w)

# %%
# replicating experiment with akshay
B, M = 1, 4
t = 2
x = np.arange(B * M).reshape(B, M).astype(np.float32)
w1 = np.random.randint(2, 8, (4, 8)).astype(np.float32)
w2 = np.random.randint(2, 8, (8, 4)).astype(np.float32)
x, w1, w2
# %%
out1 = x @ w1
out1
# %%
out2 = out1 @ w2
out2
# %%
D = 8
w11 = w1[:, :D//t]
w12 = w1[:, D//t:]

w21 = w2[:D//t,:]
w22 = w2[D//t:,:]
w11, w12, w21, w22
# %%
w1.shape
x[0][0] * w1[0] + x[0][1] * w1[1] + x[0][2] * w1[2] + x[0][3] * w1[3]
# %%
out11 = x @ w11
out12 = x @ w12
out11, out12
# %%
# out11 and out12 are the coefficients for the mat mul on each row of w2
out21 = out11 @ w21
out22 = out12 @ w22
out21, out22
# %%
out21 + out22
# %%
# do the same with tensor parallel
# TODO: draw out the dot products when doing split mat_muls
# when you are doing the first half on the first device and the second half on the second device
# you are going to have sum at the end to get the final dot product result
B, C = 2, 4
D = 8
t = 2
x = np.arange(B * C).reshape(B, C).astype(np.float32)
w1 = np.random.randint(1, 8, (C, D)).astype(np.float32)
x @ w1
# %%
# shard w1
w11 = w1[:, :D//t]
w12 = w1[:, D//t:]
x @ w11, x@ w12

# %%
# draw the dot product all of x applied to w11
(
  x[0][0] * w11[0] + x[0][1] * w11[1] + x[0][2] * w11[2] + x[0][3] * w11[3],
  x[1][0] * w11[0] + x[1][1] * w11[1] + x[1][2] * w11[2] + x[1][3] * w11[3]
)
# %%
# apply to w12
(
  x[0][0] * w12[0] + x[0][1] * w12[1] + x[0][2] * w12[2] + x[0][3] * w12[3],
  x[1][0] * w12[0] + x[1][1] * w12[1] + x[1][2] * w12[2] + x[1][3] * w12[3]
)

# %%
# the sum of the two dot products is the same as the dot product of the original matrix
# %%
# actually implement the logic with sharding

@pytree_dataclass
class SimpleWeights:
  w1: f32['in hidden/t']
  w2: f32['hidden/t out']

with Mesh(np.reshape(jax.devices()[:2], (2)), ('t')) as mesh:
   
  @partial(typed_shard_map, check_rep=False)
  def forward_pass(x: f32['batch in'], w: SimpleWeights) -> f32['batch out']:
    out1 = shardops.einsum_unreduced('B C, C D/t -> B D/t', x, w.w1)
    out2 = shardops.einsum_unreduced('B D/t, D/t C -> B C', out1, w.w2)
    out2 = shardops.psum_scatter('B C -> B C/t', out2)
    out2 = shardops.all_gather('B C/t -> B C', out2)
    return out2
  
  w = SimpleWeights(
    w1= jnp.arange(32).astype(jnp.float32).reshape(4, 8),
    w2=jnp.ones((8, 4), dtype=jnp.float32),
  )
  b = 2
  # w = jax.tree.map(jax.device_put, w, make_shardings(SimpleWeights))
  x = jnp.arange(8).reshape(b, 4).astype(jnp.float32)
  forward_pass(x, w)
  
# %%
