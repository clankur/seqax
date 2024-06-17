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
prngkey = jax.random.PRNGKey(0)
# %%
# actually implement the logic with sharding
@pytree_dataclass
class Weights:
  w1: f32['in hidden1/t']
  w2: f32['hidden1/t hidden2']

with Mesh(np.reshape(jax.devices()[:4], (2, 2)), ('d', 't')) as mesh:
  # Create dummy weights.
  w = Weights(
    w1= jax.random.randint(prngkey, shape=(4, 8), minval=-9, maxval=10).astype(jnp.float32),
    w2= jax.random.randint(prngkey, shape=(8, 4), minval=-9, maxval=10).astype(jnp.float32),
  )
  # w = jax.tree.map(jax.device_put, w, make_shardings(Weights))
 
  from functools import partial
  
  # @typed_shard_map
  @partial(typed_shard_map, check_rep=False)
  def forward_pass(x: f32[b'batch/d in'], w: Weights) -> f32[b'batch/d hidden2/t']:
    print(f"{x=}")
    print(f"{w.w1=}")
    out1 = shardops.einsum_unreduced('batch/d in, in hidden1/t -> batch/d hidden1/t', x, w.w1)
    print(f"{out1=}")
    print(f"{w.w2=}")
    out2 = shardops.einsum_unreduced('batch/d hidden1/t, hidden1/t hidden2 -> batch/d hidden2', out1, w.w2)
    print(f'{out2=}')
    out2 = shardops.psum_scatter('batch/d hidden2 -> batch/d hidden2/t', out2)
    print(f'{out2=}')
    return out2

  b = 2
  x = jax.random.randint(prngkey, shape=(b, 4), minval=2, maxval=9).astype(jnp.float32)
  print(x)
  out = forward_pass(x, w)
  print(f'{out=}')
# %%

