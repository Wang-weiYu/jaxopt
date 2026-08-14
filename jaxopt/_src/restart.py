"""Restart an LBFGSB fit from a new random point when it stalls.

Restarting is intentionally outside LBFGSB's JIT loop: only application code
knows how a seed should generate valid parameters, and each restart is a new
optimization problem/run.
"""
from typing import Any, NamedTuple, Optional

import jax
import jax.numpy as jnp


class RestartResult(NamedTuple):
  """Result and random-seed diagnostics from stall-aware restarts."""
  solution: Any
  seeds_used: int
  last_seed: Optional[int]
  attempts: int


def _random_in_bounds(key, lower, upper):
  """Generate one random parameter pytree uniformly inside finite bounds."""
  lower_leaves, structure = jax.tree_util.tree_flatten(lower)
  upper_leaves = jax.tree_util.tree_leaves(upper)
  keys = jax.random.split(key, len(lower_leaves))
  leaves = []
  for leaf_key, lo, hi in zip(keys, lower_leaves, upper_leaves):
    if not bool(jnp.all(jnp.isfinite(lo)) & jnp.all(jnp.isfinite(hi))):
      raise ValueError("Random restart requires finite lower and upper bounds")
    leaves.append(jax.random.uniform(leaf_key, lo.shape, lo.dtype, lo, hi))
  return jax.tree_util.tree_unflatten(structure, leaves)


def run_with_stall_restarts(solver, initial_params, bounds, *args,
                            max_restarts=3, first_seed=0, **kwargs):
  """Run LBFGSB and randomly restart it whenever it reports a stall.

  The best solution by objective value is returned. ``seeds_used`` counts only
  random restarts; the supplied initial parameters do not consume a seed.
  Seeds are the consecutive integers ``first_seed, first_seed + 1, ...``.
  """
  if max_restarts < 0:
    raise ValueError("max_restarts must be non-negative")

  params = initial_params
  best = None
  seeds_used = 0
  last_seed = None

  for attempt in range(max_restarts + 1):
    solution = solver.run(params, bounds, *args, **kwargs)
    jax.block_until_ready(solution.params)
    if best is None or float(solution.state.value) < float(best.state.value):
      best = solution

    if not bool(solution.state.stalled) or attempt == max_restarts:
      return RestartResult(best, seeds_used, last_seed, attempt + 1)

    last_seed = first_seed + seeds_used
    key = jax.random.PRNGKey(last_seed)
    params = _random_in_bounds(key, *bounds)
    seeds_used += 1

  raise AssertionError("unreachable")
