# One model, too much data

Both tools take the same `model_fn(data) -> pm.Model` and any of the data containers described in [Data](data.md). They split the rows into shards and call `model_fn` on each shard.

The model's free random variables (names and shapes) must not depend on which rows `model_fn` receives. **Build coords from global metadata**, not from the data.

```python
STORES = np.array(sorted(all_store_ids)) # global, known up front


def model_fn(data):
    stores = nw.from_native(data, eager_only=True)["store"].to_numpy()
    idx = np.searchsorted(STORES, stores) # good: positions in the global list
    # coords={"store": np.unique(stores)} # bad: differs between shards
    with pm.Model(coords={"store": STORES}) as model:
        ...
```

`ShardedModel` checks this and raises an error naming the offending variables.

## Exact: `ShardedModel`

The API reference page for {class}`~pymc_ray.ShardedModel` lists every argument and method.

```python
data = ray.data.read_parquet("s3://bucket/events/") # each actor loads only its shard

with pmr.ShardedModel(model_fn, data, n_shards=32, shard_by="group") as sharded:
    print(sharded.benchmark(chains=4)) # seconds per logp+gradient evaluation
    idata = sharded.sample(draws=1000, chains=4)
```

**How it works.** Each shard is held by a Ray actor that compiles the log-likelihood and its gradient for its rows. On the driver, `model_fn` is called with zero rows (leaving only the priors), and a `Potential` is added whose value is a PyTensor `Op` that sends the current parameters to every actor and sums what they return. NUTS therefore targets the exact full-data posterior. Each leapfrog step costs one round trip to the actors, and only the parameters and gradients travel over the network. The logp and gradient match the full-data model.

- **Samplers.** `sharded.model` is an ordinary `pm.Model`, and `sharded.sample(**kwargs)` passes everything on to `pm.sample`.

  - With PyMC's own NUTS (`nuts_sampler="pymc"`), chain processes are started with `mp_ctx="spawn"` because a Ray connection does not survive `fork()`, so forked chain processes would deadlock. pymc-ray raises an error instead of hanging if you try.

  - **nutpie** also works (`nuts_sampler="nutpie"`, with its default Numba backend), and PyMC uses it by default whenever it is installed. Its chains are threads in one process, so there are no chain processes to spawn, and each thread leases its own replica.

  - **Ray Client** (a `ray://` address, e.g. a notebook on your laptop): `sharded.sample()` and `sharded.benchmark()` run in a Ray task on the node where the session's driver runs on the cluster (normally the head node), and only the result comes back. The progress bar is off by default there, because it would print on the cluster.

- **Data-dependent Potentials** (e.g. a custom likelihood written as `pm.Potential`) must be listed in `likelihood_potentials=[...]` so they are summed over shards. Other Potentials are treated as part of the prior.

- **Missing-value imputation** is not supported: it creates free variables that depend on the rows.

- **Placement and communication options** (`shard_by`, `n_replicas`, `relays`, `actor_options`) are explained in [Performance](performance.md).

## Approximate: `consensus_sample`

The API reference page for {func}`~pymc_ray.consensus_sample` lists every argument.

```python
idata = pmr.consensus_sample(
    model_fn,
    data,
    n_shards=16,
    sample_kwargs={"draws": 1000, "chains": 4},
)
```

**How it works.** Consensus Monte Carlo ([Scott et al., 2016](https://doi.org/10.1080/17509653.2016.1142191)). Shard *s* is sampled independently on its own worker from the subposterior $p(\theta)^{(1/S)} \cdot p(shard_{s} | \theta)$, with no communication during sampling. The product of the subposteriors is the full posterior. The draws are combined on the unconstrained scale (as precision-weighted averages, or as a product of Gaussian approximations with `method="gaussian"`) and then transformed back, so constraints such as positivity are respected. The combination is exact when every subposterior is Gaussian.

It works well when each shard has enough data for its subposterior to be close to Gaussian, in cases like regression-type models with many rows per parameter. It breaks down for multilevel models, multimodal posteriors and discrete parameters. Check the per-shard posteriors with `return_subposteriors=True`. Only free variables are combined, so you need to recompute deterministics with `pm.compute_deterministics`. It samples with PyMC's own NUTS by default, because the combination needs the unconstrained draws, which nutpie does not return.

## Exact vs approximate, at a glance

| | `ShardedModel` | `consensus_sample` |
|---|---|---|
| Posterior | Exact | Approximate (exact for Gaussian subposteriors) |
| Communication | One round trip per leapfrog step | None during sampling |
| Sensitivity to network latency | High | None |
| Multilevel models | Exact | Biased: do not use |
| Multimodal posteriors | Exact (as far as NUTS mixes) | Unreliable |
| Scaling to many nodes | Bounded by round-trip latency | Embarrassingly parallel |

## Multilevel models

Multilevel models are where sampling gets slow, here are some helpers when fitting such models.

- **`ShardedModel` is exact for them.** A group's rows may span any number of shards.

- **Pass `shard_by="<group column>"` when there are many groups.** Every group then lives on a single shard, so each shard's gradient is zero for every other group's
  parameters, and only the nonzero entries travel back over the network. With thousands of groups this shrinks the per-step messages considerably. Results are identical either way.

- **Do not use `consensus_sample` for them.** Tempering the prior of the group-level parameters lets each shard absorb group variation into the local parameters, so every
  shard, and therefore the combination, underestimates the group-level standard deviations by roughly $1/\sqrt{n_\text{shards}}$.

- **Mind the parameterization.** NUTS makes one round trip per leapfrog step, so a model that needs fewer steps per draw samples proportionally faster. When groups have plenty of data each, the centered parameterization usually needs far fewer steps than the non-centered one.
