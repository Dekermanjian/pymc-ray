"""Sharding by group: each group on one shard, and sparse gradients back."""

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import ray

import pymc_ray as pmr
import pymc_ray.sharded as sharded_module
from pymc_ray._utils import materialize, split_data

from helpers import assert_same_density
from models import column, multilevel_model, simulate_multilevel


@pytest.fixture(scope="module")
def multilevel_data() -> pd.DataFrame:
    return simulate_multilevel(mean_group_size=60, seed=0)


def check_groups_kept_together(shards, data: pd.DataFrame, n_shards: int):
    groups = [set(column(shard, "group").tolist()) for shard in shards]
    assert len(shards) == n_shards
    assert sum(len(g) for g in groups) == data["group"].nunique()  # each group on one shard
    assert set().union(*groups) == set(data["group"])
    rows = [len(column(shard, "group")) for shard in shards]
    assert sum(rows) == len(data)
    assert max(rows) / min(rows) < 1.5


def test_split_keeps_groups_together(multilevel_data, as_backend):
    shards = split_data(as_backend(multilevel_data), 4, by=["group"])
    check_groups_kept_together(shards, multilevel_data, 4)


def test_split_dict_by_multiple_keys():
    rng = np.random.default_rng(0)
    data = {"a": rng.integers(0, 3, 600), "b": rng.integers(0, 4, 600), "y": rng.normal(size=600)}
    shards = split_data(data, 3, by=["a", "b"])
    keys = [set(zip(s["a"].tolist(), s["b"].tolist(), strict=True)) for s in shards]
    assert sum(len(k) for k in keys) == len(set(zip(data["a"], data["b"], strict=True)))
    assert sum(len(s["y"]) for s in shards) == 600


def test_split_ray_dataset_by_key(multilevel_data):
    dataset = ray.data.from_pandas(multilevel_data).repartition(3)
    shards = [materialize(s, "pandas") for s in split_data(dataset, 4, by=["group"])]
    check_groups_kept_together(shards, multilevel_data, 4)
    assert all("__pymc_ray_shard" not in shard.columns for shard in shards)


def test_split_rejects_bad_keys(multilevel_data):
    with pytest.raises(ValueError, match="not in the data"):
        split_data(multilevel_data, 2, by=["nope"])
    with pytest.raises(ValueError, match="groups into"):
        split_data(multilevel_data, 1000, by=["group"])
    with_nulls = multilevel_data.assign(group=multilevel_data["group"].astype("Int64"))
    with_nulls.loc[0, "group"] = pd.NA
    with pytest.raises(ValueError, match="nulls"):
        split_data(with_nulls, 2, by=["group"])


def test_shard_by_density_is_exact(multilevel_data):
    with pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=4, shard_by="group") as sm:
        assert sm.n_rows == len(multilevel_data)
        assert_same_density(sm.model, multilevel_model(multilevel_data))


def test_shard_by_through_relays_is_exact(multilevel_data):
    # Relays merge their shards' sparse gradients before replying.
    with pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=4, shard_by="group", relays=True) as sm:
        assert_same_density(sm.model, multilevel_model(multilevel_data))


def test_shards_return_sparse_gradients(multilevel_data):
    with pmr.ShardedModel(multilevel_model, multilevel_data, n_shards=4, shard_by="group") as sm:
        model = sm.model
        to_constrained = pm.compile(model.value_vars, model.replace_rvs_by_values(model.free_RVs), on_unused_input="ignore")
        point = model.initial_point()
        values = to_constrained(*(point[v.name] for v in model.value_vars))
        _, grads = ray.get(sm._actors[0].logp_dlogp.remote(values))
    z_grad = grads[[rv.name for rv in model.free_RVs].index("z")]
    assert isinstance(z_grad, sharded_module._SparseGrad)
    # This shard holds about a quarter of the groups, each with an intercept and a slope.
    assert z_grad.indices.size < 0.4 * multilevel_data["group"].nunique() * 2
