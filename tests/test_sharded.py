import multiprocessing
import os
import threading
import time
import warnings

import cloudpickle
import pytest
import ray
import xarray as xr

import pymc_ray as pmr
from pymc_ray import _timing as timing_module
from pymc_ray import sharded as sharded_module

from helpers import assert_same_density
from models import (
    data_dependent_coords_model,
    discrete_parameter_model,
    hierarchical_model,
    potential_likelihood_model,
    regression_model,
    regression_model_dict,
)


def test_matches_full_data_model(regression_data, as_backend):
    data = as_backend(regression_data)
    with pmr.ShardedModel(regression_model, data, n_shards=3) as sharded:
        assert sharded.n_rows == len(regression_data)
        assert_same_density(sharded.model, regression_model(data))


def test_hierarchical_model_with_global_coords(regression_data):
    with pmr.ShardedModel(hierarchical_model, regression_data, n_shards=4) as sharded:
        assert_same_density(sharded.model, hierarchical_model(regression_data))


def test_dict_input(regression_data):
    data = {"X": regression_data[["x0", "x1"]].to_numpy(), "y": regression_data["y"].to_numpy()}
    with pmr.ShardedModel(regression_model_dict, data, n_shards=2) as sharded:
        assert_same_density(sharded.model, regression_model_dict(data))


@pytest.mark.parametrize("backend", ["pandas", "polars"])
def test_ray_dataset_input(regression_data, backend):
    dataset = ray.data.from_pandas(regression_data).repartition(5)
    with pmr.ShardedModel(regression_model, dataset, n_shards=3, backend=backend) as sharded:
        assert sharded.n_rows == len(regression_data)
        assert_same_density(sharded.model, regression_model(regression_data))


def test_likelihood_potentials(regression_data):
    with pmr.ShardedModel(potential_likelihood_model, regression_data, n_shards=3, likelihood_potentials=["loglik"]) as sharded:
        assert_same_density(sharded.model, potential_likelihood_model(regression_data))


def test_unknown_likelihood_potential_raises(regression_data):
    with pytest.raises(ValueError, match="not Potentials"):
        pmr.ShardedModel(regression_model, regression_data, n_shards=2, likelihood_potentials=["nope"])


def test_data_dependent_parameters_raise(regression_data):
    with pytest.raises(ValueError, match="must not depend on the rows"):
        pmr.ShardedModel(data_dependent_coords_model, regression_data, n_shards=2)


def test_discrete_parameters_raise(regression_data):
    with pytest.raises(ValueError, match=r"continuous parameters only; \['k'\]"):
        pmr.ShardedModel(discrete_parameter_model, regression_data, n_shards=2)


def test_model_survives_pickling(regression_data):
    """Chain processes receive a pickled model and must reconnect to the actors."""
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded:
        restored = cloudpickle.loads(cloudpickle.dumps(sharded.model))
        assert_same_density(restored, regression_model(regression_data))


def test_sample_recovers_parameters(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded:
        idata = sharded.sample(draws=300, tune=300, chains=1, cores=1, random_seed=3, progressbar=False)
    posterior = idata["posterior"]
    # Data are simulated with group offsets that this model ignores; those inflate sigma.
    assert float(posterior["beta"].mean(("chain", "draw"))[0]) == pytest.approx(1.0, abs=0.1)
    assert float(posterior["beta"].mean(("chain", "draw"))[1]) == pytest.approx(-2.0, abs=0.1)


def test_ray_client_samples_in_the_cluster(regression_data, monkeypatch):
    kwargs = {"draws": 50, "tune": 50, "chains": 1, "cores": 1, "random_seed": 3}
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded:
        here = sharded.sample(progressbar=False, nuts_sampler="pymc", **kwargs)

        def run_here(*args, **kwargs):
            raise AssertionError("ran in the Ray Client process instead of the cluster")

        # Through Ray Client, sampling and benchmarks must run in a task in the cluster,
        # where these functions are not patched.
        monkeypatch.setattr(sharded_module, "_via_ray_client", lambda: True)
        monkeypatch.setattr(sharded_module.pm, "sample", run_here)
        monkeypatch.setattr(timing_module, "_compile_logp_dlogp", run_here)
        in_cluster = sharded.sample(nuts_sampler="pymc", **kwargs)
        assert sharded.benchmark(n_evals=2) > 0
    xr.testing.assert_equal(here["posterior"].to_dataset(), in_cluster["posterior"].to_dataset())


def _read_env(*_) -> dict[str, str | None]:
    names = [*sharded_module.RAY_ENV_VARS, "PYMC_RAY_A", "PYMC_RAY_B", "PYTHONPATH"]
    return {name: os.environ.get(name) for name in names}


def test_ray_settings_reach_actors_and_the_ray_client_task(regression_data, monkeypatch):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded:
        on_actor = ray.get(sharded._actors[0].run.remote(_read_env))
        monkeypatch.setattr(sharded_module, "_via_ray_client", lambda: True)
        in_task = sharded._run_in_cluster(_read_env)
    for env in (on_actor, in_task):
        for name, value in sharded_module.RAY_ENV_VARS.items():
            assert env[name] == value
        # The job's runtime_env (conftest.py) is still inherited.
        assert os.path.dirname(__file__) in env["PYTHONPATH"]


def test_env_vars_replace_the_defaults_and_actor_options_win(regression_data):
    with pmr.ShardedModel(
        regression_model,
        regression_data,
        n_shards=1,
        env_vars={"PYMC_RAY_A": "model", "PYMC_RAY_B": "model"},
        actor_options={"runtime_env": {"env_vars": {"PYMC_RAY_B": "actor"}}},
    ) as sharded:
        env = ray.get(sharded._actors[0].run.remote(_read_env))
        assert sharded._client.env_vars == {"PYMC_RAY_A": "model", "PYMC_RAY_B": "model"}
    assert (env["PYMC_RAY_A"], env["PYMC_RAY_B"]) == ("model", "actor")


def _lease_and_read_env(client) -> dict[str, str | None]:
    client._lease()
    return _read_env()


@pytest.mark.slow
def test_chain_processes_get_env_vars_before_connecting(regression_data):
    with (
        pmr.ShardedModel(regression_model, regression_data, n_shards=1, env_vars={"PYMC_RAY_A": "model"}) as sharded,
        multiprocessing.get_context("spawn").Pool(1) as pool,
    ):
        env = pool.apply(_lease_and_read_env, (sharded._client,))
    assert env["PYMC_RAY_A"] == "model"


def _assert_same_density_in_this_process(model_bytes: bytes, model_fn, data) -> None:
    assert_same_density(cloudpickle.loads(model_bytes), model_fn(data))


@pytest.fixture(scope="module")
def chain_process():
    """A spawned process that stands in for a PyMC chain process. It is shared because starting one is slow."""
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        yield pool


@pytest.mark.slow
@pytest.mark.parametrize(
    "setup",
    [
        pytest.param({"n_shards": 2}, id="shards"),
        pytest.param({"n_shards": 1, "n_replicas": 2}, id="replicas"),
        pytest.param({"n_shards": 2, "n_replicas": 2, "relays": True}, id="relays"),
    ],
)
def test_chain_process_density_is_exact(regression_data, setup, chain_process):
    """A spawned chain process must compute exactly the full-data density.

    The multiprocess sampling tests only check that the chains come back. This test checks the values
    that a chain process computes, so a chain that reached the wrong actors or lost a shard fails it.
    """
    with pmr.ShardedModel(regression_model, regression_data, **setup) as sharded:
        # Checking here first takes replica 0, so with replicas the chain process leases replica 1.
        assert_same_density(sharded.model, regression_model(regression_data))
        # PyMC pickles the step method, which holds the sharded model's client, into each chain process.
        model_bytes = cloudpickle.dumps(sharded.model)
        chain_process.apply(_assert_same_density_in_this_process, (model_bytes, regression_model, regression_data))


@pytest.mark.slow
def test_sample_multiprocess(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded:
        idata = sharded.sample(
            draws=100,
            tune=100,
            chains=2,
            cores=2,
            random_seed=3,
            progressbar=False,
            nuts_sampler="pymc",
        )
    assert idata["posterior"].sizes["chain"] == 2


def test_fork_is_rejected(regression_data):
    with (
        pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded,
        pytest.raises(ValueError, match="spawn"),
    ):
        sharded.sample(mp_ctx="fork")


def test_shutdown_kills_actors(regression_data):
    sharded = pmr.ShardedModel(regression_model, regression_data, n_shards=2)
    actors = list(sharded._actors)
    assert sharded.benchmark(n_evals=3) > 0
    sharded.shutdown()
    # ray.kill is asynchronous: a call made right after it can still be served.
    with pytest.raises(ray.exceptions.RayActorError):
        for _ in range(100):
            ray.get(actors[0].empty_data.remote())
            time.sleep(0.1)


def test_replicas_match_full_data_model(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2, n_replicas=2) as sharded:
        assert sharded.n_rows == len(regression_data)
        assert_same_density(sharded.model, regression_model(regression_data))


def test_replicas_are_leased_round_robin(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=1, n_replicas=3) as sharded:
        # Each unpickled client stands in for one chain process.
        clients = [cloudpickle.loads(cloudpickle.dumps(sharded._client)) for _ in range(6)]
        assert [client.replica for client in clients] == [0, 1, 2, 0, 1, 2]


def test_each_thread_leases_its_own_replica(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=1, n_replicas=2) as sharded:
        client = cloudpickle.loads(cloudpickle.dumps(sharded._client))
        leased = []
        threads = [threading.Thread(target=lambda: leased.append(client.replica)) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sorted(leased) == [0, 1]


def test_invalid_replicas_raise(regression_data):
    with pytest.raises(ValueError, match="n_replicas"):
        pmr.ShardedModel(regression_model, regression_data, n_shards=1, n_replicas=0)


@pytest.mark.slow
def test_sample_multiprocess_with_replicas(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=1, n_replicas=2) as sharded:
        idata = sharded.sample(
            draws=100,
            tune=100,
            chains=2,
            cores=2,
            random_seed=3,
            progressbar=False,
            nuts_sampler="pymc",
        )
        # One lease for the driver (initial checks) and one per chain process.
        counter = ray.get_actor(sharded._client.counter_name, namespace=sharded._client.namespace)
        assert ray.get(counter.next.remote()) == 3
    assert idata["posterior"].sizes["chain"] == 2


@pytest.mark.slow
def test_benchmark_with_concurrent_chains(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=1, n_replicas=2) as sharded:
        assert sharded.benchmark(n_evals=5, chains=2) > 0


def test_placement_and_actor_options(regression_data):
    with pmr.ShardedModel(
        regression_model,
        regression_data,
        n_shards=2,
        n_replicas=2,
        actor_options={"num_cpus": 0.5},
    ) as sharded:
        assert sum(sharded.placement().values()) == 4
        assert_same_density(sharded.model, regression_model(regression_data))


def test_shard_stats_count_calls_and_compute(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded:
        sharded.shard_stats(reset=True)
        sharded.benchmark(n_evals=5)
        stats = sharded.shard_stats()
    assert [s["calls"] for s in stats] == [6, 6]  # one warm-up call, then five timed
    assert all(s["compute_s"] > 0 for s in stats)


def test_no_object_mode_warning(regression_data):
    with (
        pmr.ShardedModel(regression_model, regression_data, n_shards=1) as sharded,
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        sharded.model.compile_logp()(sharded.model.initial_point())
    assert not [w for w in caught if "object mode" in str(w.message)]


def test_relays_match_full_data_model(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2, n_replicas=2, relays=True) as sharded:
        assert sharded.relays
        assert len(sharded._relays) == 2  # one per replica on this single node
        assert_same_density(sharded.model, regression_model(regression_data))


def test_relays_default_off_on_one_node(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2) as sharded:
        assert not sharded.relays
        assert not sharded._relays


def test_shutdown_kills_relays(regression_data):
    sharded = pmr.ShardedModel(regression_model, regression_data, n_shards=2, relays=True)
    relay = sharded._relays[0]
    sharded.shutdown()
    # ray.kill is asynchronous: a call made right after it can still be served.
    with pytest.raises(ray.exceptions.RayActorError):
        for _ in range(100):
            ray.get(relay.ready.remote())
            time.sleep(0.1)


@pytest.mark.slow
def test_sample_multiprocess_with_relays(regression_data):
    with pmr.ShardedModel(regression_model, regression_data, n_shards=2, n_replicas=2, relays=True) as sharded:
        idata = sharded.sample(
            draws=100,
            tune=100,
            chains=2,
            cores=2,
            random_seed=3,
            progressbar=False,
            nuts_sampler="pymc",
        )
    assert idata["posterior"].sizes["chain"] == 2


def test_nutpie_leases_one_replica_per_thread(regression_data):
    """nutpie runs chains on Rust threads, which re-enter Python on every call."""
    pytest.importorskip("nutpie")
    with pmr.ShardedModel(regression_model, regression_data, n_shards=1, n_replicas=2) as sharded:
        idata = sharded.sample(nuts_sampler="nutpie", draws=100, tune=100, chains=2, random_seed=1, progressbar=False)
        counter = ray.get_actor(sharded._client.counter_name, namespace=sharded._client.namespace)
        leases = ray.get(counter.next.remote())
    assert idata["posterior"].sizes["chain"] == 2
    # A handful of threads, not one lease per gradient evaluation (hundreds here).
    assert leases < 10
