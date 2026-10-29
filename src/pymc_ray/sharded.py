"""Exact posterior inference for one model whose data are too large for one process.

The data are split into shards, each held by a Ray actor that compiles the model's
log-likelihood and its gradient for its own rows. On the driver, the model is built on
zero rows of data (so it contains only the priors) and a Potential is added whose value
is the sum of the shard log-likelihoods. Every logp/gradient evaluation NUTS makes is
fanned out to all actors and summed, so the sampler targets the exact full-data
posterior while the data stay distributed across the cluster.

Optionally, every shard actor is replicated. Each chain process (or thread) leases one
replica, so concurrent chains evaluate in parallel instead of queuing on the same actors.

Through Ray Client (a ``ray://`` address), this process is outside the cluster, so sampling
and benchmarking run in a task on the node of the session's driver (normally the head)
instead: chain processes started here could not reach the cluster, and every step would
cross the client connection.
"""

from __future__ import annotations

import logging
import multiprocessing
import os
import threading
import time
import uuid
import warnings
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import ray
import xarray as xr
from arviz_stats.utils import ELPDDataLOO
from pymc.model.transform.conditioning import remove_value_transforms
from pytensor.gradient import DisconnectedType
from pytensor.graph.basic import Apply
from pytensor.graph.op import Op
from pytensor.tensor.type import discrete_dtypes
from ray.util.client import ray as ray_client
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from pymc_ray import predictive
from pymc_ray._timing import time_logp_dlogp
from pymc_ray._utils import (
    Data,
    ModelFn,
    empty_like,
    ensure_ray_initialized,
    free_rv_shapes,
    is_ray_dataset,
    materialize,
    num_rows,
    split_with_positions,
    validate_likelihood_potentials,
)

_FORKED_PROCESS_ERROR = (
    "A pymc-ray sharded likelihood was evaluated in a forked process. Ray connections do "
    "not survive fork(), so sampling would deadlock. Use ShardedModel.sample(), or pass "
    "mp_ctx='spawn' to pm.sample."
)

# Ray settings for the processes a ShardedModel starts; see its ``env_vars`` parameter.
RAY_ENV_VARS = {
    # Otherwise ray.get can wait a full second for results that are already available,
    # which on some clusters stalls a large share of the steps.
    "RAY_get_check_signal_interval_milliseconds": "10",
    # Send parameters and gradients of up to 8 MiB inside the call itself; through the
    # object store, every message costs extra round trips between nodes.
    "RAY_max_direct_call_object_size": str(8 * 1024 * 1024),
}


def _via_ray_client() -> bool:
    """Whether this process reaches the cluster through Ray Client (a ``ray://`` address).

    Returns
    -------
    bool
        True when connected through Ray Client.
    """
    return ray_client.is_connected()


@ray.remote(num_cpus=0)
def _in_cluster(fn: Callable, *args) -> Any:
    """Run ``fn(*args)`` in a Ray task, for :meth:`ShardedModel._run_in_cluster`.

    Parameters
    ----------
    fn : callable
        The function to run.
    *args
        Positional arguments for ``fn``.

    Returns
    -------
    object
        The value returned by ``fn``.
    """
    # No CPUs reserved: by default the shard actors reserve all of them, and the chain
    # processes this starts mostly wait on the actors.
    return fn(*args)


def _sample(model: pm.Model, kwargs: dict[str, Any]) -> xr.DataTree:
    """Run ``pm.sample`` and drop the variables that the zero-row model computes as empty.

    Parameters
    ----------
    model : pm.Model
        The driver's model.
    kwargs : dict
        Keyword arguments for ``pm.sample``.

    Returns
    -------
    xarray.DataTree
        The sampling result, without empty variables (see :func:`_drop_empty`).
    """
    return _drop_empty(pm.sample(model=model, **kwargs))


@dataclass(frozen=True, slots=True)
class _SparseGrad:
    """The nonzero entries of a flattened gradient; every other entry is exactly zero.

    Attributes
    ----------
    indices : numpy.ndarray
        Positions of the nonzero entries in the flattened gradient, as int32.
    values : numpy.ndarray
        The nonzero entries.
    """

    indices: np.ndarray
    values: np.ndarray


def _compress(grad: np.ndarray) -> np.ndarray | _SparseGrad:
    """Send a mostly-zero gradient as its nonzero entries. Lossless.

    When the data are sharded by group, a shard's gradient is zero for the parameters of
    every group it does not hold.

    Parameters
    ----------
    grad : numpy.ndarray
        A gradient.

    Returns
    -------
    numpy.ndarray or _SparseGrad
        ``grad`` itself, or its nonzero entries when sending them takes fewer bytes.
    """
    flat = grad.reshape(-1)
    nonzero = np.flatnonzero(flat)
    if nonzero.size * 12 >= flat.nbytes:  # int32 index + float64 value per entry
        return grad
    return _SparseGrad(nonzero.astype(np.int32), flat[nonzero])


def _sum_results(values: Sequence[np.ndarray], results: Sequence[tuple[float, list]]) -> tuple[float, list[np.ndarray]]:
    """Total log-likelihood and dense gradients over shard (or relay) results.

    Parameters
    ----------
    values : sequence of numpy.ndarray
        The parameter values the results were computed at, which give the gradients'
        shapes.
    results : sequence of tuple
        ``(loglik, grads)`` from every shard actor or relay, where each gradient is dense
        or a :class:`_SparseGrad`.

    Returns
    -------
    loglik : float
        The summed log-likelihood.
    grads : list of numpy.ndarray
        The summed gradient with respect to each parameter, dense.
    """
    loglik = sum(part_loglik for part_loglik, _ in results)
    grads = []
    for i, value in enumerate(values):
        total = np.zeros(np.shape(value))
        flat = total.reshape(-1)
        for _, part_grads in results:
            grad = part_grads[i]
            if isinstance(grad, _SparseGrad):
                flat[grad.indices] += grad.values
            else:
                total += grad
        grads.append(total)
    return loglik, grads


@ray.remote
class _ShardActor:
    """Holds one shard of the data and evaluates its log-likelihood and gradient.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(data) -> pm.Model``, called on the shard's rows by :meth:`compile`.
    shard : object
        The shard's data, or a Ray Dataset shard to load as ``backend``.
    backend : str
        The dataframe library to load a Ray Dataset shard as.
    """

    def __init__(self, model_fn: ModelFn, shard: Any, backend: str):
        self._model_fn = model_fn
        self._data = materialize(shard, backend)
        self._fn = None
        self._calls = 0
        self._busy = 0.0

    def empty_data(self) -> Data:
        """Zero rows of the shard, with its columns and dtypes, to build the driver's model.

        Returns
        -------
        dataframe or dict of numpy.ndarray
            The empty shard.
        """
        return empty_like(self._data)

    def node_ip(self) -> str:
        """IP address of the node this actor runs on.

        Returns
        -------
        str
            The node's IP address.
        """
        return ray.util.get_node_ip_address()

    def node_id(self) -> str:
        """Ray ID of the node this actor runs on.

        Returns
        -------
        str
            The node ID, as ``NodeAffinitySchedulingStrategy`` takes it.
        """
        return ray.get_runtime_context().get_node_id()

    def compile(self, param_names: list[str], likelihood_potentials: tuple[str, ...]) -> tuple[dict[str, tuple[int, ...]], int]:
        """Build the shard's model and compile its log-likelihood and gradient.

        The function is compiled only when the shard's model has free random variables of
        the same names as the driver's; the caller compares the returned shapes and
        reports any mismatch. Once it is compiled, the shard's data are released.

        Parameters
        ----------
        param_names : list of str
            Names of the driver model's free random variables, in the order their values
            (on the constrained scale) are passed to :meth:`logp_dlogp`.
        likelihood_potentials : tuple of str
            Potentials that depend on the data, summed with the observed variables.

        Returns
        -------
        shapes : dict of str to tuple of int
            Shape of each free random variable of the shard's model, by name.
        n_rows : int
            Number of rows of the shard.
        """
        # Kept for posterior predictive sampling and LOO; it holds the shard's data.
        self._model = self._model_fn(self._data)
        shapes = free_rv_shapes(self._model)
        n_rows = num_rows(self._data)
        if set(shapes) != set(param_names):
            return shapes, n_rows

        # The driver passes parameters on their constrained scale, so differentiate the
        # likelihood with respect to untransformed values. Match them up by name: the
        # untransformed model does not preserve the order of the original value variables.
        model = remove_value_transforms(self._model)
        values = {value.name: value for value in model.value_vars}
        inputs = [values[name] for name in param_names]
        terms = [*model.observed_RVs]
        terms += [pot for pot in model.potentials if pot.name in likelihood_potentials]
        loglik = model.logp(vars=terms, jacobian=False)
        grads = pytensor.grad(loglik, inputs, disconnected_inputs="ignore")
        self._fn = pm.compile(inputs, [loglik, *grads], on_unused_input="ignore")
        # The models now hold the data they need.
        self._data = None
        return shapes, n_rows

    def run(self, work: Callable, *args) -> Any:
        """Run ``work(model, *args)`` on this shard's model (posterior predictive, LOO).

        Parameters
        ----------
        work : callable
            ``work(model, *args)``.
        *args
            Extra positional arguments for ``work``.

        Returns
        -------
        object
            The value returned by ``work``.
        """
        return work(self._model, *args)

    def logp_dlogp(self, values: Sequence[np.ndarray]) -> tuple[float, list[np.ndarray | _SparseGrad]]:
        """Log-likelihood of the shard's rows and its gradient at ``values``.

        Calls and compute time are recorded for :meth:`stats`.

        Parameters
        ----------
        values : sequence of numpy.ndarray
            Value of each parameter, on the constrained scale, in the order given to
            :meth:`compile`.

        Returns
        -------
        loglik : float
            The shard's log-likelihood.
        grads : list of numpy.ndarray or _SparseGrad
            Its gradient with respect to each parameter, compressed by :func:`_compress`.
        """
        start = time.perf_counter()
        loglik, *grads = self._fn(*values)
        self._busy += time.perf_counter() - start
        self._calls += 1
        return float(loglik), [_compress(grad) for grad in grads]

    def stats(self, reset: bool) -> dict[str, Any]:
        """Report the calls served and seconds spent in :meth:`logp_dlogp`, and the node.

        Parameters
        ----------
        reset : bool
            Reset the counts after reading them.

        Returns
        -------
        dict
            ``node`` (IP address), ``calls`` and ``compute_s``.
        """
        stats = {"node": self.node_ip(), "calls": self._calls, "compute_s": self._busy}
        if reset:
            self._calls, self._busy = 0, 0.0
        return stats


@ray.remote(num_cpus=0, max_concurrency=64)
class _NodeRelay:
    """Forwards a chain's request to the shard actors on its own node and sums their results.

    A chain then exchanges one message per node per step, instead of one per shard, and
    the shards' gradients are added up before they cross the network. Threaded, so that
    several chains can be in flight at once; each shard still serves one call at a time.

    Parameters
    ----------
    shards : list of ray.actor.ActorHandle
        The shard actors of one replica that run on this relay's node.
    """

    def __init__(self, shards: list[ray.actor.ActorHandle]):
        self._shards = shards

    def ready(self) -> bool:
        """Return True once the actor runs, so that its creation can be waited for.

        Returns
        -------
        bool
            Always True.
        """
        return True

    def logp_dlogp(self, values: Sequence[np.ndarray]) -> tuple[float, list[np.ndarray | _SparseGrad]]:
        """Total log-likelihood and gradient of this node's shards at ``values``.

        Parameters
        ----------
        values : sequence of numpy.ndarray
            Value of each parameter, on the constrained scale.

        Returns
        -------
        loglik : float
            The summed log-likelihood of the node's shards.
        grads : list of numpy.ndarray or _SparseGrad
            The summed gradient with respect to each parameter, compressed by
            :func:`_compress`.
        """
        results = ray.get([shard.logp_dlogp.remote(values) for shard in self._shards])
        loglik, grads = _sum_results(values, results)
        return loglik, [_compress(grad) for grad in grads]


@ray.remote(num_cpus=0)
class _ReplicaCounter:
    """Hands out replica indices round-robin, one per client process or thread."""

    def __init__(self):
        self._next = 0

    def next(self) -> int:
        """Return the next replica index.

        Returns
        -------
        int
            0, 1, 2, ... on successive calls; callers take it modulo the number of
            replicas.
        """
        self._next += 1
        return self._next - 1


class _ShardClient:
    """Picklable handle to the shard actors.

    PyMC pickles the step method into each chain process when sampling with
    ``mp_ctx="spawn"``. Actor handles cannot cross process boundaries, so the client
    pickles the actors' names and reconnects to the cluster on first use. Every process,
    and every thread within it, then leases one replica of the shards from a counter
    actor, round-robin, since the pickled client is identical in every chain process.

    Leases are keyed by OS thread id rather than kept in a ``threading.local``: samplers
    such as nutpie call in from non-Python threads, whose Python thread state (and with
    it any thread-local data) is recreated on every call.

    Parameters
    ----------
    actor_names : list of list of str
        Names of the actors that chains call, ``[replica][shard]``; with relays, the
        relays' names, ``[replica][node]``.
    counter_name : str or None
        Name of the :class:`_ReplicaCounter` actor; None with a single replica.
    namespace : str
        Ray namespace of the actors.
    address : str
        Address of the cluster's GCS, to connect to from chain processes.
    env_vars : dict of str to str
        Ray settings to put in the environment of a chain process before it connects,
        unless they are already set there.
    """

    _STATE = ("actor_names", "counter_name", "namespace", "address", "env_vars")

    def __init__(
        self,
        actor_names: list[list[str]],
        counter_name: str | None,
        namespace: str,
        address: str,
        env_vars: dict[str, str],
    ):
        self.actor_names = actor_names  # [replica][shard]
        self.counter_name = counter_name
        self.namespace = namespace
        self.address = address
        self.env_vars = env_vars
        self._reset()

    def _reset(self) -> None:
        """Forget all leases, on creation and after unpickling in another process."""
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._leases: dict[int, tuple[int, list[ray.actor.ActorHandle]]] = {}

    def __getstate__(self) -> dict[str, Any]:
        """State to pickle: the actors' names and the cluster's address, not the leases.

        Returns
        -------
        dict
            The attributes named in ``_STATE``.
        """
        return {name: getattr(self, name) for name in self._STATE}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the pickled state, without leases; they are taken again on first use.

        Parameters
        ----------
        state : dict
            The attributes from :meth:`__getstate__`.
        """
        self.__dict__.update(state)
        self._reset()

    @property
    def replica(self) -> int:
        """The replica leased by the calling thread."""
        return self._lease()[0]

    @property
    def actors(self) -> list[ray.actor.ActorHandle]:
        """The calling thread's replica: one actor per shard."""
        return self._lease()[1]

    def _lease(self) -> tuple[int, list[ray.actor.ActorHandle]]:
        """Get the calling thread's lease, taking it on first use.

        Connects this process to Ray if needed, with :attr:`env_vars` in its environment,
        takes the next replica from the counter actor (replica 0 without one) and looks
        up that replica's actors by name.

        Returns
        -------
        replica : int
            The leased replica's index.
        actors : list of ray.actor.ActorHandle
            Its actors: one per shard, or one per node with relays.

        Raises
        ------
        RuntimeError
            If called in a forked process, whose inherited Ray connection would deadlock.
        """
        if os.getpid() != self._pid:
            raise RuntimeError(_FORKED_PROCESS_ERROR)
        thread = threading.get_ident()
        lease = self._leases.get(thread)
        if lease is not None:
            return lease
        with self._lock:
            if not ray.is_initialized():
                # Ray reads its settings from the environment when it connects.
                for name, value in self.env_vars.items():
                    os.environ.setdefault(name, value)
                ray.init(
                    address=self.address,
                    namespace=self.namespace,
                    logging_level=logging.ERROR,
                    log_to_driver=False,
                )
        replica = 0
        if self.counter_name is not None:
            counter = ray.get_actor(self.counter_name, namespace=self.namespace)
            replica = ray.get(counter.next.remote()) % len(self.actor_names)
        names = self.actor_names[replica]
        lease = (replica, [ray.get_actor(n, namespace=self.namespace) for n in names])
        self._leases[thread] = lease
        return lease

    def logp_dlogp(self, values: Sequence[np.ndarray]) -> tuple[float, list[np.ndarray]]:
        """Total log-likelihood and gradient over all shards, from the leased replica.

        Parameters
        ----------
        values : sequence of array_like
            Value of each parameter, on the constrained scale.

        Returns
        -------
        loglik : float
            The total log-likelihood.
        grads : list of numpy.ndarray
            The total gradient with respect to each parameter, dense.
        """
        values = [np.asarray(value) for value in values]
        results = ray.get([actor.logp_dlogp.remote(values) for actor in self.actors])
        return _sum_results(values, results)


class ShardedLogLikOp(Op):
    """Total log-likelihood over all shards, and its gradient, evaluated on Ray actors.

    Outputs ``[loglik, *grads]``. The gradient outputs let the pullback reuse the values
    computed by the same call, so each NUTS leapfrog step contacts the actors only once.

    Parameters
    ----------
    client : _ShardClient
        Handle to the shard actors, pickled along with the Op into chain processes.
    """

    def __init__(self, client: _ShardClient):
        self.client = client

    def make_node(self, *params) -> Apply:
        """Apply the Op to the model's free random variables.

        Parameters
        ----------
        *params : TensorVariable
            The parameters, cast to float64.

        Returns
        -------
        pytensor.graph.basic.Apply
            Node with outputs ``[loglik, *grads]``: a scalar, then a gradient of the same
            type as each parameter.
        """
        params = [pt.as_tensor_variable(p).astype("float64") for p in params]
        return Apply(self, params, [pt.dscalar(), *(p.type() for p in params)])

    def perform(self, node, inputs, output_storage) -> None:
        """Evaluate the log-likelihood and its gradient on the shard actors.

        Parameters
        ----------
        node : pytensor.graph.basic.Apply
            The node being evaluated.
        inputs : list of numpy.ndarray
            The parameter values.
        output_storage : list of list
            Storage cells for ``[loglik, *grads]``.
        """
        loglik, grads = self.client.logp_dlogp(inputs)
        output_storage[0][0] = np.asarray(loglik, dtype="float64")
        for storage, grad in zip(output_storage[1:], grads, strict=True):
            storage[0] = np.asarray(grad, dtype="float64")

    def pullback(self, inputs, outputs, cotangents):
        """Gradient of the log-likelihood output, reusing the gradients already computed.

        Parameters
        ----------
        inputs : list of TensorVariable
            The parameters.
        outputs : list of TensorVariable
            The outputs ``[loglik, *grads]``.
        cotangents : list of TensorVariable
            Cotangents of the outputs; only the log-likelihood's may be connected.

        Returns
        -------
        list of TensorVariable
            The log-likelihood's cotangent times its gradient, for each parameter.

        Raises
        ------
        NotImplementedError
            If a gradient output is itself differentiated (a second-order gradient).
        """
        if not all(isinstance(ct.type, DisconnectedType) for ct in cotangents[1:]):
            raise NotImplementedError("Sharded likelihoods only provide first-order gradients")
        return [cotangents[0] * grad for grad in outputs[1:]]

    if not hasattr(Op, "pullback"):  # PyTensor < 3.3

        def L_op(self, inputs, outputs, output_grads):
            """Alias of :meth:`pullback` under its name before PyTensor 3.3.

            Parameters
            ----------
            inputs : list of TensorVariable
                The parameters.
            outputs : list of TensorVariable
                The outputs ``[loglik, *grads]``.
            output_grads : list of TensorVariable
                Cotangents of the outputs.

            Returns
            -------
            list of TensorVariable
                See :meth:`pullback`.
            """
            return self.pullback(inputs, outputs, output_grads)


try:
    from pytensor.link.numba.dispatch.basic import generate_fallback_impl, numba_funcify
except ImportError:  # PyTensor without the numba backend
    pass
else:

    @numba_funcify.register(ShardedLogLikOp)
    def _numba_funcify_sharded_loglik(op, node=None, storage_map=None, **kwargs):
        """Numba implementation of :class:`ShardedLogLikOp`: PyTensor's object-mode fallback.

        Parameters
        ----------
        op : ShardedLogLikOp
            The Op.
        node : pytensor.graph.basic.Apply, optional
            The node to implement.
        storage_map : dict, optional
            PyTensor's storage map.
        **kwargs
            Passed to ``generate_fallback_impl``.

        Returns
        -------
        callable
            The fallback implementation.
        """
        # The Op must run in Python (it calls Ray), so PyTensor's object-mode fallback is
        # the right implementation; only its warning is silenced. The shard likelihoods
        # themselves are compiled normally on the actors.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return generate_fallback_impl(op, node, storage_map, **kwargs)


class ShardedModel:
    """A PyMC model whose likelihood is evaluated over data shards held by Ray actors.

    Parameters
    ----------
    model_fn : callable
        ``model_fn(data) -> pm.Model``, called with one shard on each actor and with zero
        rows on the driver. Its free random variables (names and shapes) must not depend
        on the data: derive coords from global metadata, not from the rows it receives.
    data : dataframe, dict of arrays or ray.data.Dataset
        The full dataset: an eager dataframe from any library narwhals supports (pandas,
        polars, pyarrow, ...), a dict of equal-length arrays, or a Ray Dataset, which is
        split on the cluster with each shard loaded directly by its actor, so the data
        never need to fit on the driver.
    n_shards : int, optional
        Number of shards the data are split into. Defaults to the cluster's CPUs divided
        by ``num_cpus_per_shard * n_replicas``.
    n_replicas : int
        Number of copies of every shard actor. Each chain process (or thread) leases one
        replica, round-robin, so with as many replicas as chains, concurrent chains never
        queue on each other's actors. Costs ``n_replicas`` times the actors and memory.
        Compare layouts with :meth:`benchmark` and ``chains=``.
    num_cpus_per_shard : float
        CPUs reserved by each actor.
    likelihood_potentials : sequence of str
        Names of Potentials in ``model_fn`` that depend on the data (e.g. a custom
        likelihood). They are summed across shards along with the observed variables.
        All other Potentials are treated as part of the prior and evaluated once.
    potential_name : str
        Name of the Potential that carries the sharded log-likelihood.
    backend : str
        Ray Dataset input only: the dataframe library ``model_fn`` receives each shard
        as, by narwhals name (``"pandas"``, ``"polars"``, ``"pyarrow"``, ...).
    shard_by : str or sequence of str, optional
        Column(s) whose rows must stay together: every group goes to a single shard, with
        groups spread so that shards hold similar numbers of rows. Use the grouping
        column of a multilevel model. Each shard's gradient is then zero for the
        parameters of every group it does not hold, and only the nonzero entries are sent
        back, which cuts the return traffic of every step for models with many groups.
    actor_options : dict, optional
        Extra Ray options for every shard actor, passed to ``.options()`` (e.g.
        ``memory``, ``resources`` or ``scheduling_strategy``). By default actors are
        spread across nodes (``scheduling_strategy="SPREAD"``) so that the shards use
        the cores and memory bandwidth of every node, rather than filling one node first.
        A ``runtime_env`` given here is combined with ``env_vars``; its own environment
        variables take precedence.
    relays : bool, optional
        Route each chain's requests through one relay actor per node (and replica), which
        forwards them to the shards on its node and sums their gradients. A chain then
        sends one message per node per step instead of one per shard. Defaults to on
        when the shards span more than one node.
    env_vars : dict of str to str, optional
        Ray settings, as environment variables, for every process this model starts: the
        shard actors, the relays, the chain processes of PyMC's sampler, and the task that
        samples through Ray Client. They take precedence over the same variables in the
        job's ``runtime_env``. Defaults to::

            {
                "RAY_get_check_signal_interval_milliseconds": "10",
                "RAY_max_direct_call_object_size": "8388608",
            }

        The first stops ``ray.get`` from sometimes waiting a full second for results that
        are already available. The second sends parameters and gradients of up to 8 MiB
        inside each call, instead of through the object store, which costs extra round
        trips between nodes. Pass a dict to use other variables instead, or ``{}`` for
        Ray's own defaults. They do not reach this process, which is already connected:
        when chains run in it (nutpie, whose chains are threads, or PyMC's sampler with
        ``cores=1``) without Ray Client, set them in its environment before ``ray.init``.

    Examples
    --------
    .. code-block:: python

        def model_fn(df):
            with pm.Model() as model:
                beta = pm.Normal("beta", 0, 1, shape=3)
                sigma = pm.HalfNormal("sigma", 1)
                pm.Normal("y", df[["x0", "x1", "x2"]].to_numpy() @ beta, sigma, observed=df["y"])
            return model

        with ShardedModel(model_fn, big_df, n_shards=4, n_replicas=4) as sharded:
            idata = sharded.sample(draws=1000, chains=4)
    """

    def __init__(
        self,
        model_fn: ModelFn,
        data: Data,
        n_shards: int | None = None,
        *,
        n_replicas: int = 1,
        num_cpus_per_shard: float = 1,
        likelihood_potentials: Sequence[str] = (),
        potential_name: str = "sharded_loglik",
        backend: str = "pandas",
        actor_options: dict[str, Any] | None = None,
        shard_by: str | Sequence[str] | None = None,
        relays: bool | None = None,
        env_vars: dict[str, str] | None = None,
    ):
        if n_replicas < 1:
            raise ValueError(f"n_replicas must be at least 1, got {n_replicas}")
        self.env_vars = dict(RAY_ENV_VARS if env_vars is None else env_vars)
        ensure_ray_initialized()
        cluster_cpus = ray.cluster_resources().get("CPU", 1)
        if n_shards is None:
            n_shards = max(1, int(cluster_cpus // (num_cpus_per_shard * n_replicas)))
        if n_shards * n_replicas * num_cpus_per_shard > cluster_cpus:
            warnings.warn(
                f"{n_shards} shards x {n_replicas} replicas x {num_cpus_per_shard} CPUs "
                f"exceeds the cluster's {cluster_cpus:g} CPUs; actors that cannot be "
                "scheduled will block until the cluster scales up.",
                stacklevel=2,
            )
        likelihood_potentials = tuple(likelihood_potentials)

        context = ray.get_runtime_context()
        prefix = f"pymc-ray-shard-{uuid.uuid4().hex[:12]}"
        actor_names = [[f"{prefix}-r{r}-s{s}" for s in range(n_shards)] for r in range(n_replicas)]
        counter_name = f"{prefix}-counter" if n_replicas > 1 else None
        self.n_shards = n_shards
        self.n_replicas = n_replicas
        # Through Ray Client, sampling runs on the node of this session's driver on the
        # cluster (normally the head). That process creates and calls every actor, so it is
        # known to reach them all, and it is where the sampler would run if this process
        # were on the cluster.
        self._driver_node = context.get_node_id()
        self.n_rows = None
        self.relays = False
        self._relays: list[ray.actor.ActorHandle] = []
        model_fn_ref = ray.put(model_fn)
        # Store each shard once in the object store and let its replicas share it.
        by = [shard_by] if isinstance(shard_by, str) else shard_by
        shards, self._positions = split_with_positions(data, n_shards, by)
        # Without shard_by, Ray Dataset shards are contiguous and in order; with it, the
        # rows' original order is unknown.
        self._rows_ordered = self._positions is not None or by is None
        shards = [s if is_ray_dataset(s) else ray.put(s) for s in shards]
        options = {
            "num_cpus": num_cpus_per_shard,
            "scheduling_strategy": "SPREAD",
            **(actor_options or {}),
        }
        options["runtime_env"] = self._runtime_env(options.get("runtime_env"))
        self._actors = [
            _ShardActor.options(name=name, **options).remote(model_fn_ref, shard, backend)
            for replica_names in actor_names
            for name, shard in zip(replica_names, shards, strict=True)
        ]
        if counter_name is not None:
            self._actors.append(_ReplicaCounter.options(name=counter_name).remote())

        try:
            self.model = model_fn(ray.get(self._actors[0].empty_data.remote()))
            validate_likelihood_potentials(self.model, likelihood_potentials)
            expected = free_rv_shapes(self.model)
            if not expected:
                raise ValueError("The model has no free random variables")
            discrete = [rv.name for rv in self.model.free_RVs if rv.dtype in discrete_dtypes]
            if discrete:
                raise ValueError(f"ShardedModel supports continuous parameters only; {discrete} are discrete")
            shard_actors = self._actors[: n_shards * n_replicas]
            compiled = ray.get([a.compile.remote(list(expected), likelihood_potentials) for a in shard_actors])
            for i, (shapes, _) in enumerate(compiled[:n_shards]):
                _check_shard_parameters(i, expected, shapes)
            # Evaluate every actor once at the initial point: numba compiles the shard
            # functions now, in parallel, instead of inside the first sampling step, and
            # the compute statistics start clean.
            to_constrained = pm.compile(
                self.model.value_vars,
                self.model.replace_rvs_by_values(self.model.free_RVs),
                on_unused_input="ignore",
            )
            point = self.model.initial_point()
            values = to_constrained(*(point[value.name] for value in self.model.value_vars))
            ray.get([actor.logp_dlogp.remote(values) for actor in shard_actors])
            ray.get([actor.stats.remote(True) for actor in shard_actors])

            nodes = ray.get([actor.node_id.remote() for actor in shard_actors])
            self.relays = len(set(nodes)) > 1 if relays is None else relays
            if self.relays:
                actor_names = self._start_relays(prefix, shard_actors, nodes)
        except BaseException:
            self.shutdown()
            raise
        self._shard_rows = [n for _, n in compiled[:n_shards]]
        self.n_rows = sum(self._shard_rows)

        self._client = _ShardClient(actor_names, counter_name, context.namespace, context.gcs_address, self.env_vars)
        with self.model:
            loglik, *_ = ShardedLogLikOp(self._client)(*self.model.free_RVs)
            pm.Potential(potential_name, loglik)

    def _runtime_env(self, runtime_env: dict[str, Any] | None = None) -> dict[str, Any] | None:
        """Add :attr:`env_vars` to a Ray ``runtime_env`` for an actor or task this model starts.

        Ray merges it with the job's ``runtime_env``: its environment variables are added to
        the job's, replacing any of the same name, and every other field it leaves out is
        inherited.

        Parameters
        ----------
        runtime_env : dict, optional
            A ``runtime_env`` from the user; its own environment variables take precedence.

        Returns
        -------
        dict or None
            The combined ``runtime_env``, or None when it would be empty.
        """
        runtime_env = dict(runtime_env or {})
        env_vars = {**self.env_vars, **runtime_env.get("env_vars", {})}
        if env_vars:
            runtime_env["env_vars"] = env_vars
        return runtime_env or None

    def _start_relays(self, prefix: str, shard_actors: list[ray.actor.ActorHandle], nodes: list[str]) -> list[list[str]]:
        """Start one relay per (replica, node); return the relay names of each replica.

        Parameters
        ----------
        prefix : str
            Prefix of this model's actor names.
        shard_actors : list of ray.actor.ActorHandle
            Every shard actor, replica by replica.
        nodes : list of str
            The node ID of each shard actor.

        Returns
        -------
        list of list of str
            The relay names of each replica, ``[replica][node]``.
        """
        relay_names = []
        for replica in range(self.n_replicas):
            first = replica * self.n_shards
            shards_by_node: dict[str, list[ray.actor.ActorHandle]] = {}
            for shard, node in zip(
                shard_actors[first : first + self.n_shards],
                nodes[first : first + self.n_shards],
                strict=True,
            ):
                shards_by_node.setdefault(node, []).append(shard)
            names = [f"{prefix}-r{replica}-relay{i}" for i in range(len(shards_by_node))]
            for name, (node, shards) in zip(names, shards_by_node.items(), strict=True):
                on_node = NodeAffinitySchedulingStrategy(node, soft=False)
                relay = _NodeRelay.options(name=name, scheduling_strategy=on_node, runtime_env=self._runtime_env())
                self._relays.append(relay.remote(shards))
            relay_names.append(names)
        ray.get([relay.ready.remote() for relay in self._relays])
        return relay_names

    def _run_in_cluster(self, fn: Callable, *args) -> Any:
        """Run ``fn(*args)`` here, or, through Ray Client, in a task on the driver's node.

        Parameters
        ----------
        fn : callable
            The function to run.
        *args
            Positional arguments for ``fn``.

        Returns
        -------
        object
            The value returned by ``fn``.
        """
        if not _via_ray_client():
            return fn(*args)
        on_node = NodeAffinitySchedulingStrategy(self._driver_node, soft=False)
        ref = _in_cluster.options(scheduling_strategy=on_node, runtime_env=self._runtime_env()).remote(fn, *args)
        try:
            return ray.get(ref)
        except KeyboardInterrupt:
            ray.cancel(ref, force=True)
            raise

    def sample(self, **kwargs) -> Any:
        """Run ``pm.sample`` on :attr:`model`.

        With PyMC's own NUTS, chain processes are started with ``spawn``: forked
        processes inherit a Ray connection that no longer works and would deadlock.
        nutpie (PyMC's default sampler when it is installed, or ``nuts_sampler="nutpie"``)
        runs chains as threads in this process instead; each thread leases its own
        replica.

        Through Ray Client, sampling runs in a task on the node of the session's driver on
        the cluster (normally the head node), which can reach every shard, and only the
        result is sent back. The progress bar is off by default there, because it would
        print on the cluster. To keep every step on one node, place the shards on that
        node too (``actor_options``).

        Variables that the zero-row driver model computes as empty (deterministics that
        depend on the data, and the observed data) are dropped from the result, so that
        ``pm.sample_posterior_predictive`` on new data recomputes them. Recompute them on
        the training data with :meth:`sample_posterior_predictive` and ``var_names``.

        Parameters
        ----------
        **kwargs
            Passed to ``pm.sample``. ``mp_ctx`` defaults to ``"spawn"``, the only start
            method allowed.

        Returns
        -------
        xarray.DataTree
            The sampling result.

        Raises
        ------
        ValueError
            If ``mp_ctx`` is not a spawn context.
        """
        mp_ctx = kwargs.setdefault("mp_ctx", "spawn")
        if isinstance(mp_ctx, str):
            mp_ctx = multiprocessing.get_context(mp_ctx)
        if mp_ctx.get_start_method() != "spawn":
            raise ValueError("ShardedModel requires mp_ctx='spawn'; Ray does not survive fork()")
        if _via_ray_client():
            kwargs["mp_ctx"] = "spawn"  # a context object stays in this process
            kwargs.setdefault("progressbar", False)
        return self._run_in_cluster(_sample, self.model, kwargs)

    def _shards(self) -> predictive.ActorShards:
        """Gather the shard actors for posterior predictive sampling and LOO.

        Returns
        -------
        predictive.ActorShards
            The first replica's actors, with the layout of the data across them.
        """
        # Every replica holds the same rows; the first one is enough.
        actors = self._actors[: self.n_shards]
        return predictive.ActorShards(actors, self._positions, self._shard_rows, self._rows_ordered)

    def sample_posterior_predictive(
        self,
        idata: xr.DataTree | xr.Dataset,
        *,
        var_names: Sequence[str] | None = None,
        num_draws: int | None = None,
        random_seed: int | None = None,
        postprocess_fn: predictive.PostprocessFn | None = None,
        extend_inferencedata: bool = False,
    ) -> xr.DataTree | list[Any]:
        """Posterior predictive draws for every row, computed on the shard actors.

        Like ``pm.sample_posterior_predictive``: each actor samples its own rows, and the
        draws are put back in the order of the data.

        Parameters
        ----------
        idata : xarray.DataTree or xarray.Dataset
            A posterior, e.g. from :meth:`sample`. Only its free variables are used.
        var_names : sequence of str, optional
            Variables to sample. Defaults to the observed variables. Each must have one
            entry per row of the data along its first non-sample dimension.
        num_draws : int, optional
            Use this many evenly spaced draws per chain. The result holds
            ``chains x num_draws x rows`` values for every variable.
        random_seed : int, optional
            Seed for reproducible draws; every shard gets an independent stream.
        postprocess_fn : callable, optional
            ``postprocess_fn(shard, model, result) -> Any``, run on the actor with the
            shard's index, its model and its ``DataTree`` (``posterior_predictive`` and
            ``observed_data`` for that shard's rows). Use it to write each shard's draws
            to storage and return the paths when the full result would not fit in memory.
            A list of the returned values, in shard order, is returned instead.
        extend_inferencedata : bool
            Add the groups to ``idata`` (a ``DataTree``) and return it.

        Returns
        -------
        xarray.DataTree or list
            ``posterior_predictive`` and ``observed_data`` groups, with rows in the order
            of the data; or the list of ``postprocess_fn`` results.
        """
        return predictive._run_posterior_predictive(
            self._shards(),
            idata,
            var_names=var_names,
            num_draws=num_draws,
            random_seed=random_seed,
            postprocess_fn=postprocess_fn,
            extend_inferencedata=extend_inferencedata,
        )

    def compute_log_likelihood(
        self,
        idata: xr.DataTree | xr.Dataset,
        *,
        var_names: Sequence[str] | None = None,
        num_draws: int | None = None,
        extend_inferencedata: bool = True,
    ) -> xr.DataTree:
        """Pointwise log-likelihood of every row, computed on the shard actors.

        Like ``pm.compute_log_likelihood``. The result holds ``chains x draws x rows``
        values per observed variable; for LOO on data too large for that, use :meth:`loo`.

        Parameters
        ----------
        idata : xarray.DataTree or xarray.Dataset
            A posterior, e.g. from :meth:`sample`. Only its free variables are used.
        var_names : sequence of str, optional
            Observed variables to compute. Defaults to all of them.
        num_draws : int, optional
            Use this many evenly spaced draws per chain.
        extend_inferencedata : bool
            Add a ``log_likelihood`` group to ``idata`` (a ``DataTree``) and return it.

        Returns
        -------
        xarray.DataTree
            ``idata`` with a ``log_likelihood`` group, with rows in the order of the data;
            or, without ``extend_inferencedata`` (or for a ``Dataset``), a new tree
            holding only that group.
        """
        return predictive._run_log_likelihood(
            self._shards(),
            idata,
            var_names=var_names,
            num_draws=num_draws,
            extend_inferencedata=extend_inferencedata,
        )

    def loo(
        self,
        idata: xr.DataTree | xr.Dataset,
        *,
        var_name: str | None = None,
        reff: float | None = None,
    ) -> ELPDDataLOO:
        """PSIS-LOO cross-validation, computed on the shard actors.

        Each actor computes the pointwise log-likelihood of its rows and their PSIS-LOO
        values, and returns only one value per row, so the full log-likelihood array never
        has to fit anywhere. The result equals ``arviz.loo`` on the full data and works with
        ``arviz.compare``.

        Parameters
        ----------
        idata : xarray.DataTree or xarray.Dataset
            A posterior, e.g. from :meth:`sample`.
        var_name : str, optional
            Observed variable to cross-validate. Required when there are several.
        reff : float, optional
            Relative MCMC efficiency. Defaults to the value ``arviz.loo`` would compute.

        Returns
        -------
        arviz_stats.utils.ELPDDataLOO
            The LOO estimate, with pointwise ``elpd_i`` and ``pareto_k`` in the order of
            the data.

        Warns
        -----
        UserWarning
            If a Pareto shape estimate exceeds the threshold for reliable importance
            sampling.
        """
        return predictive._run_loo(self._shards(), idata, var_name=var_name, reff=reff)

    def shard_stats(self, reset: bool = False) -> list[dict[str, Any]]:
        """Per shard actor: its node, and the calls served and seconds spent computing.

        Counts accumulate from creation or from the last ``reset=True``. If a step takes
        much longer than the actors' compute per call (times the chains sharing each
        actor), the rest of the time goes to communication.

        Parameters
        ----------
        reset : bool
            Reset the counts after reading them.

        Returns
        -------
        list of dict
            For each shard actor, replica by replica: ``node`` (IP address), ``calls``
            and ``compute_s``.
        """
        shard_actors = self._actors[: self.n_shards * self.n_replicas]
        return ray.get([actor.stats.remote(reset) for actor in shard_actors])

    def placement(self) -> dict[str, int]:
        """Count the shard actors on each node, keyed by the node's IP address.

        Returns
        -------
        dict of str to int
            Number of shard actors, over all replicas, on each node.
        """
        shard_actors = self._actors[: self.n_shards * self.n_replicas]
        return dict(Counter(ray.get([actor.node_ip.remote() for actor in shard_actors])))

    def benchmark(self, n_evals: int = 20, chains: int = 1) -> float:
        """Mean wall time, in seconds, of one joint logp and gradient evaluation.

        With ``chains > 1``, that many spawned processes evaluate concurrently, as chains
        do while sampling. Compare with :func:`time_logp_dlogp` on the in-process model,
        with the same ``chains``, to decide whether sharding pays off and to choose
        ``n_shards`` and ``n_replicas``. Through Ray Client, it runs on the driver's node, as
        :meth:`sample` does.

        Parameters
        ----------
        n_evals : int
            Number of timed evaluations per process.
        chains : int
            Number of processes that evaluate concurrently.

        Returns
        -------
        float
            Mean seconds per evaluation, averaged over processes when ``chains > 1``.
        """
        return self._run_in_cluster(time_logp_dlogp, self.model, n_evals, chains)

    def shutdown(self) -> None:
        """Terminate the shard actors and release their memory."""
        for actor in [*self._relays, *self._actors]:
            ray.kill(actor)
        self._actors, self._relays = [], []

    def __enter__(self) -> ShardedModel:
        """Enter the context; the shard actors are shut down when it exits.

        Returns
        -------
        ShardedModel
            This model.
        """
        return self

    def __exit__(self, *exc_info) -> None:
        """Shut down the shard actors when the context exits.

        Parameters
        ----------
        *exc_info
            The exception type, value and traceback, if any; an exception is not
            suppressed.
        """
        self.shutdown()

    def __repr__(self) -> str:
        """Shard layout and size of the model.

        Returns
        -------
        str
            The representation, with ``n_shards``, ``n_replicas``, ``n_rows`` and ``relays``.
        """
        return f"ShardedModel(n_shards={self.n_shards}, n_replicas={self.n_replicas}, n_rows={self.n_rows}, relays={self.relays})"


def _drop_empty(idata: xr.DataTree) -> xr.DataTree:
    """Drop what the zero-row driver model computed as empty.

    Deterministics that depend on the data (one value per row, say) and the observed data
    itself have no rows on the driver. Left in the posterior, an empty deterministic would
    be reused, instead of recomputed, by ``pm.sample_posterior_predictive`` on new data.

    Parameters
    ----------
    idata : xarray.DataTree
        A sampling result of the driver's model. It may be modified in place.

    Returns
    -------
    xarray.DataTree
        The result without empty variables, and without the groups that held only those.
    """
    for name in list(idata.children):
        group = idata[name].to_dataset()
        empty = [var for var in group.variables if group[var].size == 0]
        if not empty:
            continue
        group = group.drop_vars(empty)
        if group.data_vars:
            idata[name] = group
        else:
            idata = idata.drop_nodes(name)
    return idata


def _check_shard_parameters(shard: int, expected: dict[str, tuple[int, ...]], found: dict[str, tuple[int, ...]]) -> None:
    """Check that a shard's model has the same free random variables as the driver's.

    Parameters
    ----------
    shard : int
        Index of the shard, for the error message.
    expected : dict of str to tuple of int
        Shape of each free random variable of the driver's (zero-row) model.
    found : dict of str to tuple of int
        Shape of each free random variable of the shard's model.

    Raises
    ------
    ValueError
        If the names or shapes differ; the message lists every difference.
    """
    if found == expected:
        return
    lines = [f"Shard {shard} builds a model with different free random variables than the driver:"]
    for name in sorted(expected.keys() | found.keys()):
        if expected.get(name) != found.get(name):
            driver = expected.get(name, "missing")
            lines.append(f"  {name}: driver (zero rows) {driver}, shard {found.get(name, 'missing')}")
    lines.append(
        "The parameters of model_fn must not depend on the rows it receives. Build coords "
        "from global metadata (e.g. all group labels), not from the data passed in."
    )
    raise ValueError("\n".join(lines))
