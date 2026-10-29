"""Does ShardedModel beat in-process sampling on a multi-node Ray cluster?

Submit it as a Ray job from the project root, so that dependencies are installed into an
isolated environment on every node rather than into the cluster's own Python:

    ray job submit --address http://<head-node>:8265 --working-dir benchmarks \\
        --runtime-env-json '{"pip": ["pymc==6.3.2", "narwhals>=2", "polars"],
                             "py_modules": ["src/pymc_ray"]}' \\
        -- python cluster_scaling.py --rows 5000000 --sample

Uploading only benchmarks/ keeps the upload small, and avoids Ray < 2.59 crashing on
unusual .gitignore files elsewhere in the tree.

or run it directly on the head node of a cluster whose Python already has pymc_ray and
its dependencies (it then ships this checkout of pymc_ray to the workers itself).

For the multilevel model in gradient_eval.py it reports the wall time of one joint
logp+gradient evaluation (one NUTS leapfrog step) while ``--chains`` chains evaluate
concurrently:

- in-process: plain PyMC on a single worker node. It runs as a Ray task pinned to that
  node, so it gets the same hardware as the shards rather than the head node's.
- every SHARDSxREPLICAS layout of ShardedModel, with the shard actors spread across nodes.

With ``--sample``, it also times a short end-to-end ``pm.sample`` run for the in-process
baseline and for the fastest layout.
"""

import argparse
import os
import time

import gradient_eval
import numpy as np
import pymc as pm
import ray
from gradient_eval import multilevel_model_fn, simulate_multilevel
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import pymc_ray as pmr


@ray.remote
def in_process_timing(model_fn, data, n_evals: int, chains: int) -> float:
    return pmr.time_logp_dlogp(model_fn(data), n_evals, chains)


@ray.remote
def in_process_sampling(model_fn, data, chains: int, draws: int) -> tuple[float, float]:
    start = time.perf_counter()
    idata = pm.sample(
        model=model_fn(data),
        draws=draws,
        tune=draws,
        chains=chains,
        cores=chains,
        mp_ctx="spawn",
        progressbar=False,
        compute_convergence_checks=False,
    )
    return time.perf_counter() - start, float(idata["sample_stats"]["n_steps"].mean())


def pick_worker_node() -> dict:
    """The worker node with the most CPUs (the head node if there are no workers)."""
    head = ray.get_runtime_context().get_node_id()
    nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU", 0) > 0]
    workers = [n for n in nodes if n["NodeID"] != head] or nodes
    return max(workers, key=lambda n: n["Resources"]["CPU"])


def auto_layouts(cpus_per_node: list[float], chains: int) -> list[str]:
    """Shards per node doubling up to 3/4 of a node's CPUs, plus one layout with a replica
    per chain. Across nodes, 1-16 per node (every shard a chain contacts adds per-step
    overhead); on a single node, 4-48."""
    n_nodes, per_node = len(cpus_per_node), int(min(cpus_per_node))
    candidates = (1, 2, 4, 8, 16) if n_nodes > 1 else (4, 8, 16, 32, 48)
    widths = [k for k in candidates if k <= max(1, 3 * per_node // 4)] or [1]
    layouts = [f"{n_nodes * k}x1" for k in widths]
    replica_shards = 2 * n_nodes if n_nodes > 1 else max(1, 3 * per_node // (8 * chains))
    layouts.append(f"{replica_shards}x{chains}")
    return layouts


def describe_cluster() -> None:
    head = ray.get_runtime_context().get_node_id()
    print("Cluster nodes:")
    for node in ray.nodes():
        if node["Alive"]:
            role = "head" if node["NodeID"] == head else "worker"
            cpus = node["Resources"].get("CPU", 0)
            memory = node["Resources"].get("memory", 0) / 1e9
            print(f"  {node['NodeManagerAddress']:>15}  {role:6}  {cpus:5g} CPUs  {memory:6.1f} GB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto", help="Ray cluster address")
    parser.add_argument("--rows", type=int, default=5_000_000)
    parser.add_argument("--groups", type=int, help="number of groups (default: rows / 1000)")
    parser.add_argument("--shard-by", help="keep this column's groups on one shard each (group)")
    parser.add_argument(
        "--relays",
        choices=["auto", "on", "off"],
        default="auto",
        help="one relay per node (auto: on when shards span several nodes)",
    )
    parser.add_argument(
        "--single-node",
        action="store_true",
        help="put every shard (and the baseline) on this process's node",
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--layouts", nargs="+", default=["auto"], help="SHARDSxREPLICAS, or auto from cluster size")
    parser.add_argument("--evals", type=int, default=30)
    parser.add_argument("--sample", action="store_true", help="also time end-to-end sampling")
    parser.add_argument("--draws", type=int, default=200, help="draws (and tuning steps)")
    args = parser.parse_args()
    relays = {"auto": None, "on": True, "off": False}[args.relays]

    # A job from `ray job submit` already ships pymc_ray and its dependencies to every node,
    # and Ray rejects a second runtime_env inside it. Otherwise ship this checkout ourselves.
    in_submitted_job = "RAY_JOB_CONFIG_JSON_ENV_VAR" in os.environ
    runtime_env = None if in_submitted_job else {"py_modules": [pmr]}
    ray.init(address=args.address, runtime_env=runtime_env, log_to_driver=False)
    # Model functions travel to the workers by value.
    ray.cloudpickle.register_pickle_by_value(gradient_eval)
    describe_cluster()

    data = simulate_multilevel(args.rows, n_groups=args.groups)
    model_fn = multilevel_model_fn(args.rows, n_groups=args.groups)
    data_ref = ray.put(data)
    if args.single_node:
        here = ray.get_runtime_context().get_node_id()
        worker = next(n for n in ray.nodes() if n["NodeID"] == here)
    else:
        worker = pick_worker_node()
    on_worker = NodeAffinitySchedulingStrategy(worker["NodeID"], soft=False)
    # Pin the shards to the baseline's node, or let them spread over the cluster.
    actor_options = {"scheduling_strategy": on_worker} if args.single_node else None
    sharding = f", sharded by {args.shard_by!r}" if args.shard_by else ""
    print(f"\nMultilevel model, {args.rows:,} rows, {args.chains} concurrent chains{sharding}")
    print(f"In-process baseline runs on {worker['NodeManagerAddress']}\n")

    baseline = ray.get(
        in_process_timing.options(num_cpus=args.chains, scheduling_strategy=on_worker).remote(model_fn, data_ref, args.evals, args.chains)
    )
    n_params = sum(np.size(v) for v in model_fn(data.head(0)).initial_point().values())
    print(
        f"Each step, every chain sends {n_params:,} parameters ({n_params * 8 / 1024:,.0f} KB) "
        "to each shard it uses and receives a gradient of the same size from it "
        "(with --shard-by, only the gradient's nonzero entries)."
    )
    print(
        "'compute' is the actors' own time per call (mean / slowest node). With R replicas, "
        f"up to {args.chains}/R chains queue on each actor, so a step needs at least "
        "(chains / R) x the slowest compute; the rest of a step is communication.\n"
    )
    header = f"{'layout':>10} {'per step':>10} {'vs in-proc':>11} {'compute (ms)':>16}"
    print(f"{header}   actors per node")
    print(f"{'in-process':>10} {baseline * 1e3:>7.1f} ms {'1.00x':>11}")

    timings = {}
    cpus = [n["Resources"]["CPU"] for n in ray.nodes() if n["Alive"] and n["Resources"].get("CPU")]
    if args.single_node:
        cpus = [worker["Resources"]["CPU"]]
    layouts = auto_layouts(cpus, args.chains) if args.layouts == ["auto"] else args.layouts
    for layout in layouts:
        n_shards, n_replicas = (int(n) for n in layout.split("x"))
        with pmr.ShardedModel(
            model_fn,
            data,
            n_shards=n_shards,
            n_replicas=n_replicas,
            shard_by=args.shard_by,
            relays=relays,
            actor_options=actor_options,
        ) as sharded:
            sharded.shard_stats(reset=True)
            per_step = sharded.benchmark(args.evals, args.chains)
            stats = sharded.shard_stats()
            placement = sorted(sharded.placement().values(), reverse=True)
            used_relays = sharded.relays
        timings[layout] = per_step
        mean_compute = sum(s["compute_s"] for s in stats) / sum(s["calls"] for s in stats)
        by_node = {}
        for s in stats:
            calls, seconds = by_node.get(s["node"], (0, 0.0))
            by_node[s["node"]] = (calls + s["calls"], seconds + s["compute_s"])
        slowest = max(seconds / calls for calls, seconds in by_node.values())
        compute = f"{mean_compute * 1e3:.1f} / {slowest * 1e3:.1f}"
        speedup = f"{baseline / per_step:.2f}x"
        row = f"{layout:>10} {per_step * 1e3:>7.1f} ms {speedup:>11} {compute:>16}"
        print(f"{row}   {placement}{' via relays' if used_relays else ''}", flush=True)

    if not args.sample:
        return
    best = min(timings, key=timings.get)
    n_shards, n_replicas = (int(n) for n in best.split("x"))
    iterations = 2 * args.draws
    print(f"\nEnd-to-end pm.sample: {args.chains} chains x {iterations} iterations")
    # Multilevel posteriors often need 100+ leapfrog steps per draw.
    for name, per_step in (("in-process", baseline), (best, timings[best])):
        hours = per_step * iterations * 100 / 3600
        print(f"  {name}: about {hours:.1f} h at 100 leapfrog steps per draw")
    seconds, steps = ray.get(
        in_process_sampling.options(num_cpus=args.chains, scheduling_strategy=on_worker).remote(model_fn, data_ref, args.chains, args.draws)
    )
    print(f"{'in-process':>14} {seconds:>7.0f} s   {steps:.0f} leapfrog steps per draw", flush=True)
    with pmr.ShardedModel(
        model_fn,
        data,
        n_shards=n_shards,
        n_replicas=n_replicas,
        shard_by=args.shard_by,
        relays=relays,
        actor_options=actor_options,
    ) as sharded:
        start = time.perf_counter()
        idata = sharded.sample(
            draws=args.draws,
            tune=args.draws,
            chains=args.chains,
            cores=args.chains,
            progressbar=False,
            compute_convergence_checks=False,
        )
        seconds = time.perf_counter() - start
    steps = float(np.mean(idata["sample_stats"]["n_steps"]))
    print(f"{best:>14} {seconds:>7.0f} s   {steps:.0f} leapfrog steps per draw")


if __name__ == "__main__":
    main()
