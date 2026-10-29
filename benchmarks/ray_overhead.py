"""Raw cost of Ray actor calls on a cluster, with no PyMC involved.

ShardedModel makes (shards x chains) actor calls per leapfrog step, so this per-call cost
sets a floor under its step time. Measures round trips to trivial actors from this
process (fanning out across nodes) and from an actor to actors on its own node (what a
relay does). Run from the head node, e.g. as a Ray job:

    ray job submit --working-dir benchmarks -- python ray_overhead.py
"""

import argparse
import os
import time

import numpy as np
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=1)
class Echo:
    """Stands in for a shard: takes a parameter vector, returns a small result."""

    def call(self, values):
        return 0.0, values[0][:1]

    def node_id(self) -> str:
        return ray.get_runtime_context().get_node_id()


def time_round_trips(actors, payload, n_iter: int) -> float:
    """Mean ms per round trip in which ``payload`` goes to every actor in parallel."""
    for _ in range(5):
        ray.get([actor.call.remote(payload) for actor in actors])
    start = time.perf_counter()
    for _ in range(n_iter):
        ray.get([actor.call.remote(payload) for actor in actors])
    return (time.perf_counter() - start) / n_iter * 1e3


@ray.remote(num_cpus=0)
class LocalCaller:
    """Times calls from one node to actors on that same node, as a relay makes them."""

    def time(self, actors, payload, n_iter: int) -> float:
        return time_round_trips(actors, payload, n_iter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--address", default="auto")
    parser.add_argument("--widths", type=int, nargs="+", default=[1, 3, 12, 24, 48])
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    ray.init(address=args.address, log_to_driver=False)

    payloads = {
        "1 KB": [np.zeros(128)],
        "78 KB": [np.zeros(10_006)],
    }
    actors = [Echo.options(scheduling_strategy="SPREAD").remote() for _ in range(max(args.widths))]
    nodes = ray.get([actor.node_id.remote() for actor in actors])
    print(f"{len(actors)} trivial actors on {len(set(nodes))} nodes (pid {os.getpid()})\n")

    print("From this process to N actors spread over all nodes (ms per round trip):")
    print(f"{'N':>4} " + " ".join(f"{name:>8}" for name in payloads))
    for width in args.widths:
        cells = [time_round_trips(actors[:width], p, args.iters) for p in payloads.values()]
        print(f"{width:>4} " + " ".join(f"{ms:>8.2f}" for ms in cells), flush=True)

    # Calls that stay on one node, as from a relay to its shards.
    node = max(set(nodes), key=nodes.count)
    local = [a for a, n in zip(actors, nodes, strict=True) if n == node]
    caller = LocalCaller.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node, soft=False)).remote()
    print(f"\nFrom an actor to N actors on its own node ({len(local)} available):")
    print(f"{'N':>4} " + " ".join(f"{name:>8}" for name in payloads))
    for width in sorted({w for w in (1, 4, 8, 16) if w <= len(local)}):
        cells = [ray.get(caller.time.remote(local[:width], p, args.iters)) for p in payloads.values()]
        print(f"{width:>4} " + " ".join(f"{ms:>8.2f}" for ms in cells), flush=True)


if __name__ == "__main__":
    main()
