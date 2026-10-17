"""Load a saved .everl checkpoint and run offline evaluation (optionally in parallel).

Usage (must be run with the project environment active):
    source activate.sh
    python evaluate_checkpoint.py                          # best_checkpoint, all 98 seeds, 1 process
    python evaluate_checkpoint.py --n-parallel 24 --device cpu
    python evaluate_checkpoint.py --seeds 20 --n-parallel 8
    python evaluate_checkpoint.py --checkpoint results/.../checkpoint2771849.everl

Prints per-episode results and aggregate metrics (success rate, reward, path ratio,
steps, average translation speed, trajectory length).
"""
import os
import argparse
import multiprocessing as mp
import torch

from eve_rl.agent import SingleEvalOnly

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Same seed set used by DualDeviceNav_train.py
EVAL_SEEDS = [
    1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 27, 31,
    34, 35, 37, 39, 42, 43, 44, 47, 48, 50, 52, 55, 56, 58, 61, 62, 63, 68,
    69, 70, 71, 73, 79, 80, 81, 84, 89, 91, 92, 93, 95, 97, 102, 103, 108,
    109, 110, 115, 116, 117, 118, 120, 122, 123, 124, 126, 127, 128, 129, 130,
    131, 132, 134, 136, 138, 139, 140, 141, 142, 143, 144, 147, 148, 149, 150,
    151, 152, 154, 155, 156, 158, 159, 161, 162, 167, 168, 171, 175,
]

DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "results/eve_paper/neurovascular/full/mesh_ben/"
    "2026-10-09_233811_run_large/checkpoints/best_checkpoint.everl",
)

INFO_KEYS = ["path_ratio", "average translation speed", "trajectory length"]


def _worker(checkpoint, device_str, seeds):
    """Runs in a child process. Evaluate a chunk of seeds, return per-episode results."""
    device = torch.device(device_str)
    agent = SingleEvalOnly.from_checkpoint(checkpoint, device=device)
    print(f"[worker] checkpoint loaded, {len(seeds)} seeds", flush=True)
    results = []
    for ep in agent.evaluate(seeds=list(seeds)):
        info = ep.infos[-1]
        results.append(
            {
                "seed": ep.seed,
                "success": bool(info.get("success")),
                "reward": float(ep.episode_reward),
                "steps": info.get("steps", 0),
                "path_ratio": info.get("path_ratio"),
                "average translation speed": info.get("average translation speed"),
                "trajectory length": info.get("trajectory length"),
            }
        )
        print(
            f"[worker] seed {ep.seed:>4}  success={int(info.get('success'))}"
            f"  reward={ep.episode_reward:+.3f}  steps={info.get('steps')}",
            flush=True,
        )
    agent.close()
    return results


def _run_chunk(checkpoint, device_str, chunk, queue):
    """Process target: run one chunk and push results to the queue."""
    queue.put(_worker(checkpoint, device_str, chunk))


def _aggregate(results):
    results = sorted(results, key=lambda r: r["seed"])
    n = len(results)
    successes = sum(r["success"] for r in results)
    rewards = [r["reward"] for r in results]
    steps = [r["steps"] for r in results]

    print("\n--- per-episode ---")
    for r in results:
        print(
            f"seed {r['seed']:>4}  success={int(r['success'])}"
            f"  reward={r['reward']:+.3f}  steps={r['steps']}"
        )

    print("\n--- aggregate ---")
    print(f"episodes        : {n}")
    print(f"success rate    : {successes}/{n} = {successes / n:.3f}")
    print(f"mean reward     : {sum(rewards) / n:+.3f}")
    print(f"mean steps      : {sum(steps) / n:.1f}")
    for k in INFO_KEYS:
        vals = [r[k] for r in results if r[k] is not None]
        if vals:
            print(f"mean {k:<22}: {sum(vals) / len(vals):.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to .everl file"
    )
    parser.add_argument(
        "--seeds", type=int, default=None, help="Evaluate only the first N seeds (default: all)"
    )
    parser.add_argument(
        "--n-parallel", type=int, default=1, help="Number of parallel worker processes"
    )
    parser.add_argument(
        "--device", type=str, default="cpu", choices=["cpu", "cuda", "cuda:0", "cuda:1"]
    )
    args = parser.parse_args()

    seeds = EVAL_SEEDS if args.seeds is None else EVAL_SEEDS[: args.seeds]

    print(f"checkpoint : {args.checkpoint}")
    print(f"device     : {args.device}")
    print(f"seeds      : {len(seeds)}")
    print(f"parallel   : {args.n_parallel}")
    print("", flush=True)

    if args.n_parallel <= 1 or len(seeds) == 1:
        results = _worker(args.checkpoint, args.device, seeds)
    else:
        # round-robin chunks for better load balance across variable-length episodes
        chunks = [[] for _ in range(args.n_parallel)]
        for i, seed in enumerate(seeds):
            chunks[i % args.n_parallel].append(seed)

        # Use explicit (non-daemonic) Processes: SOFA's SimulationMP spawns its own
        # child process per worker, which is forbidden from daemonic pool workers.
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(
                target=_run_chunk,
                args=(args.checkpoint, args.device, chunk, queue),
            )
            for chunk in chunks
        ]
        for p in procs:
            p.start()
        results = []
        for _ in procs:
            results.extend(queue.get())
        for p in procs:
            p.join()

    _aggregate(results)


if __name__ == "__main__":
    main()