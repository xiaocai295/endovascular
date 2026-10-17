"""Probe which seeds actually advance the catheter (device index 1), not just the guidewire.

Usage:
    source activate.sh
    python probe_cath.py --checkpoint <best_checkpoint.everl> --seeds 24 --n-parallel 3
"""
import argparse
import multiprocessing as mp
import numpy as np

from eve_rl.util import get_env_from_checkpoint, flatten_obs
from eve_rl.algo import AlgoPlayOnly

EVAL_SEEDS = [
    1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 27, 31,
    34, 35, 37, 39, 42, 43, 44, 47, 48, 50, 52, 55, 56, 58, 61, 62, 63, 68,
    69, 70, 71, 73, 79, 80, 81, 84, 89, 91, 92, 93, 95, 97, 102, 103, 108,
    109, 110, 115, 116, 117, 118, 120, 122, 123, 124, 126, 127, 128, 129, 130,
    131, 132, 134, 136, 138, 139, 140, 141, 142, 143, 144, 147, 148, 149, 150,
    151, 152, 154, 155, 156, 158, 159, 161, 162, 167, 168, 171, 175,
]


def _worker(checkpoint, seeds):
    env = get_env_from_checkpoint(checkpoint, "eval")
    env.intervention.make_non_mp()
    algo = AlgoPlayOnly.from_checkpoint(checkpoint)
    algo.reset()
    out = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        flat_obs, _ = flatten_obs(obs)
        max_guide = 0.0
        max_cath = 0.0
        terminal = False
        truncation = False
        step = 0
        while not (terminal or truncation) and step < 1000:
            a = algo.get_eval_action(flat_obs)
            ea = a.reshape(env.action_space.shape)
            ea = (ea + 1) / 2 * (env.action_space.high - env.action_space.low) + env.action_space.low
            obs, _, terminal, truncation, info = env.step(ea)
            flat_obs, _ = flatten_obs(obs)
            il = np.asarray(env.intervention.device_lengths_inserted, dtype=float)
            if il.size >= 2:
                max_guide = max(max_guide, float(il[0]))
                max_cath = max(max_cath, float(il[1]))
            step += 1
        out.append((int(seed), float(max_guide), float(max_cath), int(info.get("success")), step))
        print(
            f"[probe] seed {seed:>4} guide={max_guide:6.1f}mm cath={max_cath:6.1f}mm "
            f"success={int(info.get('success'))} steps={step}",
            flush=True,
        )
    algo.close()
    env.close()
    return out


def _run_chunk(checkpoint, chunk, queue):
    queue.put(_worker(checkpoint, chunk))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seeds", type=int, default=24)
    ap.add_argument("--n-parallel", type=int, default=3)
    a = ap.parse_args()
    seeds = EVAL_SEEDS[: a.seeds]
    chunks = [[] for _ in range(a.n_parallel)]
    for i, s in enumerate(seeds):
        chunks[i % a.n_parallel].append(s)

    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_run_chunk, args=(a.checkpoint, c, q)) for c in chunks if c]
    for p in procs:
        p.start()
    results = []
    for _ in procs:
        results.extend(q.get())
    for p in procs:
        p.join()

    print("\n=== sorted by catheter insertion (desc) ===")
    for seed, g, c, succ, steps in sorted(results, key=lambda r: -r[2]):
        print(f"seed {seed:>4}  guide={g:6.1f}mm  cath={c:6.1f}mm  success={succ}  steps={steps}")


if __name__ == "__main__":
    main()