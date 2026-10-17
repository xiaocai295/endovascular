"""Render a saved checkpoint's rollout into an mp4 video (3D SOFA view).

SofaPygame needs an OpenGL display. On this headless server, run under Xvfb:
    sudo apt-get install -y xvfb          # (one-time, requires sudo password)
    source activate.sh
    xvfb-run -a -s "-screen 0 1280x1024x24" python generate_video.py --seed 1

The 3D scene is rendered by SOFA (vessel tree + guidewire/catheter). For each env step
the rendered frame is collected via env.render() and written to an mp4 with OpenCV.
"""
import os
import argparse
import numpy as np
import torch
import cv2

from eve_rl.util import get_env_from_checkpoint, flatten_obs
from eve_rl.algo import AlgoPlayOnly
from eve.visualisation import SofaPygame

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

DEFAULT_CHECKPOINT = os.path.join(
    PROJECT_ROOT,
    "results/eve_paper/neurovascular/full/mesh_ben/"
    "2026-10-09_233811_run_large/checkpoints/best_checkpoint.everl",
)


def main():
    parser = argparse.ArgumentParser(description="Record a checkpoint rollout as mp4")
    parser.add_argument(
        "--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to .everl file"
    )
    parser.add_argument("--seed", type=int, default=1, help="Eval seed")
    parser.add_argument(
        "--output", type=str, default="rollout_seed1.mp4", help="Output mp4 path"
    )
    parser.add_argument("--fps", type=int, default=15, help="Output video fps")
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Hard cap on rollout length"
    )
    args = parser.parse_args()

    print(f"checkpoint : {args.checkpoint}")
    print(f"seed       : {args.seed}")
    print(f"output     : {args.output}")

    # Rebuild the eval env and load the (play-only) policy from the checkpoint.
    env = get_env_from_checkpoint(args.checkpoint, "eval")
    env.intervention.make_non_mp()
    # Recolor the two devices for clarity: guidewire (orange) vs catheter (blue).
    devices = env.intervention.devices
    if len(devices) >= 1:
        devices[0].color = (1.0, 0.5, 0.0)
    if len(devices) >= 2:
        devices[1].color = (0.3, 0.55, 1.0)
    env.visualisation = SofaPygame(
        env.intervention, env.interim_target, pathfinder=env.pathfinder
    )

    algo = AlgoPlayOnly.from_checkpoint(args.checkpoint)
    algo.reset()

    obs, _ = env.reset(seed=args.seed)
    flat_obs, _ = flatten_obs(obs)

    frames = []

    def grab():
        frame = env.render()
        if frame is not None:
            frames.append(np.asarray(frame))

    grab()  # initial frame
    terminal = False
    truncation = False
    step = 0
    while not (terminal or truncation) and step < args.max_steps:
        action = algo.get_eval_action(flat_obs)
        env_action = action.reshape(env.action_space.shape)
        # same action denormalization as training-time eval (SingleEvalOnly)
        env_action = (env_action + 1) / 2 * (
            env.action_space.high - env.action_space.low
        ) + env.action_space.low

        obs, _, terminal, truncation, info = env.step(env_action)
        flat_obs, _ = flatten_obs(obs)
        grab()
        step += 1

    algo.close()
    env.close()

    if not frames:
        raise RuntimeError("No frames captured (render returned None).")

    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        args.output, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h)
    )
    for frame in frames:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()

    print(f"done: {len(frames)} frames, {w}x{h}, {args.fps}fps -> {args.output}")


if __name__ == "__main__":
    main()