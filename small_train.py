#!/usr/bin/env python3
"""服务器小规模训练入口（官方 DualDeviceNav SAC 流程，规模可调）。

镜像 stEVE_training/training_scripts/DualDeviceNav_train.py 的接线
（同样的 DualDeviceNav + BenchEnv + BenchAgentSynchron + Runner），
但把 HEATUP / TRAINING / EXPLORE_STEPS_BTW_EVAL / eval-seeds 等规模参数
开放成命令行，便于先在服务器上用小规模验证成功率曲线能否爬升，
再决定是否放大到正式训练（2e7 步）。

用法（在 endovascular/ 下，先 source activate.sh）：
  python small_train.py -d cuda:1 -nw 8 --name small1 \
      --heatup 5000 --training 50000 --explore-eval 7500 --eval-seeds 0,1,2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
import torch.multiprocessing as mp

HERE = Path(__file__).resolve().parent
TRAINING_SCRIPTS = HERE / "repositories/stEVE_training/training_scripts"
if str(TRAINING_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(TRAINING_SCRIPTS))

from util.util import get_result_checkpoint_config_and_log_path
from util.env import BenchEnv
from util.agent import BenchAgentSynchron
from eve_rl import Runner
from eve_bench import DualDeviceNav

RESULTS_FOLDER = str(HERE / "results/runs")

# 官方超参数（保持不变）
GAMMA = 0.99
REWARD_SCALING = 1
REPLAY_BUFFER_SIZE = 1e4
CONSECUTIVE_ACTION_STEPS = 1
LR_END_FACTOR = 0.15
LR_LINEAR_END_STEPS = 6e6
DEBUG_LEVEL = logging.DEBUG

LR = 0.00021989352630306626
HIDDEN_LAYERS = [900, 900, 900, 900]
EMBEDDER_NODES = 500
EMBEDDER_LAYERS = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="官方 DualDeviceNav SAC 小规模训练")
    p.add_argument(
        "-d", "--device", default="cuda:1",
        help="trainer 设备（只用 1 张卡；卡0不可用，默认 cuda:1）",
    )
    p.add_argument(
        "-nw", "--n-worker", type=int, default=8,
        help="并行 SOFA worker 数（CPU 采集速度，按 nproc 设，不要超过核数）",
    )
    p.add_argument("--name", default="small1", help="试验名")
    p.add_argument("--heatup", type=int, default=5000,
                   help="预热随机步数（官方 5e5）")
    p.add_argument("--training", type=int, default=50000,
                   help="总训练步数（官方 2e7）")
    p.add_argument("--explore-eval", type=int, default=7500,
                   help="两次 eval 之间的探索步数（官方 2.5e5）")
    p.add_argument("--consecutive-episodes", type=int, default=10,
                   help="更新前连续探索 episode 数（官方 100）")
    p.add_argument("--eval-seeds", default="0,1,2",
                   help="逗号分隔的 eval 种子，如 0,1,2")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--update-per-explore", type=float, default=1 / 20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    trainer_device = torch.device(args.device)
    worker_device = torch.device("cpu")
    eval_seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]

    custom_parameters = {
        "lr": LR,
        "hidden_layers": HIDDEN_LAYERS,
        "embedder_nodes": EMBEDDER_NODES,
        "embedder_layers": EMBEDDER_LAYERS,
        "HEATUP_STEPS": args.heatup,
        "TRAINING_STEPS": args.training,
        "EXPLORE_STEPS_BTW_EVAL": args.explore_eval,
        "CONSECUTIVE_EXPLORE_EPISODES": args.consecutive_episodes,
        "BATCH_SIZE": args.batch_size,
        "UPDATE_PER_EXPLORE_STEP": args.update_per_explore,
        "n_worker": args.n_worker,
        "device": args.device,
    }

    (
        results_file,
        checkpoint_folder,
        config_folder,
        log_file,
    ) = get_result_checkpoint_config_and_log_path(
        all_results_folder=RESULTS_FOLDER, name=args.name
    )
    logging.basicConfig(
        filename=log_file,
        level=DEBUG_LEVEL,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        force=True,
    )

    intervention = DualDeviceNav()
    env_train = BenchEnv(
        intervention=intervention, mode="train", visualisation=False, n_max_steps=500
    )
    intervention_eval = DualDeviceNav()
    env_eval = BenchEnv(
        intervention=intervention_eval,
        mode="eval",
        visualisation=False,
        n_max_steps=1000,
    )
    agent = BenchAgentSynchron(
        trainer_device,
        worker_device,
        LR,
        LR_END_FACTOR,
        LR_LINEAR_END_STEPS,
        HIDDEN_LAYERS,
        EMBEDDER_NODES,
        EMBEDDER_LAYERS,
        GAMMA,
        args.batch_size,
        REWARD_SCALING,
        REPLAY_BUFFER_SIZE,
        env_train,
        env_eval,
        CONSECUTIVE_ACTION_STEPS,
        args.n_worker,
        False,
        False,
    )

    env_train_config = str(Path(config_folder) / "env_train.yml")
    env_train.save_config(env_train_config)
    env_eval_config = str(Path(config_folder) / "env_eval.yml")
    env_eval.save_config(env_eval_config)
    infos = list(env_eval.info.info.keys())
    runner = Runner(
        agent=agent,
        heatup_action_low=[[-10.0, -1.0], [-11.0, -1.0]],
        heatup_action_high=[[35, 3.14], [30, 3.14]],
        agent_parameter_for_result_file=custom_parameters,
        checkpoint_folder=checkpoint_folder,
        results_file=results_file,
        info_results=infos,
        quality_info="success",
    )
    runner_config = str(Path(config_folder) / "runner.yml")
    runner.save_config(runner_config)

    reward, success = runner.training_run(
        args.heatup,
        args.training,
        args.explore_eval,
        args.consecutive_episodes,
        args.update_per_explore,
        eval_seeds=eval_seeds,
    )
    agent.close()
    print(f"SMALL_TRAIN_DONE results_file={results_file} reward={reward} success={success}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()