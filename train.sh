#!/usr/bin/env bash
# 服务器训练入口 —— 上传到 endovascular/train.sh
# 用法: bash train.sh -d cuda -nw 4 --name run1
# cwd 固定在工程根，结果/checkpoint/日志全部落盘到 endovascular/results/ 下。

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/activate.sh"
cd "$SCRIPT_DIR"

python repositories/stEVE_training/training_scripts/DualDeviceNav_train.py "$@"