#!/usr/bin/env bash
# Collect SFT dataset from MiniGrid using the expert planner (Dijkstra)
set -euo pipefail

cd "$(dirname "$0")/.."

python -m nanovlm.data_collection.collect_data \
    --env-ids \
        MiniGrid-Empty-5x5-v0 \
        MiniGrid-Empty-6x6-v0 \
        MiniGrid-Empty-8x8-v0 \
        MiniGrid-Empty-16x16-v0 \
        MiniGrid-Empty-Random-5x5-v0 \
        MiniGrid-Empty-Random-6x6-v0 \
    --episodes 100 \
    --out-dir nanovlm/data/minigrid
