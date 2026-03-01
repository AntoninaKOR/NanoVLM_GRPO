#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python -m nanovlm.data_collection.collect_data \
    --env-ids \
        MiniGrid-Empty-5x5-v0 \
        MiniGrid-Empty-8x8-v0 \
        MiniGrid-Empty-Random-5x5-v0 \
    --episodes 10 \
    --out-dir nanovlm/data/minigrid_small