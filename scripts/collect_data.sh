#!/bin/bash
export PYTHONPATH=.

for i in {1548..1999}; do
    python3 src/data/ppo.py \
        --id "$i" \
        --seed "$i" \
        --env.goal-index $((i % 81)) \
        --num-envs 100 \
        --total-timesteps 200000 \
        --learning-rate 0.001 \
        --num-steps 10 \
        --num-minibatches 1
done