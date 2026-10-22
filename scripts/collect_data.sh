
#!/bin/bash

cd /d/projects/Distill/ad-icrl-main || exit 1

export PYTHONPATH="."

for i in {0..80}; do
    echo "Starting goal $i"

    python src/data/ppo.py \
        --env.goal-index "$i" \
        --num-envs 30 \
        --total-timesteps 600000 \
        --learning-rate 0.001 \
        --num-steps 10 \
        --num-minibatches 1 \
        --track

    status=$?

    if [ "$status" -ne 0 ]; then
        echo "Goal $i failed with exit code $status"
        exit "$status"
    fi
done
