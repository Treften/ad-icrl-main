@echo off
cd /d D:\projects\Distill\ad-icrl-main
if errorlevel 1 exit /b 1

set "PYTHONPATH=."

for /L %%i in (0,1,80) do (
    echo Starting goal %%i

    python src\data\ppo.py --env.goal-index %%i --num-envs 30 --total-timesteps 600000 --learning-rate 0.001 --num-steps 10 --num-minibatches 1 --track

    if errorlevel 1 (
        echo Goal %%i failed
        exit /b 1
    )
)
