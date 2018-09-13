#!/bin/bash

# 1. Generate expert experiences for all tasks
mkdir -p expert_data
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2; do
    python run_expert.py experts/$e.pkl $e --num_rollouts=20
done

# 2. Run BC for all tasks
mkdir -p results/BC
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2; do
	python run_BC.py $e
done
