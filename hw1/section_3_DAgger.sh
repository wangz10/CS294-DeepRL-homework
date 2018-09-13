#!/bin/bash
# Run DAgger for all tasks
mkdir -p results/DAgger
for e in Hopper-v2 Ant-v2 HalfCheetah-v2 Humanoid-v2 Reacher-v2 Walker2d-v2; do
	python run_DAgger.py experts/$e.pkl $e --n_epochs 10 --n_dagger_iter 20
done
