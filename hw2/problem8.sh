#!/bin/bash
for b in 10000 30000 50000; do
	for r in 0.005 0.01 0.02; do
		echo $b, $r, "hc_b"$b"_r"$r
		python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 1 -l 2 \
			-s 32 -b $b -lr $r --exp_name "hc_b"$b"_r"$r
	done
done


b=50000
r=0.02
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 1 \
   -l 2 -s 32 -b $b -lr $r --exp_name hc_base
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 1 \
   -l 2 -s 32 -b $b -lr $r -rtg --exp_name hc_rtg
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 1 \
-l 2 -s 32 -b $b -lr $r --nn_baseline --exp_name hc_nn
python train_pg_f18.py HalfCheetah-v2 -ep 150 --discount 0.9 -n 100 -e 1 \
   -l 2 -s 32 -b $b -lr $r -rtg --nn_baseline --exp_name hc_rtg_nn
