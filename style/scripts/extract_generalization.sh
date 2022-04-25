#! /usr/bin/env bash

dataset=v4
exp='pretrain'

mkdir -p results/$dataset/$exp
echo 'step, irm, split, envs, seed, ADE, FDE' | tee results/$dataset/$exp/summary.csv

for filename in log/$dataset/$exp/*.log; do
	cat ${filename} \
	| grep "Model:\|Split:\|Envs:\|Seed:\|ADE:" \
	| sed "s/.*Model: .*\/P\([0-9.]*\).*irm\[\([0-9.]*\)\].*/\1, \2,/g" \
	| sed "s/.*Split: \([a-z.]*\).*/\1,/g" \
	| sed "s/.*Envs: \([0-9.-]*\).*/\1,/g" \
	| sed "s/.*Seed: \([0-9]*\).*/\1,/g" \
	| sed "s/.*ADE: \([0-9.]*\).*FDE: \([0-9.]*\).*/\1, \2 /g" \
	| paste -d " " - - - - - \
	| tee -a results/$dataset/$exp/summary.csv
done

python visualize.py --exp $exp