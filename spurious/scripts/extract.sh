#! /usr/bin/env bash

DATASET=eth
FOLDERNAME=log

echo 'method, irm, data, tr_shift, tr_k, epoch, seed, ev_shift, split, ade, fde' | tee results/${DATASET}/summary.csv

for filename in ${FOLDERNAME}/${DATASET}/*.log; do
	cat ${filename} \
	| grep "Model:\|Eval shift:\|Dataset type:\|ADE:" \
	| sed "s/.*Model: STGAT_\([a-z.]*\)_irm_\([0-9.]*\)_data_\([a-z.]*\)_ds_\([0-9-]*\)_bk_\([0-9]*\)_ep_\([0-9-]*\)_seed_\([0-9]*\).*/\1, \2, \3, \4, \5, \6, \7, \8,/g" \
	| sed "s/.*Eval shift: \([0-9-]*\).*/\1,/g" \
	| sed "s/.*Dataset type: \(.*\).*/\1,/g" \
	| sed "s/.*ADE: \([0-9.]*\).*FDE: \([0-9.]*\).*/\1, \2 /g" \
	| paste -d " " - - - - \
	| tee -a results/${DATASET}/summary.csv
done
