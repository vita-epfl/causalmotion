#! /usr/bin/env bash

DATASET=hotel
FOLDERNAME=log

mkdir -p results/${DATASET}

echo 'method, risk, lambda, data_tr, batch_type, tr_shift, tr_k, epoch, seed, data_te, ev_shift, split, ade, fde' | tee results/${DATASET}/summary.csv

for filename in ${FOLDERNAME}/${DATASET}/*.log; do
	cat ${filename} \
	| grep "Model:\|Dataset:\|Eval shift:\|Dataset type:\|ADE:" \
	| sed "s/.*Model: STGAT_\([a-z.]*\)_\([a-z.]*\)_\([0-9.]*\)_data_\([a-z.]*\)_\([a-z.]*\)_ds_\([0-9-]*\)_bk_\([0-9]*\)_ep_\([0-9-]*\)_seed_\([0-9]*\).*/\1, \2, \3, \4, \5, \6, \7, \8, \9,/g" \
	| sed "s/.*Dataset: \([a-z.]*\).*/\1,/g"\
	| sed "s/.*Eval shift: \([0-9-]*\).*/\1,/g" \
	| sed "s/.*Dataset type: \(.*\).*/\1,/g" \
	| sed "s/.*ADE: \([0-9.]*\).*FDE: \([0-9.]*\).*/\1, \2 /g" \
	| paste -d " " - - - - -\
	| tee -a results/${DATASET}/summary.csv
done
