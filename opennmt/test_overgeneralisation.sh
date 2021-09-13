#!/bin/bash

DATA_FOLDER="../wiktionary"
MODEL_FOLDER="models"

MODEL_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/lstms2s"
PRD_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/wiktionary/lstms2s"

sum=0
number=719
for step in {1..25}
do
    sum=$((sum + number))
    
    python translate.py -src "${DATA_FOLDER}/wiktionary_train.src" -tgt "${DATA_FOLDER}/wiktionary_train.tgt" \
                       -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 64 -beam_size 5 --n_best 1 \
                       -output "${PRD_FILENAME}_train_pred_${step}.txt" --seed 1
    wait
    python translate.py -src "${DATA_FOLDER}/wiktionary_val.src" -tgt "${DATA_FOLDER}/wiktionary_val.tgt" \
                       -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 64 -beam_size 5 --n_best 1 \
                       -output "${PRD_FILENAME}_val_pred_${step}.txt" --seed 1
    wait
done
