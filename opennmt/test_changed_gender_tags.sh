#!/bin/bash
DATA_FOLDER="../wiktionary"
SAVE_DATA_FOLDER="data"
MODEL_FOLDER="models"

SAVE_STEPS=719
EPOCH_STEPS=719
TOTAL_STEPS=18694

mkdir "${MODEL_FOLDER}/"
mkdir "${MODEL_FOLDER}/seed=${1}_wiktionary"
MODEL_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/lstms2s"
PRD_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/enforce_gender/lstms2s"

mkdir "${MODEL_FOLDER}/seed=${1}_wiktionary/enforce_gender"

# 1. Forced fem tag
# Real data
python translate.py -src "${DATA_FOLDER}/enforce_gender/wiktionary_val_fem-original.src" \
                       -tgt "${DATA_FOLDER}/enforce_gender/wiktionary_val_fem-original.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_val_fem-original.prd" --seed 1
wait
# With the fake tag
python translate.py -src "${DATA_FOLDER}/enforce_gender/wiktionary_val_fem-fem.src" \
                       -tgt "${DATA_FOLDER}/enforce_gender/wiktionary_val_fem-original.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_val_fem-fem.prd" --seed 1
wait

# 2. Forced masculine tag
# Real data
python translate.py -src "${DATA_FOLDER}/enforce_gender/wiktionary_val_mas-original.src" \
                       -tgt "${DATA_FOLDER}/enforce_gender/wiktionary_val_mas-original.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_val_mas-original.prd" --seed 1
wait
# With the fake tag
python translate.py -src "${DATA_FOLDER}/enforce_gender/wiktionary_val_mas-mas.src" \
                       -tgt "${DATA_FOLDER}/enforce_gender/wiktionary_val_mas-original.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_val_mas-mas.prd" --seed 1
wait

# 3. Forced neuter tag
# Real data
python translate.py -src "${DATA_FOLDER}/enforce_gender/wiktionary_val_neut-original.src" \
                       -tgt "${DATA_FOLDER}/enforce_gender/wiktionary_val_neut-original.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_val_neut-original.prd" --seed 1
wait
# With the fake tag
python translate.py -src "${DATA_FOLDER}/enforce_gender/wiktionary_val_neut-neut.src" \
                       -tgt "${DATA_FOLDER}/enforce_gender/wiktionary_val_neut-original.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_val_neut-neut.prd" --seed 1
wait

