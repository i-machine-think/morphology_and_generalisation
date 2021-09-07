#!/bin/bash

DATA_FOLDER="../wiktionary"
SAVE_DATA_FOLDER="data"
MODEL_FOLDER="models"

MODEL_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/lstms2s"
PRD_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/nonce/lstms2s"

mkdir "${MODEL_FOLDER}/seed=${1}_wiktionary/nonce"

sum=0
number=719
for step in {1..25}
do
    sum=$((sum + number))

    # Nonce word with neuter tag
    python translate.py -src "${DATA_FOLDER}/nonce/nonce_n.src" -tgt "${DATA_FOLDER}/nonce/nonce_n.src" \
                        -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 5 \
                        -output "${PRD_FILENAME}_nonce_n_${step}.prd" --seed 1
    wait
    # Nonce word with feminine tag
    python translate.py -src "${DATA_FOLDER}/nonce/nonce_f.src" -tgt "${DATA_FOLDER}/nonce/nonce_f.src" \
                       -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 5 \
                       -output "${PRD_FILENAME}_nonce_f_${step}.prd" --seed 1
    wait
    # Nonce word with masculine tag
    python translate.py -src "${DATA_FOLDER}/nonce/nonce_m.src" -tgt "${DATA_FOLDER}/nonce/nonce_m.src" \
                       -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 5 \
                       -output "${PRD_FILENAME}_nonce_m_${step}.prd" --seed 1
    wait
    # Nonce word with neuter tag and a word before the nonce word to mimic a compound setup
    python translate.py -src "${DATA_FOLDER}/nonce/nonce_compound_zahn.src" -tgt "${DATA_FOLDER}/nonce/nonce_compound_zahn.src" \
                       -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 5 \
                       -output "${PRD_FILENAME}_nonce_compound_zahn_${step}.prd" --seed 1 
    wait
    # Nonce word with neuter tag and a word before the nonce word to mimic a compound setup
    python translate.py -src "${DATA_FOLDER}/nonce/nonce_compound_tier.src" -tgt "${DATA_FOLDER}/nonce/nonce_compound_tier.src" \
                       -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 5 \
                       -output "${PRD_FILENAME}_nonce_compound_tier_${step}.prd" --seed 1 
    wait
    # Nonce word with neuter tag and a word before the nonce word to mimic a compound setup
    python translate.py -src "${DATA_FOLDER}/nonce/nonce_compound_hand.src" -tgt "${DATA_FOLDER}/nonce/nonce_compound_hand.src" \
                       -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 5 \
                       -output "${PRD_FILENAME}_nonce_compound_hand_${step}.prd" --seed 1 
    wait

done
