#!/bin/bash

DATA_FOLDER="../wiktionary"
SAVE_DATA_FOLDER="data"
MODEL_FOLDER="models"

MODEL_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/lstms2s"
PRD_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/length/lstms2s"

rm "${MODEL_FOLDER}/seed=${1}_wiktionary/length" -r
mkdir "${MODEL_FOLDER}/seed=${1}_wiktionary/length"

sum=17975
number=719
for step in 25
do
    #sum=$((sum + number))

    # Nonce word with neuter tag
    python translate.py -src "../wiktionary/s_length.src" -tgt "../wiktionary/s_length_lieblings.src" \
                        -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 1 \
                        -output "${PRD_FILENAME}_length_${step}.prd" --seed 1
    wait
    python translate.py -src "../wiktionary/s_length_haupt.src" -tgt "../wiktionary/s_length_lieblings.src" \
                        -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 1 \
                        -output "${PRD_FILENAME}_length_haupt_${step}.prd" --seed 1
    wait
    python translate.py -src "../wiktionary/s_length_not.src" -tgt "../wiktionary/s_length_lieblings.src" \
                        -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 1 \
                        -output "${PRD_FILENAME}_length_not_${step}.prd" --seed 1
    wait
    python translate.py -src "../wiktionary/s_length_see.src" -tgt "../wiktionary/s_length_lieblings.src" \
                        -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 1 \
                        -output "${PRD_FILENAME}_length_see_${step}.prd" --seed 1
    wait
    python translate.py -src "../wiktionary/s_length_lieblings.src" -tgt "../wiktionary/s_length_lieblings.src" \
                        -model "${MODEL_FILENAME}_step_${sum}.pt" -gpu 0 -batch_size 20 -beam_size 5 -n_best 1 \
                        -output "${PRD_FILENAME}_length_lieblings_${step}.prd" --seed 1
    wait

done
