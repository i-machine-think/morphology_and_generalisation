#!/bin/bash

DATA_FOLDER="../wiktionary"
SAVE_DATA_FOLDER="data"
MODEL_FOLDER="models"

SAVE_STEPS=719
EPOCH_STEPS=719
TOTAL_STEPS=17975

mkdir "${MODEL_FOLDER}/${2}"
mkdir "${MODEL_FOLDER}/${2}/seed=${1}_wiktionary"
mkdir "${MODEL_FOLDER}/${2}/seed=${1}_wiktionary/wiktionary/"
MODEL_FILENAME="${MODEL_FOLDER}/${2}/seed=${1}_wiktionary/lstms2s"
PRD_FILENAME="${MODEL_FOLDER}/${2}/seed=${1}_wiktionary/wiktionary/lstms2s"

# Use OpenNMT file to preprocess the data into format usable during training
python preprocess.py -train_src "${DATA_FOLDER}/wiktionary_train.src" -train_tgt "${DATA_FOLDER}/wiktionary_train.tgt" \
                        -valid_src "${DATA_FOLDER}/wiktionary_subset.src" -valid_tgt "${DATA_FOLDER}/wiktionary_subset.tgt" \
                        -save_data "${SAVE_DATA_FOLDER}/wiktionary_preprocessed" -overwrite -share_vocab
wait

# Use OpenNMT file to train the model
python train.py --data "${SAVE_DATA_FOLDER}/wiktionary_preprocessed" \
                   --save_checkpoint_steps ${EPOCH_STEPS} --word_vec_size 128 --rnn_size 128 --batch_size 64 \
                   --valid_steps ${EPOCH_STEPS} --save_checkpoint_steps ${EPOCH_STEPS} --share_embeddings --dropout 0.1 \
                   --save_model ${MODEL_FILENAME} --world_size 1 --gpu_ranks 0 --train_steps ${TOTAL_STEPS} --optim adadelta
wait

# Translate the training data
python translate.py -src "${DATA_FOLDER}/wiktionary_train.src" -tgt "${DATA_FOLDER}/wiktionary_train.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_train.prd" --seed 1
wait
# And compute the training set accuracy
python accuracy.py --gold "${DATA_FOLDER}/wiktionary_train.tgt" --pred "${PRD_FILENAME}_train.prd"
wait

# Translate the validation data
python translate.py -src "${DATA_FOLDER}/wiktionary_val.src" -tgt "${DATA_FOLDER}/wiktionary_val.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_val.prd" --seed 1
wait
# And compute the validation set accuracy
python accuracy.py --gold "${DATA_FOLDER}/wiktionary_val.tgt" --pred "${PRD_FILENAME}_val.prd"
wait

# Translate the test data
python translate.py -src "${DATA_FOLDER}/wiktionary_test.src" -tgt "${DATA_FOLDER}/wiktionary_test.tgt" \
                       -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -gpu 0 -batch_size 20 -beam_size 5 \
                       -output "${PRD_FILENAME}_test.prd" --seed 1
wait
# And compute the test set accuracy
python accuracy.py --gold "${DATA_FOLDER}/wiktionary_test.tgt" --pred "${PRD_FILENAME}_test.prd"
wait
