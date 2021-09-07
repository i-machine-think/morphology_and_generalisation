#/bin/bash
# 1 Argument: seed for the model
DATA_FOLDER="../wiktionary"
SAVE_DATA_FOLDER="data"
MODEL_FOLDER="models"

SAVE_STEPS=1000
EPOCH_STEPS=1000
TOTAL_STEPS=50000 # Around 30 epochs

MODEL_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/lstm_mccurdy"
PRD_FILENAME="${MODEL_FOLDER}/seed=${1}_wiktionary/lstm_mccurdy"

mkdir "${MODEL_FOLDER}"
mkdir "${MODEL_FOLDER}/seed=${1}_wiktionary"

# Build  data
onmt_preprocess -train_src "${DATA_FOLDER}/wiktionary_train.src" -train_tgt "${DATA_FOLDER}/wiktionary_train.tgt" \
                        -valid_src "${DATA_FOLDER}/wiktionary_subset.src" -valid_tgt "${DATA_FOLDER}/wiktionary_subset.tgt" \
                        -save_data "${SAVE_DATA_FOLDER}/wiktionary_preprocessed" -overwrite -share_vocab --seed "${1}"
wait

# Train model
onmt_train --src_word_vec_size 300 --rnn_size 100 --tgt_word_vec_size 300 --data "${SAVE_DATA_FOLDER}/wiktionary_preprocessed" --gpu_ranks 0\
                --batch_size 20 --seed "${1}" --dropout 0.3 --encoder_type brnn\
                --valid_steps "${SAVE_STEPS}" --train_steps "${TOTAL_STEPS}" --optim adadelta \
                --learning_rate 0.9 --save_checkpoint_steps "${SAVE_STEPS}" --save_model ${MODEL_FILENAME}
wait

# Evaluate model on train, validation and test set
onmt_translate -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -src "${DATA_FOLDER}/wiktionary_train.src" \
                        --output "${PRD_FILENAME}_train.prd"  \
                        --beam_size 12  --gpu 0 --n_best 1 --seed "${1}"
wait

python accuracy.py --gold "${DATA_FOLDER}/wiktionary_train.tgt" --pred "${PRD_FILENAME}_train.prd"
wait

# Validation set
onmt_translate -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -src "${DATA_FOLDER}/wiktionary_val.src" \
                        --output "${PRD_FILENAME}_val.prd"  \
                        --beam_size 12  --gpu 0 --n_best 1 --seed "${1}"
wait

python accuracy.py --gold "${DATA_FOLDER}/wiktionary_val.tgt" --pred "${PRD_FILENAME}_train.prd"
wait

# Test set
onmt_translate -model "${MODEL_FILENAME}_step_${TOTAL_STEPS}.pt" -src "${DATA_FOLDER}/wiktionary_test.src" \
                        --output "${PRD_FILENAME}_test.prd"  \
                        --beam_size 12 --gpu 0 --n_best 1 --seed "${1}"
wait

python accuracy.py --gold "${DATA_FOLDER}/wiktionary_test.tgt" --pred "${PRD_FILENAME}_train.prd"
wait