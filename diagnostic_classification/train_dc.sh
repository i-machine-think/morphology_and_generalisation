#!/bin/bash

if [[ "${1}" == "regular" ]]
then
    echo "Run regular"
    # Train DC on h-c-enc for epochs 3 - 25
    python train_dc.py --model_epochs 3 4 5 10 15 20 25 --models_path ../opennmt/models/seed=1_wiktionary --enc_type h_c_enc --hidden_dim 512 \
                       --target_pos_train -1 --target_pos_test -1 --model_seed 1 2 3 4 5 --dc_seed 1 2 3 4 5 --lr 0.00025 --epochs 50 --batch_size 16
    wait
elif [[ "${1}" == "l0" ]]
then
    echo "Run l0"
    # Train DC on h-c-enc for epochs 3 - 25 with l0 objective
    python train_dc.py --model_epochs 5 25 --models_path ../opennmt/models/seed=1_wiktionary --enc_type h_c_enc --hidden_dim 512 \
                       --target_pos_train -1 --target_pos_test -1 --model_seed 1 2 3 4 5 --dc_seed 1 2 3 4 5 --l0  --lr 0.001 --epochs 50 --batch_size 16
    wait
elif [[ "${1}" == "components" ]]
then
    echo "Run components"
    # Train DC across model components
    for enc_type in o_enc # c_enc i_enc f_enc o_enc
    do
        python train_dc.py --model_epochs 5 25 --models_path ../opennmt/models/seed=1_wiktionary --enc_type $enc_type --hidden_dim 256 \
                       --target_pos_train $2 --target_pos_test $2 --model_seed 1 2 3 4 5 --dc_seed 1 2 3 4 5 --lr 0.00025 --epochs 50 --batch_size 16
        wait
    done

elif [[ "${1}" == "time" ]]
then
    # Train DC to test generalisability over time steps
    for train_position in 0 1 2 -3 -2 -1
    do
        for test_position in 0 1 2 -3 -2 -1
        do
            python train_dc.py --model_epochs 5 25 --models_path ../opennmt/models/seed=1_wiktionary --enc_type h_c_enc --hidden_dim 512  \
                               --target_pos_train $train_position --target_pos_test $test_position \
                               --model_seed 1 2 3 4 5 --dc_seed 1 2 3 4 5 --lr 0.00025 --epochs 50 --batch_size 16
            wait
        done
    done
else
    echo "Don't know this mode"
fi