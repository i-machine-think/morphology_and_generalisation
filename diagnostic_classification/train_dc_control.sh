#!/bin/bash

echo "Run regular DC training on control task"
# Train DC on h-c-enc for epochs 3 - 25
python train_dc.py --model_epochs 3 4 5 10 15 20 25 --models_path ../opennmt/models/seed=1_wiktionary --enc_type h_c_enc --hidden_dim 512 \
                    --target_pos_train -1 --target_pos_test -1 --model_seed 1 2 3 4 5 --dc_seed 1 2 3 4 5 --lr 0.00025 --epochs 50 --batch_size 16 \
                    --controltask