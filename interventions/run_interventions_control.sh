#!/bin/bash

for lr in 0.5 1.0 1.5 2.0 2.5 3.0 3.5
do
    python intervention.py  -l $lr --steps $1 --model_path ../opennmt/models/seed\=${2}_wiktionary/ --dc_lr 0.00025 --model_seed $2 --dc_seed 1 \
                            --dc_path ../diagnostic_classification/dcs_pytorch --data_path ../opennmt/data --controltask
    wait
done