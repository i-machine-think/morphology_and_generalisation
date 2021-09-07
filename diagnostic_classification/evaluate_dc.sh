#!/bin/bash

python evaluate_dc.py --models_path ../opennmt/models/seed=1_wiktionary --mode $1
wait
