## OpenNMT models for morphological inflection

### To reproduce our main results
1. Train the models by running `train.sh <seed>` with the desired seed as argument. The trained models (extension `.pt`) and extracted hidden representations will be saved to `models/seed=x_wiktionary/...`.
2. Extract nonce predictions by running `test_nonce.sh <seed>` with the desired seed as argument. The predictions will be saved in `models/seed=x_wiktionary/nonce/...`.
3. Extract overgeneralisation predictions by running `test_overgeneralisation.sh <seed>` with the desired seed as argument. The predictions will be saved in `models/seed=x_wiktionary/wiktionary/...`.
4. Extract the "enforced gender" predictions by running `test_changed_gender_tags.sh <seed>` with the desired seed as argument. The predictions will be saved in `models/seed=x_wiktionary/enforce_gender/...`.
5. Test increasing lengths for -s by running `test_lengths.sh <seed>` with the desired seed as argument. The predictions will be saved in `models/seed=x_wiktionary/lengths...`.

:rotating_light: The OpenNMT installation has been adapted to include a custom LSTM that allows one to output hidden states, gates etc.
As a result, the normal OpenNMT functionalities that are not being used in our project may no longer work as anticipated.
This code base is not meant to be used to train other OpenNMT models than those trained by us.

### To train additional models
This directory provides an additional training scripts, that work with unedited OpenNMT for the necessary OpenNMT version).
- Install OpenNMT-py 1.2.
- To reproduce the original bidirectional LSTM with attention as used by McCurdy et al., run `train_mccurdy.sh <seed>`.


