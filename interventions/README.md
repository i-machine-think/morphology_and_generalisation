## Use Diagnostic Classifiers to perform Interventions
Interventions can be done using saved DCs, see `../diagnostic_classification` for details on how to train diagnostic classifiers.

#### 1. Run interventions
Run `run_interventions.sh  <epochs> <model_seed> <dc_seed>` to intervene on those words from the validation set that are wrongly predicted by the model trained with seed `model_seed` in epoch `epoch`. The results reported use `epoch = 5`.

#### 2. Visualize interventions
The `visualize.ipynb` notebook can be used to recreate the visualisations from the paper.


#### 3. Running Control Interventions
Run `run_interventions_control.sh  <epochs> <model_seed> <dc_seed>` to use the control DC for intervention. 