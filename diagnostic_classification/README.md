## Train Diagnostic Classifiers

#### 1. Train DCs.
Run `train_dc.sh` in multiple modes to get the various types of DC results:
1. `train_dc.sh regular` to train on concatenated hidden and memory cell states for multiple training epochs.
2. `train_dc.sh l0` to train sparse DCs.
3. `train_dc.sh components` to train on gates and hidden and memory cell states.
4. `train_dc.sh time` to train on 6 time steps and test each DC on all other 6 time steps.

#### 2. Evaluate DCs.
While the performance is written to file, you can also recompute the scores and gather them in a pickled format, by running: `evaluate_dc.sh MODE`, that will store pickled files named:
- `results_components.pickle`
- `results_time.pickle`
- `results_epochs.pickle`
The pickled files are dictionaries, with keys `(component_name, epoch, train_time_step, test_time_step)`.

#### 3. Visualise DCs.
- You can create TSNE visualisations by running `python visualise.py --epoch X --model_seed X`.
- You can recreate the paper's DC graphs with `visualise_diagnostic_classification.ipynb`.

#### 4. Control tasks.
To reproduce the results presented in the appendix, run the DC:
1. Run `train_dc_control.sh` to train on concatenated hidden and memory cell states for multiple training epochs.
2. Run `evaluate_dc_control.sh` to evaluate the control DC. This will create a file called `results_epochs_control.pickle`.
3. The `results_epochs_control.pickle` can be used to visualize the plot for control DC accuracy as shown in the paper, using `visualise_diagnostic_classification.ipynb`.

Any of the shell scripts defined in this directory can be modified to be used in a control task setting, by adding `--controltask` as an extra argument to the python call.

