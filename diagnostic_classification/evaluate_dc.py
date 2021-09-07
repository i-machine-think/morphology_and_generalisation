import os
import torch
import argparse
import logging
import pickle
import numpy as np
from collections import defaultdict
from data import DataHandler
from train_dc import DiagnosticClassifier, set_seed, score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_path", type=str, required=True)
    parser.add_argument("--mode", type=str,
                        choices=["components", "time", "epochs", "time_all"])
    parser.add_argument('--controltask', action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))

    # Check if we are running control task
    categorise_fn = 'control' if args.controltask else None
    control_str = '_control' if args.controltask else ''
    set_seed(1)

    if args.mode == "components":
        enc_types = ["h_c_enc", "h_enc", "c_enc", "i_enc", "f_enc", "o_enc"]
        epochs = [25]
        timesteps = [(0, 0), (1, 1), (2, 2), (-3, -3), (-2, -2), (-1, -1)]
        filename = f"results_components{control_str}.pickle"
    elif args.mode == "time":
        enc_types = ["h_c_enc"]
        epochs = [5]
        timesteps = [(x, y) for x in [0, 1, 2, -3, -2, -1]
                     for y in [0, 1, 2, -3, -2, -1]]
        filename = f"results_time{control_str}.pickle"
    elif args.mode == "epochs":
        enc_types = ["h_c_enc"]
        epochs = [3, 4, 5, 10, 15, 20, 25]
        timesteps = [(-1, -1)]
        filename = f"results_epochs{control_str}.pickle"
    else:
        logging.error("Don't know this mode, exiting.")
        exit()

    enc_type_scores = defaultdict(lambda: defaultdict)
    for epoch in epochs:
        for enc_type in enc_types:
            for target_pos_train, target_pos_test in timesteps:
                tmp_scores, baselines_max, baselines_random = defaultdict(list), [
                ], []
                for model_seed in [1, 2, 3, 4, 5]:
                    models_path = args.models_path.replace(
                        f"seed=1", f"seed={model_seed}")
                    logging.info(models_path)
                    dataset_train = DataHandler(
                        epoch, enc_type, "prediction",
                        data_path=models_path, target_pos=target_pos_train, categorise_fn=categorise_fn)
                    if target_pos_train == target_pos_test:
                        dataset_test = dataset_train
                    else:
                        dataset_test = DataHandler(
                            epoch, enc_type, "prediction",
                            data_path=models_path, target_pos=target_pos_test, categorise_fn=categorise_fn)

                    for dc_seed in [1, 2, 3, 4, 5]:
                        set_seed(dc_seed)

                        dc_filename = \
                            f"dcs_pytorch/parameters_enctype={enc_type}_" + \
                            f"target=prediction_epoch={epoch}_l0=False" + \
                            f"_model-seed={model_seed}_dc-seed={dc_seed}_lr=0.00025_" + \
                            f"timestep={target_pos_train}{control_str}.pt"

                        model = torch.load(dc_filename)

                        # Now predict the targets for different test - train subsets
                        f1, baseline_max, baseline_random, f1s_per_class = \
                            score(model, dataset_test.test)
                        for x in f1s_per_class:
                            tmp_scores[x].append(f1s_per_class[x])
                        baselines_max.append(baseline_max)
                        baselines_random.append(baseline_random)

                means, stds = [], []
                for inflection_class in range(5):
                    means.append(np.mean(tmp_scores[inflection_class]))
                    stds.append(np.std(tmp_scores[inflection_class]))

                macro = []
                for seed in range(1, 6):
                    macro.append(np.mean([tmp_scores[x][seed]
                                 for x in range(5)]))

                enc_type_scores[(enc_type, epoch, target_pos_train, target_pos_test)] = \
                    {"means": means + [np.mean(macro)],
                     "stds": stds + [np.std(macro)],
                     "baseline_max": np.mean(baselines_max),
                     "baseline_random": np.mean(baselines_random)}
                print(enc_type_scores)
                pickle.dump(dict(enc_type_scores), open(filename, 'wb'))
    print(enc_type_scores)
