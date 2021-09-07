import pickle
import argparse
import random
from collections import defaultdict
from warnings import filterwarnings
import logging
import os
import torch
import sklearn.metrics
from sklearn.linear_model import LogisticRegression
import numpy as np
from l0module import L0Linear
from data import DataHandler


class DiagnosticClassifier(torch.nn.Module):
    """Simple linear layer to classify hidden states."""

    def __init__(self, l0=False, hidden_dim=400):
        super().__init__()
        if not l0:
            self.linear = torch.nn.Linear(hidden_dim, 5, bias=True)
        else:
            self.linear = L0Linear(hidden_dim, 5)

    def forward(self, hidden_state):
        hidden_state = self.linear(hidden_state)
        return torch.log_softmax(hidden_state, dim=-1)


def set_seed(seed):
    """Set random seed."""
    if seed == -1:
        seed = random.randint(0, 1000)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train(model, inputs, targets, l0=False, epochs=50, batch_size=16, lr=0.00025):
    """
    Train a simple linear classifier to predict one of 5 targets.

    Args:
        model (DiagnosticClassifier): pre-initialised linear classifier
        inputs (list of vectors): hidden states extracted form a neural net
        targets (list of ints): integers indicating the inflection class
        l0 (bool): whether to train a sparse classifier or not
        epochs (int): number of epochs to train for
        batch_size (int): batch size
        lr (float): learning rate

    Returns:
        model, custom DiagnosticClassifier object
    """
    model.train()

    # Optimizer shouldn't matter, just need one to collect all the grads
    loss_function = torch.nn.NLLLoss()
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    indices = list(range(0, len(inputs)))
    for _ in range(epochs):
        random.shuffle(indices)
        indices = list(indices)
        losses = []

        for i in range(0, len(indices), batch_size):
            inp = torch.FloatTensor([inputs[j] for j in indices[i:i + batch_size]])
            tgt = torch.FloatTensor([targets[j] for j in indices[i:i + batch_size]])
            if inp.shape[-1] == 0:
                continue

            # Forward pass, collect gradient
            outputs = model(inp)
            optim.zero_grad()

            # Error between actual class and prediction
            #tgt = torch.nn.functional.one_hot(tgt.long(), num_classes=5).float()
            penalty = 0 if not l0 else (model.linear.l0_norm() * 0.005)
            loss = loss_function(outputs, tgt.long()) + penalty
            loss.backward()
            losses.append(loss.item())
            optim.step()
            model.eval()

    if l0:
        # Report how many neurons are zeroed out in the mask
        mask = model.linear._get_mask()
        logging.info(torch.sum(mask == 0, dim=1) / 512)
    return model


def score(model, dataset):
    """
    Computes F1-scores with baselines.

    Args:
        samples (list): list of tuples with (target, prediction)

    Returns:
        macro_f1 (float): F1 score macro-average for all classes
        baseline_max (float): F1 if always choosing the same class
        baseline_random (float): F1 if random shuffling predictions
        f1s_per_class (dict): F1s per inflection class
    """
    model.eval()
    pred = torch.argmax(model(torch.FloatTensor(dataset.X)), dim=-1)
    true = dataset.y

    # Macro-averaged F1 for multi-class classification
    macro_f1 = sklearn.metrics.f1_score(true, pred, average="macro")

    # F1s per class
    f1s_per_class = dict()
    for inflection_class in range(5):
        f1s_per_class[inflection_class] = sklearn.metrics.f1_score(
            [1 if p == inflection_class else 0 for p in true],
            [1 if p == inflection_class else 0 for p in pred],
            average="binary", pos_label=1)
    logging.info(f1s_per_class)

    # Baseline for always choosing one class
    baseline_max = max([
        sklearn.metrics.f1_score(true, [z] * len(true), average="macro")
        for z in range(5)])

    # Baseline by predicting the classes according to the freqency in the data
    random.shuffle(pred)
    baseline_random = sklearn.metrics.f1_score(true, pred, average="macro")
    return macro_f1, baseline_max, baseline_random, f1s_per_class


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_epochs', nargs='+', type=int, required=True)
    parser.add_argument("--target_type", type=str, default="prediction")
    parser.add_argument("--models_path", type=str, required=True)
    parser.add_argument('--l0', action="store_true")
    parser.add_argument("--dc_seed", type=int, nargs='+')
    parser.add_argument("--model_seed", type=int, nargs='+')
    parser.add_argument("--enc_type", type=str, default="h_c_enc")
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--target_pos_train", default=-1)
    parser.add_argument("--target_pos_test", type=int, default=-1)
    parser.add_argument("--sklearn", action="store_true")

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--controltask', action="store_true")

    args = parser.parse_args()
    if args.target_pos_train != "all":
        args.target_pos_train = int(args.target_pos_train)
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))
    filterwarnings('ignore')
    set_seed(1)

    categorise_fn = 'control' if args.controltask else None

    for epoch in args.model_epochs:
        scores = defaultdict(list)

        # Iterate over model seeds
        for model_seed in args.model_seed:
            models_path = args.models_path.replace(
                f"seed=1", f"seed={model_seed}")

            # Load data. If target pos train == target pos test, use the same
            # data handler for train and test, otherwise load test separately
            dataset_train = DataHandler(
                epoch, args.enc_type, args.target_type,
                data_path=models_path, target_pos=args.target_pos_train, categorise_fn=categorise_fn)
            if args.target_pos_train == args.target_pos_test:
                dataset_test = dataset_train
            else:
                dataset_test = DataHandler(
                    epoch, args.enc_type, args.target_type,
                    data_path=models_path, target_pos=args.target_pos_test, categorise_fn=categorise_fn)

            # Train 5 DC seeds for this model
            for dc_seed in args.dc_seed:
                set_seed(dc_seed)
                logging.info(
                    f"Epoch {epoch}, model {model_seed}, DC {dc_seed}")

                model = DiagnosticClassifier(args.l0, args.hidden_dim)
                model = train(
                    model, dataset_train.train.X, dataset_train.train.y,
                    args.l0, args.epochs, args.batch_size, args.lr)

                # Report statistics to the user
                macro_f1_test, bl_f1_test_max, bl_f1_test_random, f1s = score(
                    model, dataset_test.test)
                for x in f1s:
                    scores[x].append(f1s[x])
                macro_f1_train, bl_f1_train_max, bl_f1_train_random, _ = score(
                    model, dataset_train.train)
                logging.info(
                    f"Train: {macro_f1_train:.3f} ({bl_f1_train_max:.3f}, {bl_f1_train_random:.3f}), " +
                    f"Test: {macro_f1_test:.3f} ({bl_f1_test_max:.3f}, {bl_f1_test_random:.3f})")

                # Save DC parameters to file
                control_str = '_control' if args.controltask else ''
                dc_filename = \
                    f"parameters_enctype={args.enc_type}_" + \
                    f"target={args.target_type}_epoch={epoch}_l0={args.l0}" + \
                    f"_model-seed={model_seed}_dc-seed={dc_seed}_lr={args.lr}_" + \
                    f"timestep={args.target_pos_train}{control_str}.pt"
                torch.save(model, f"dcs_pytorch/{dc_filename}")

        # Report means and stds for the current model seed
        means, stds = [], []
        for x in [0, 1, 2, 3, 4]:
            means.append(str(round(np.mean(scores[x]), 3)))
            stds.append(str(round(np.std(scores[x]), 3)))
        logging.info("[" + ','.join(means) + "]")
        logging.info("[" + ','.join(stds) + "]")
