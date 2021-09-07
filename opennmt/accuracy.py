"""
Script to compute (sequence) accuracy score.
"""

import csv
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--gold', type=str,
        help="File with targets."
    )
    parser.add_argument(
        '--pred', type=str,
        help="Output file of translate.py."
    )
    parser.add_argument(
        "--last_n", type=int, default=0
    )
    args = vars(parser.parse_args())

    correct, all_samples = 0, 0

    # Open the prediction files, that contain src, tgt & pred
    predictions = open(args["pred"]).readlines()
    targets = open(args["gold"]).readlines()

    for prediction, target in zip(predictions, targets):
        if prediction.strip()[args["last_n"]:] == target.strip()[args["last_n"]:]:
            correct += 1
        all_samples += 1

    # Report statistics to user
    print(f"Sequence Accuracy: {correct / all_samples}")

