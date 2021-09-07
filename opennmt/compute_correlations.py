import seaborn as sns
import numpy as np
import csv
import scipy.stats
import argparse

from matplotlib import pyplot as plt
from collections import defaultdict, Counter


def compute_correlations(filename):
    """
    Compute the correlation to human production probabilities,
    using the predictions of model with seed i.

    Args:
        - i: seed number

    Returns:
        - regular_correlation: correlation for regular past tense
        - irregular_correlation: correlation for irregular past tense
    """
    nonce_predictions = defaultdict(lambda : dict())
    nonce_human_reg = defaultdict(lambda: list())
    nonce_human_irreg = defaultdict(lambda: list())

    with open(filename, encoding="utf-8") as f:
        samples = f.read().split("SENT")[1:91]
        for i, s in enumerate(samples):
            source = s.split("BEST HYP:")[0].strip()
            source = eval(" ".join(source.split("\n")[0].split()[1:]))
            hypotheses = s.split("BEST HYP:")[1].strip()
            hypotheses = hypotheses.split("\n")
            for h in hypotheses:
                try:
                    log_prob = eval(h.split(" ")[0])[0]
                    past = eval("".join(h.split(" ")[1:]))
                except:
                    # print(h)
                    continue

                if "".join(past) not in nonce_predictions["".join(source)]:
                    nonce_predictions["".join(source)]["".join(past)] = np.exp(log_prob)

    with open("../celex/nonce_statistics.csv", encoding="utf-8") as f:
        f = csv.reader(f, delimiter=';')
        for i, line in enumerate(f):
            if i == 0 or i > 58:
                continue
            nonce_human_reg[line[2]] = (line[4], float(line[7]))
            nonce_human_irreg[line[2]] = (line[8], float(line[11]))

    nonce_predictions = dict(nonce_predictions)
    nonce_human_reg = dict(nonce_human_reg)
    nonce_human_irreg = dict(nonce_human_irreg)

    a = [nonce_human_irreg[w][1] for w in nonce_human_irreg]
    b = [nonce_predictions[w][nonce_human_irreg[w][0]]
         if nonce_human_irreg[w][0] in nonce_predictions[w] else 0 for w in nonce_human_irreg]

    irregular_correlation = scipy.stats.spearmanr(a, b)

    a = [nonce_human_reg[w][1] for w in nonce_human_reg]
    b = [nonce_predictions[w][nonce_human_reg[w][0]]
         if nonce_human_reg[w][0] in nonce_predictions[w] else 0 for w in nonce_human_reg]

    regular_correlation = scipy.stats.spearmanr(a, b)
    return regular_correlation, irregular_correlation


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="trace.tsv")
    args = parser.parse_args()

    r, i = compute_correlations(args.file)
    print(f"Regular correlation: {r[0]:.3f}, p={r[1]:.3f}")
    print(f"Irregular correlation: {i[0]:.3f}, p={i[1]:.3f}")
