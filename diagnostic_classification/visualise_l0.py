import random
import argparse
import numpy as np
import torch
from train_dc import DiagnosticClassifier
from data import DataHandler
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from collections import defaultdict, Counter
import sys
import unidecode
import enchant
d = enchant.Dict("en_US")
sns.set_context("talk")


def visualise(EPOCH, MODEL_SEED, to_be_kept):
    for i in range(5):
        encodings, genders, lengths, first_l = [], [], [], []
        model_predictions, dc_predictions = [], []
        last_one, last_two, last_three = [], [], []
        vowels = []
        umlaut, loan = [], []

        frequencies_last_one, frequencies_last_two, frequencies_last_three = \
            Counter(), Counter(), Counter()

        # Load the encodings per inflection class
        per_category = defaultdict(list)
        for source, target in data.encodings:
            frequencies_last_one[source.replace(" </s>", "")[-1]] += 1
            frequencies_last_two[source.replace(" </s>", "")[-3:]] += 1
            frequencies_last_three[source.replace(" </s>", "")[-5:]] += 1

            prediction, _ = data.encodings[(source, target)]
            if source not in train:
                continue
            cat = data.categorise(source, prediction)
            if len(per_category[cat]) < 500:
                per_category[cat].append((source, target))

        frequencies_last_one = sorted(list(zip(*frequencies_last_one.most_common()[:5]))[0])
        frequencies_last_two = sorted(list(zip(*frequencies_last_two.most_common()[:5]))[0])
        frequencies_last_three = sorted(list(zip(*frequencies_last_three.most_common()[:5]))[0])

        # Compute truncated encodings
        keys = [key for c in per_category for key in per_category[c]]
        for source, target in keys:
            prediction, _ = data.encodings[(source, target)]
            cat = data.categorise(source, prediction)
            if cat == 5 or cat == 6:
                continue

            # Store gender and length
            genders.append(source.split()[0])
            lengths.append(len(source.split()))
            last_one.append(
                source.replace(" </s>", "")[-1] if source.replace(" </s>", "")[-1] in frequencies_last_one else None)
            last_two.append(
                source.replace(" </s>", "")[-3:] if source.replace(" </s>", "")[-3:] in frequencies_last_two else None)
            last_three.append(
                source.replace(" </s>", "")[-5:] if source.replace(" </s>", "")[-5:] in frequencies_last_three else None)
            last_letter = unidecode.unidecode(source.replace(" </s>", ""))[-1]
            vowels.append(last_letter if last_letter in ['a', 'o', 'i', 'u', 'e', 'y'] else None)

            # Store ED predictions and DC predictions
            outputs = dc(torch.FloatTensor(data.encodings[(source, target)][1]))
            model_predictions.append(labels[cat])
            predicted_cat = torch.argmax(outputs).item()
            dc_predictions.append(labels[predicted_cat] if predicted_cat == i else "other")
            umlaut.append("umlaut" if target != unidecode.unidecode(target) else "other")
            loan.append("loan" if d.check("".join(source.split()[1:-1])) else "non-loan")

            truncated_encoding = []
            for x, y in zip(data.encodings[(source, target)][1], to_be_kept[i]):
                if y:
                    truncated_encoding.append(x)
            encodings.append(truncated_encoding)

        # Run TSNE
        colours_gender = sns.color_palette("Spectral", 3)
        colours_gender[1] = "purple"

        colours_letters = sns.color_palette("Spectral", 5)
        colours_letters[2] = "purple"

        colours = {i : c for i, c in enumerate(sns.color_palette("Spectral", 5))}
        colours[2] = "purple"
        for j in range(5):
            colours[labels[j]] = colours[j]
        colours["other"] = "white"
        tsne_encodings = TSNE(n_components=2, perplexity=50, n_iter=1000).fit_transform(encodings)
        random_indices = list(range(len(tsne_encodings)))
        x, y = zip(*tsne_encodings)
        indices = list(range(len(x)))
        random.shuffle(indices)
        indices = indices[:2500]
        print(len(x))
        x = [x[k] for k in indices]
        y = [y[k] for k in indices]

        lab = labels[i].replace('$\\o$', 'eps')

        # Visualise the scatterplots
        fig, axes = plt.subplots(1, 10, figsize=(43, 4), sharex=True, sharey=True)
        hues = [model_predictions, dc_predictions, genders, lengths, vowels,
                umlaut, loan, last_one, last_two, last_three]

        for k, hue in enumerate(hues):
            print(k, len(hue))
            a = [hue[l] for l in indices]
            hues[k] = a

        colour_schemes = [colours, colours, colours_gender, "Spectral", "Spectral",
                          "Spectral", "Spectral", colours_letters, colours_letters, colours_letters]

        for ax, hue, colour_scheme in zip(axes, hues, colour_schemes):
            sns.scatterplot(x=x, y=y, hue=hue, alpha=0.4, palette=colour_scheme, ax=ax, edgecolors=None)
            plt.legend(bbox_to_anchor=(1, 1.05))
            plt.xlabel("")
            plt.ylabel("")

        plt.savefig(
            f"figures/tsne_class={lab}_epoch={EPOCH}_model-seed={MODEL_SEED}.pdf",
            bbox_inches="tight")
        plt.show()

        names = ["ED-predictions", "DC-predictions", "gender", "length", "vowel",
                 "umlaut", "loan", "last_letter", "last_two", "last_three"]
        for name, hue, colour_scheme in zip(names, hues, colour_schemes):
            plt.figure(figsize=(4.8, 4.8))
            sns.scatterplot(x=x, y=y, hue=hue, alpha=0.4, edgecolors=None,
                palette=colour_scheme)
            plt.xlabel("")
            plt.ylabel("")

            plt.savefig(
                f"figures/tsne_class={lab}_epoch={EPOCH}_model-seed={MODEL_SEED}_colour-scheme={name}.pdf",
                bbox_inches="tight")
            plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=5)
    parser.add_argument("--model_seed", type=int, default=1)
    args = parser.parse_args()

    sys.path.append("../opennmt/")

    all_dcs = []
    for seed in range(1, 6):
        filename = f"dcs_pytorch/parameters_enctype=h_c_enc_" + \
                   f"target=prediction_epoch={args.epoch}_l0=True" + \
                   f"_model-seed={args.model_seed}_dc-seed={seed}_lr=0.001_" + \
                   f"timestep=-1.pt"
        dc = torch.load(filename)
        weights_to_be_kept = dc.linear._get_mask() > 0
        all_dcs.append(weights_to_be_kept)

    to_be_kept = []
    for class_ in range(5):
        class_dc = []
        for i in range(512):
            class_dc.append(all([all_dcs[seed - 1][class_][i].item() for seed in range(1, 6)]))
        to_be_kept.append(class_dc)

    data = DataHandler(
        args.epoch, "h_c_enc", "prediction",
        f"../opennmt/models/seed={args.model_seed}_wiktionary", target_pos=-1)

    labels = ["-(e)n", "-e", r"-$\o$", "-er", "-s"]
    train = [x.strip() for x in open(
            "../wiktionary/wiktionary_train.src", encoding="utf-8").readlines()]

    visualise(args.epoch, args.model_seed, to_be_kept)
