import csv
import random
import unidecode
from collections import defaultdict


def categorise( source, prediction):
    source = source.replace(" </s>", "")
    prediction = prediction.replace(" </s>", "")
    source = unidecode.unidecode(source).split()[1:]
    prediction = unidecode.unidecode(prediction).split()
    if prediction == source:
        return 2
    elif prediction[:len(source)] == source and prediction[-1:] == ["n"]:
        return 0
    elif prediction[:len(source)] == source and prediction[-1:] == ["e"]:
        return 1
    elif prediction[:len(source)] == source and prediction[-2:] == ["e", "r"]:
        return 3
    elif prediction[:len(source)] == source and prediction[-1:] == ["s"]:
        return 4
    elif prediction[:len(source)] == source:
        return 5
    else:
        return 6


genders = {x["lemma"]: x["genus"] for x in csv.DictReader(open("_GN_splits.csv", encoding="utf-8"))}

samples = csv.DictReader(open("_GN_splits.csv", encoding="utf-8"))
train, val, test = [], [], []
for x in samples:
    if x['genus']:
        src = f"<{x['genus']}> {' '.join(list(x['lemma']))}"
        tgt = f"{' '.join(list(x['pl_form']))}"
    # elif x["genus_approx_from"] in genders:
    #     src = f"<{genders[x['genus_approx_from']]}> {' '.join(list(x['lemma']))} <EOS>"
    #     tgt = f"{' '.join(list(x['pl_form']))} <EOS>"
    else:
        continue

    if x["split"] == "train":
        train.append((src, tgt))
    elif x["split"] == "val":
        val.append((src, tgt))
    else:
        test.append((src, tgt))

with open("wiktionary_train.src", 'w', encoding="utf-8") as f_src, \
     open("wiktionary_train.tgt", 'w', encoding="utf-8") as f_tgt:
    for src, tgt in train:
        f_src.write(src + "\n")
        f_tgt.write(tgt + "\n")

with open("wiktionary_val.src", 'w', encoding="utf-8") as f_src, \
     open("wiktionary_val.tgt", 'w', encoding="utf-8") as f_tgt:
    for src, tgt in val:
        f_src.write(src + "\n")
        f_tgt.write(tgt + "\n")

with open("wiktionary_test.src", 'w', encoding="utf-8") as f_src, \
     open("wiktionary_test.tgt", 'w', encoding="utf-8") as f_tgt:
    for src, tgt in test:
        f_src.write(src + "\n")
        f_tgt.write(tgt + "\n")


per_cat = defaultdict(list)
for s, t in train:
    cat = categorise(s, t)
    per_cat[cat].append((s, t))

print(per_cat.keys())


mini = min([len(per_cat[x]) for x in per_cat if x not in [5, 6]])

print(mini)

with open("wiktionary_all.src", 'w', encoding="utf-8") as f_src, \
     open("wiktionary_all.tgt", 'w', encoding="utf-8") as f_tgt:
    for x in per_cat:
        if x not in [5, 6]:
            for src, tgt in random.sample(per_cat[x], mini):
                f_src.write(src + "\n")
                f_tgt.write(tgt + "\n")
    for src, tgt in val + test:
        f_src.write(src + "\n")
        f_tgt.write(tgt + "\n")
