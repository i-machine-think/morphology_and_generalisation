import pandas as pd
import unidecode
import logging
logging.getLogger().setLevel(logging.INFO)

"""
Upon import, true outputs and predictions are loaded
"""
SEED_LIST = [1, 2, 3, 4, 5]

srcs = dict()
tgts = dict()
srcs['all'] = []
tgts['all'] = []
logging.info("Loading true data")
for dataset in ["train", "val", "test"]:
    srcs[dataset] = []
    tgts[dataset] = []
    with open(f"../wiktionary/wiktionary_{dataset}.src", "r") as f:
        srcs[dataset] += f.readlines()
        srcs['all'] += srcs[dataset]

    with open(f"../wiktionary/wiktionary_{dataset}.tgt", "r") as f:
        tgts[dataset] += f.readlines()
        tgts['all'] += tgts[dataset]

logging.info("Loading prediction data")
preds = dict()
for seed in SEED_LIST:
    preds[seed] = {}
    preds[seed]['all'] = []
    for dataset in ["train", "val", "test"]:
        preds[seed][dataset] = []
        with open(f"../opennmt/models/seed={seed}_wiktionary/wiktionary/lstms2s_{dataset}.prd", "r") as f:
            preds[seed][dataset] += f.readlines()
            preds[seed]['all'] += preds[seed][dataset]


def categorise_(source, target):
    """
    Return a numerical label indicating inflection class.
    Args:
        source (str): source with gender as first character ("<f> k a t z e ")
        target (str): (predicted) target
    Returns:
        int indicating class, ranges from 0 - 6
    """
    assert "<" in source.split()[0], "Your source sequence has no gender tag!"

    source = source.replace(" </s>", "")
    target = target.replace(" </s>", "")
    source = unidecode.unidecode(source).split()[1:]
    target = unidecode.unidecode(target).split()

    # zero or epsilon
    if target == source:
        category = "empty"
    elif len(target) > len(source) and target[:len(source)] == source:
        # (e)n
        if target[-1:] == ["n"]:
            category = "en"
        # e
        elif target[-1:] == ["e"]:
            category = "e"
        # e r
        elif target[-2:] == ["e", "r"]:
            category = "er"
        # s
        elif target[-1:] == ["s"]:
            category = "s"
        # repeated input but odd suffix
        elif target[:len(source)] == source:
            category = "other"
    else:
        # didn't even repeat the input
        category = "other"
    return category


def umlaut_remover(word):
    return word.replace("ä", "a").replace("ü", "u").replace("ö", "o")


def transform_dataframe(df):
    df['last_letter'] = df['in'].str[-2:]
    df['last_two_letter'] = df['in'].str[-4:]
    df["combo"] = df['last_letter'] + df['gender']
    df["combo_two"] = df['last_two_letter'] + df['gender']
    df["combo_three"] = df['last_two_letter'] + \
        df['gender'] + df['len'].apply(str)
    return df


def create_dataframe(setname='train', seed=1):
    gender = []
    orig_input = []
    orig_output = []
    lengths = []
    orig_class = []
    umlaut_change = []
    umlaut_change_tgt = []
    pred_class = []
    predis = []

    for src, tgt, prd in zip(srcs[setname], tgts[setname], preds[seed][setname]):
        src = src.strip().replace('</s>', '')
        tgt = tgt.strip()
        prd = prd.strip()

        gender.append(src[1])
        orig_input.append(src[3:])
        orig_output.append(tgt)
        orig_class.append(categorise_(src, tgt))
        pred_class.append(categorise_(src, tgt))
        predis.append(prd)
        umlaut_change.append(umlaut_remover(
            src) != src and umlaut_remover(tgt) == tgt)
        umlaut_change_tgt.append(umlaut_remover(
            src) == src and umlaut_remover(tgt) != tgt)

        lengths.append(len(src[3:].replace(" ", "").replace("<SPACE>", " ")))

    df = pd.DataFrame({'gender': gender, 'in': orig_input, 'out': orig_output,
                       'orig_class': orig_class, 'len': lengths,
                      'umlautsrc': umlaut_change, 'umlaut_tgt': umlaut_change_tgt,
                       'preds': predis,
                       'pred_class': pred_class})
    df = transform_dataframe(df)
    return df


def create_dataframes_and_run_baseline(seed=1):
    """
    Calculates majority predictions based on several features from the trainset - 
    Then applies this prediction to all examples in all sets.
    """
    # Ugly cache to make everything go faster
    lastletter_cache = dict()
    last2letter_cache = dict()
    len_cache = dict()
    combo_cache = dict()
    combo2_cache = dict()
    combo3_cache = dict()

    logging.info("Creating dataframes for seed %s" % seed)
    df_train = create_dataframe('train', seed)
    df_val = create_dataframe('val', seed)
    df_test = create_dataframe('test', seed)

    def classify(gender):
        """Returns majority baseline based on gender tag only"""
        if gender == 'f':
            return "en"
        if gender == 'm':
            return "e"
        if gender == 'n':
            return 'e'

    def classify_letter(last_letter):
        """
        Returns majority baseline based on last letter.
        Uses train set to get the majority for that letter.
        If no sample in the training set ends with that letter, simply choose the majority
        """
        if last_letter in lastletter_cache:
            return lastletter_cache[last_letter]
        try:
            result = df_train[df_train.last_letter ==
                              last_letter].pred_class.value_counts().idxmax()
        except:
            logging.debug("Failed to classify %s " % last_letter)
            result = 'en'
        lastletter_cache[last_letter] = result
        return result

    def classify_last_two_letter(l):
        if l in last2letter_cache:
            return last2letter_cache[l]
        try:
            result = df_train[df_train.last_two_letter ==
                              l].pred_class.value_counts().idxmax()
        except:
            logging.debug("Failed to classify %s" % l)
            result = 'en'
        last2letter_cache[l] = result
        return result

    def classify_len(l):
        if l in len_cache:
            return len_cache[l]
        result = df_train[df_train.len == l].pred_class.value_counts().idxmax()
        len_cache[l] = result
        return result

    def classify_combo(l):
        if l in combo_cache:
            return combo_cache[l]
        try:
            result = df_train[df_train.combo ==
                              l].pred_class.value_counts().idxmax()
        except:
            result = 'en'
        combo_cache[l] = result
        return result

    def classify_combo2(l):
        if l in combo2_cache:
            return combo2_cache[l]
        try:
            result = df_train[df_train.combo_two ==
                              l].pred_class.value_counts().idxmax()
        except:
            result = 'en'
        combo2_cache[l] = result
        return result

    def classify_combo3(l):
        if l in combo3_cache:
            return combo3_cache[l]
        try:
            result = df_train[df_train.combo_three ==
                              l].pred_class.value_counts().idxmax()
        except:
            result = 'en'
            logging.debug("Combo not found %s" % l)
        combo3_cache[l] = result
        return result

    def apply_baseline(df):
        df['baseline_gender'] = df['gender'].apply(classify)
        df['baseline_letter'] = df['last_letter'].apply(classify_letter)
        df['baseline_letter2'] = df['last_two_letter'].apply(
            classify_last_two_letter)
        df['baseline_len'] = df['len'].apply(classify_len)
        df['baseline_combo'] = df['combo'].apply(classify_combo)
        df['baseline_combo2'] = df['combo_two'].apply(classify_combo2)
        df['baseline_combo3'] = df['combo_three'].apply(classify_combo3)

        return df
    df_train = apply_baseline(df_train)
    df_val = apply_baseline(df_val)
    df_test = apply_baseline(df_test)
    return df_train, df_val, df_test
