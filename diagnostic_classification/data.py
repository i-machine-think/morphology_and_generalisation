import pickle
import io
import random
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import unidecode
import json
import os
import logging

"""
Control task must be deterministic across train/val/test set
Ensure that all samples are classified according to same random mapping
"""
distribution = np.array([0.451, 0.26, 0.17, 0.035, 0.054, 0.032])
distribution = distribution / distribution.sum()
control_task = 2
control_file = f'.controltask{control_task}.json'

if not os.path.exists(control_file):
    empty = {'_TEMP_':1}
    with open(control_file, 'w') as f:
        json.dump(empty, f)


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        return super().find_class(module, name)


class CustomDataset(Dataset):
    """Dataset object."""

    def __init__(self, samples):
        self.samples = samples

    def __getitem__(self, idx):
        """Extract one sample with index idx.

        Args:
            idx (int): sample number

        Returns:
            tuple of src, encoding, tgt
        """
        return self.samples[idx]

    def __len__(self):
        """Compute number of samples in dataset."""
        return len(self.samples)

    @staticmethod
    def collate_fn(batch):
        """Gather the batches' text, lengths, labels and masks.

        Args:
            batch (list of tuples): texts, lengths and labels

        Returns:
            batch_text (list of list of str): words
            lengths (LongTensor): vector with sentence lengths
            labels (LongTensor): binary labels of metaphoricity
            mask (LongTensor): binary indications of words vs padding
        """
        src, enc, tgt = zip(*batch)
        return src, torch.FloatTensor(enc), torch.LongTensor(tgt)


class DataHandler():
    def __init__(self, step, enc_type="h_c", target_type="prediction",
                 data_path="data", target_pos=0, categorise_fn=None):
        self.samples = []
        self.labels = []
        self.encodings = dict()
        self.preprocess(step, enc_type, data_path, target_pos)
        random.seed(42)
        random.shuffle(self.samples)
        self.classify_samples(target_type, step, categorise_fn)

    def preprocess(self, step, enc_type, data_path="data", target_pos=0):
        x = 719
        contents = CPU_Unpickler(
            open(f"{data_path}/lstms2s_step_{x * step}_hidden_states.pickle", 'rb')).load()
        src_itos = contents["vocab"]["src"].base_field.vocab.itos
        tgt_itos = contents["vocab"]["tgt"].base_field.vocab.itos

        data = zip(contents["srcs"], contents["tgts"], contents["preds"],
                   contents["h_enc"], contents["c_enc"],
                   contents[enc_type if "h_c" not in enc_type else "h_enc"])

        for src, tgt, pred, all_h, all_c, all_custom_enc in data:
            # Reshape into batch_size x seq_length
            src = src.squeeze(-1).transpose(0, 1)
            tgt = tgt.squeeze(-1).transpose(0, 1)
            prd = torch.LongTensor(pred).transpose(0, 1)

            # Reshape into batch_size x 200, contracting the two directions
            all_h = torch.FloatTensor(all_h)
            all_c = torch.FloatTensor(all_c)
            all_custom_enc = torch.FloatTensor(all_custom_enc)

            if "h_c" in enc_type:
                encoding = torch.cat((all_h, all_c), dim=-1)
            else:
                encoding = all_custom_enc

            # Turns the src and tgt indices into readable strings
            for i, (s, t, p) in enumerate(zip(src, tgt, prd)):
                s2 = s
                enc = encoding[:, :, i, :]
                s = self.clean([src_itos[w] for w in s])
                t = self.clean([tgt_itos[w]
                               for w in t][1:-1]).replace(" </s>", "")
                p = self.clean([tgt_itos[w] for w in p])

                samples = []
                for k, char in enumerate(s.split()):
                    if target_pos == -1 and k == range(len(s.split()))[target_pos]:
                        assert char == "</s>"
                    if target_pos != "all":
                        # Only add time step k
                        if k == range(len(s.split()))[target_pos]:
                            vec = enc[k, :, :].contiguous().view(-1)
                            samples.append((s, t, p, k, vec.tolist()))
                            self.encodings[(s, t)] = (p, vec.tolist())
                    else:
                        # Add all time steps
                        vec = enc[k, :, :].contiguous().view(-1)
                        samples.append((s, t, p, k, vec.tolist()))
                        self.encodings[(s, t)] = (p, vec.tolist())

                if samples:
                    # Append the list, such that we can keep track of
                    # all samples that come from one source sequence
                    self.samples.append(samples)

    def load(self, filename_src, filename_tgt, categorise_fn=None):
        """
        Collect encodings and return them for data from filename_src.

        Args:
            filename_src (str): filename of source file
            filename_tgt (tgt): filename of target file

        Returns:
            list of tuples, with present tense - past tense - encoding - suffix
        """
        if categorise_fn is None:
            categorise_fn = self.categorise
        elif categorise_fn == 'control':
            categorise_fn = self.categorise_control

        data, srcs = [], []
        with open(filename_src, encoding="utf-8") as f_src:
            for line in f_src:
                srcs.append(line.strip())

        with open(filename_src, encoding="utf-8") as f_src, \
                open(filename_tgt, encoding="utf-8") as f_tgt:
            for present, past in zip(f_src, f_tgt):
                present = present.replace("   ", " ").strip()
                past = past.strip()
                category = categorise_fn(present, past)
                if (present, past) in self.encodings:
                    _, encoding = self.encodings[(present, past)]
                    data.append((present, past, encoding, category))
                else:
                    data.append((present, past, None, category))
        return data

    def classify_samples(self, target_type, step, categorise_fn=None):
        """
        Separate samples in train and test data to prep for training.

        Args:
            target_type (str): whether to train on prediction | target
            step (int): epoch number
        """
        if categorise_fn is None:
            categorise_fn = self.categorise
        elif categorise_fn == 'control':
            categorise_fn = self.categorise_control

        self.labels = []
        # Load train and test source sequences from Wiktionary
        train = [x.strip() for x in open(
            "../wiktionary/wiktionary_train.src", encoding="utf-8").readlines()]
        test = [x.strip() for x in open(
            "../wiktionary/wiktionary_val.src", encoding="utf-8").readlines()]

        train_samples, test_samples = [], []
        samples_per_class = defaultdict(list)

        for sample in self.samples:
            # Only keep data that is not category 5 or 6
            s, t, p, _, _ = sample[0]
            label_pred = categorise_fn(
                s, p if target_type == "prediction" else t)
            if label_pred not in [5, 6]:
                self.labels.append(label_pred)
            else:
                label_pred = None

            # Put samples into train or test categories
            if label_pred is not None and s in test:
                test_samples.append((sample, label_pred))
            elif label_pred is not None and s in train:
                samples_per_class[label_pred].append((sample, label_pred))

        logging.info([len(samples_per_class[x]) for x in range(7)
                      if x in samples_per_class])
        # AFter epoch 5 ensure the amount of data per class is the same for all
        if step > 5:
            mini = min([len(samples_per_class[x]) for x in samples_per_class])
            for x in samples_per_class:
                random.shuffle(samples_per_class[x])
                samples_per_class[x] = samples_per_class[x][:mini]

        # Create custom dataset object for train & test
        train_samples = [y for x in samples_per_class
                         for y in samples_per_class[x]]
        random.shuffle(train_samples)
        self.test = Subset(test_samples)
        self.train = Subset(train_samples)

    @staticmethod
    def categorise_control(source, target):
        """
        Return a numerical label indicating inflection class.
        Args:
            source (str): source with gender as first character ("<f> k a t z e ")
            target (str): (predicted) target
        Returns:
            int indicating class, ranges from 0 - 6
        """
        assert "<" in source.split(
        )[0], "Your source sequence has no gender tag!"

        source = source.replace(" </s>", "")
        target = target.replace(" </s>", "")

        tag, *source = unidecode.unidecode(source).split()
        target = unidecode.unidecode(target).split()
        with open(control_file, 'r') as f:
            controldict = json.load(f)

        identifier = tag + ''.join(source[-2:])

        if identifier in controldict:
            return controldict[identifier]
        else:
            category = int(np.random.choice(np.arange(6), 1, p=distribution)[0])
            controldict[identifier] = category
            logging.debug("Added new control mapping %s -> %s" % (identifier, category))
        with open(control_file, 'w') as f:
            json.dump(controldict, f)
        return category

    @staticmethod
    def categorise(source, target):
        """
        Return a numerical label indicating inflection class.
        Args:
            source (str): source with gender as first character ("<f> k a t z e ")
            target (str): (predicted) target
        Returns:
            int indicating class, ranges from 0 - 6
        """
        assert "<" in source.split(
        )[0], "Your source sequence has no gender tag!"

        source = source.replace(" </s>", "")
        target = target.replace(" </s>", "")
        source = unidecode.unidecode(source).split()[1:]
        target = unidecode.unidecode(target).split()

        # zero or epsilon
        if target == source:
            category = 2
        elif len(target) > len(source) and target[:len(source)] == source:
            # (e)n
            if target[-1:] == ["n"]:
                category = 0
            # e
            elif target[-1:] == ["e"]:
                category = 1
            # e r
            elif target[-2:] == ["e", "r"]:
                category = 3
            # s
            elif target[-1:] == ["s"]:
                category = 4
            # repeated input but odd suffix
            else:
                category = 5
        else:
            # didn't even repeat the input
            category = 6
        return category

    @staticmethod
    def clean(s):
        """
        Clean string by removing OpenNMT characters and break off after </s>.

        Args:
            s (list): tokenised string

        Returns:
            string with blanks removed & cut off after </s>
        """
        s = [x for x in s if x != "<blank>"]
        if "</s>" in s:
            eos = s.index("</s>")
            s = s[:eos + 1]
        return " ".join(s)


class Subset():
    def __init__(self, samples):
        samples = [tuple(z) + (y,) for x, y in samples for z in x]
        self.src, self.tgt, self.prd, self.pos, self.X, self.y = \
            zip(*samples)

        self.lengths = [len(x.split()) for x in self.prd]
        self.X = self.X
        self.y = np.array(self.y)

    def get_samples(self):
        return list(zip(self.src, self.tgt, self.prd, self.pos, self.lengths))
