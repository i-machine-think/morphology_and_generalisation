import sys
import os
import logging
import argparse
from collections import defaultdict
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

# Custom imports from own files in different folders
sys.path.append("../diagnostic_classification/")
sys.path.append("../opennmt/")
from make_translator import make_translator, get_iter_builder
from train_dc import set_seed, DiagnosticClassifier
from data import DataHandler

logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))


def get_gradients(prds, tgts, my_model, encodings, exclude_incorrect=False):
    """
    Use the Diagnostic Classifier to get the gradients w.r.t. the hidden states

    Args:
        prds (list): model predictions inflection classes
        tgts (list): data targets inflection classes
        my_model: a torch module
        encodings: the hidden states extracted from OpenNTM
        targets: the verb categories

    Returns:
        list of ALL the gradients wrt the inputs in the batch loader
        (Including gradients where the LSTM is not on the 'wrong path')
    """
    grads = []
    my_model.train()

    # Optimizer shouldn't matter, just need one to collect all the grads    
    loss_function = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(my_model.parameters(), lr=0.1)
    
    for p, t, encoding in zip(prds, tgts, encodings):
        # Skip if we don't have a target or encoding,
        # the prediction already equals the
        # target or the target is not well-formed
        if t is None or encoding is None or p == t or t in [5, 6] or p in [5, 6]:
            grads.append(None)
            continue
        encoding = torch.FloatTensor(encoding)

        # Collect model output class, DC predicted class and REAL class
        encoding.requires_grad = True # important
        target = torch.Tensor([t])

        # Forward pass, collect gradient
        outputs = my_model(encoding)
        if exclude_incorrect and not torch.argmax(outputs).item() == p:
            grads.append(None)
            continue
        optim.zero_grad()
        
        # Error between actual class and prediction
        #target = torch.nn.functional.one_hot(target.long(), num_classes=5).float()
        loss = loss_function(outputs.unsqueeze(0), target.long())
        loss.backward()
        grads.append(encoding.grad)

    return grads


def translate(cutoff, filename, translator, model, gradients=None, eta=0.5):
    """
    Translate the data present in FILENAME with interventions.

    Note: This will intervene on ALL the inputs, even where the model
    was going the right way.

    Args:
        cutoff (int): stop after this many translation have been processed
        filename (str): OpenNMT data .pt file
        translator: custom OpenNMT Translator object
        model: custom OpenNMT model object
        gradients (list): list of gradients in the order of the data in FILENAME
        eta (float): step size of intervention

    Returns:
        src: source extracted from the data .pt file
        predictions: list of predictions with intervention
    """
    iter_, the_builder = get_iter_builder(filename, model['vocab'])
    srcs, predictions = [], []

    for j, batch in enumerate(iter_):
        if gradients is not None and gradients[j] is None:
            predictions.append(None)

        else:
            if gradients is not None:
                intervene = gradients[j].clone() * eta
            else:
                intervene = None
            trans_batch = translator.translate_batch(
                batch=batch, src_vocabs=[model['vocab']["src"].base_field.vocab],
                attn_debug=False, intervention_grad=intervene)
            translations = the_builder.from_batch(trans_batch)

            for translation in translations:
                srcs.append(" ".join(translation.src_raw))
                predictions.append(" ".join(translation.pred_sents[0]))

        if j == cutoff:
            break
    return srcs, predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cutoff', type=int, default=-1)
    parser.add_argument('-l', '--learning_rate', type=float, default=1)
    parser.add_argument('--steps', nargs='+', type=int, required=True)
    parser.add_argument("--target_type", type=str, default="prediction")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dc_path", type=str, default="../diagnostic_classification/dcs_pytorch")
    parser.add_argument("--override_target", action="store_true")
    parser.add_argument("--focus_target", type=int, default=1)
    parser.add_argument("--model_seed", type=int, default=1)
    parser.add_argument("--dc_seed", type=int, default=1)
    parser.add_argument("--dc_lr", type=float, default=0.00025)
    parser.add_argument("--exclude_incorrect", action="store_true")
    parser.add_argument("--random_dc", action="store_true")
    parser.add_argument("--controltask", action="store_true")
    args = parser.parse_args()
    set_seed(42)
    if args.exclude_incorrect and args.controltask:
        logging.warning("Warning: Excluding incorrectly classified samples while using control task.")

    enc_type = "h_c_enc"
    control_str = '_control' if args.controltask else ''

    for epoch in args.steps:
        # Create a DC model from parameters fit scikit-learn in separate script
        if args.random_dc:
            dc = model = DiagnosticClassifier(False, 512)
        else:
            dc = torch.load(
                f"{args.dc_path}/parameters_enctype={enc_type}_target=" +
                f"{args.target_type}_epoch={epoch}_l0=False_model-seed="+
                f"{args.model_seed}_dc-seed={args.dc_seed}_lr={args.dc_lr}_timestep=-1" + control_str + ".pt"
            )

        # Create a dataset that can classify real and fake past tenses
        dataset = DataHandler(epoch, enc_type, args.target_type, args.model_path, target_pos=-1)

        # Load the validation data, create a dict to map src to tgt and encoding
        srcs, tgts, encodings, _ = zip(*dataset.load(
            "../wiktionary/wiktionary_val.src", "../wiktionary/wiktionary_val.tgt"))
        src_to_tgt = {s: t for s, t in zip(srcs, tgts)}
        src_to_enc = {s: e for s, e in zip(srcs, encodings)}
        cutoff = args.cutoff if args.cutoff > 0 else len(srcs)

        # Load a pretrained LSTMS2S model from a certain epoch
        model = torch.load(
            f"{args.model_path}/lstms2s_step_{719 * epoch}.pt",                   
            map_location=torch.device("cpu"))
        translator = make_translator(model, 5)
        srcs, prds = translate(
            cutoff, f"{args.data_path}/wiktionary_preprocessed.valid_only.0.pt", translator, model,
        )
        tgts = [src_to_tgt[s] if s in src_to_tgt else None for s in srcs]

        # Get the gradients from the DC w.r.t. the hidden states from that epoch
        prd_categories = [
            dataset.categorise(x, " ".join(y))
            if x in src_to_tgt else None for x, y in zip(srcs, prds)]

        if args.override_target:
            tgt_categories = [args.focus_target if s in src_to_tgt else None for s in srcs]
        else:
            tgt_categories = [dataset.categorise(s, src_to_tgt[s])
                              if s in src_to_tgt else None for s in srcs]

        encodings = [src_to_enc[s] if s in src_to_tgt else None for s in srcs]
        gradients = get_gradients(
            prd_categories, tgt_categories, dc, encodings,
            exclude_incorrect=args.exclude_incorrect)
        assert len(gradients) == len(encodings) == len(srcs)
        _, all_intervention_preds = translate(
            cutoff, f"{args.data_path}/wiktionary_preprocessed.valid_only.0.pt", translator, model,
            gradients, eta=args.learning_rate
        )

        num_grads = sum([1 for x in gradients if x is not None])
        logging.info(num_grads)
        trace = defaultdict(list)

        for focus_target in range(5):
            predictions, intervention_predictions, targets = [], [], []
            data = zip(srcs, prds, tgts, all_intervention_preds)
            for s, p, t, i in data:
                target_category = dataset.categorise(s, t)
                pred_category = dataset.categorise(s, p)
                if i is None or target_category != focus_target or \
                   pred_category == target_category or pred_category in [5, 6]:
                    continue

                int_category = dataset.categorise(s, i)
                intervention_predictions.append(int_category)
                predictions.append(pred_category)
                if args.override_target:
                    target_category = args.focus_target

                targets.append(target_category)
                trace[focus_target].append(
                    (s, t, target_category, p, pred_category, i, int_category)
                )

            a = accuracy_score(intervention_predictions, predictions)
            b = accuracy_score([focus_target for x in targets], predictions)
            c = accuracy_score(intervention_predictions, [focus_target for x in targets])
            logging.info(f"({focus_target}, {epoch}, {a}, {b}, {c}),")

        filename = \
            f"trace/trace_epoch={epoch}_lr={args.learning_rate}_model-seed=" + \
            f"{args.model_seed}_dc-seed={args.dc_seed}_dc-lr={args.dc_lr}_" + \
            f"exclude-incorrect={args.exclude_incorrect}_random-dc={args.random_dc}" + control_str + ".txt"
        with open(filename, 'w', encoding="utf-8") as f:
            for focus_target in trace:
                f.write(f"focus target {focus_target}\n")
                for s, t, t_c, p, p_c, i, i_c in trace[focus_target]:
                    s = ''.join(s.split())
                    p = ''.join(p.split())
                    i = ''.join(i.split())
                    t = ''.join(t.split())

                    if p_c == i_c:
                        c = "no_change"
                    elif p_c == t_c and i_c != t_c:
                        c = "new_error"
                    elif i_c == t_c and p_c != t_c:
                        c = "correction"
                    else:
                        c = "other"

                    f.write(
                        f"\t\t- source: {s}, prediction: {p}, intervention: {i}" +
                        f", target: {t}, change: {c}, {t_c}/{p_c}/{i_c}\n")
