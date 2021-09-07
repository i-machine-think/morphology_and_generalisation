from onmt.decoders import decoder
from onmt.encoders import rnn_encoder
from onmt.modules import embeddings
from onmt.models import NMTModel
from onmt.translate.translator import Translator
from onmt.translate import GNMTGlobalScorer, TranslationBuilder
from onmt.inputters import inputter
from argparse import Namespace
import torch
import torch.nn as nn


def get_iter_builder(filename, vocab_fields):
    i = inputter.DatasetLazyIter(
        dataset_paths=[filename], fields=vocab_fields,
        batch_size=1, batch_size_multiple=1, batch_size_fn=None,
        device=torch.device("cpu"),
        is_train=False, repeat=False, pool_factor=1)
    b = TranslationBuilder(data=torch.load(filename), fields=vocab_fields)
    return i, b

def make_translator(m, beamsearch=1):
    """
    Given a loaded model using torch.load(), reconstruct an OpenNMT object
    """
    VOC_SIZE_ENC = 70
    VOC_SIZE_DEC = 70
    EMB_SIZE = 128
    BIDIRECTIONAL = False
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    emb = embeddings.Embeddings(EMB_SIZE, VOC_SIZE_ENC, 1)
    emb.word_lut.weight.data.copy_(
        m["model"]["encoder.embeddings.make_embedding.emb_luts.0.weight"]
    )

    emb_decoder = embeddings.Embeddings(EMB_SIZE, VOC_SIZE_DEC, 1)
    emb_decoder.word_lut.weight.data.copy_(
        m["model"]["decoder.embeddings.make_embedding.emb_luts.0.weight"]
    )

    enc = rnn_encoder.RNNEncoder(
        "LSTM", BIDIRECTIONAL, NUM_LAYERS, hidden_size=HIDDEN_SIZE, embeddings=emb
    )
    with torch.no_grad():
        # enc.rnn.weight_ih_l0.data.copy_(m["model"]["encoder.rnn.weight_ih_l0"])
        # enc.rnn.weight_hh_l0.data.copy_(m["model"]["encoder.rnn.weight_hh_l0"])

        # enc.rnn.bias_ih_l0.data.copy_(m["model"]["encoder.rnn.bias_ih_l0"])
        # enc.rnn.bias_hh_l0.data.copy_(m["model"]["encoder.rnn.bias_hh_l0"])

        # enc.rnn.weight_ih_l1.data.copy_(m["model"]["encoder.rnn.weight_ih_l1"])
        # enc.rnn.weight_hh_l1.data.copy_(m["model"]["encoder.rnn.weight_hh_l1"])

        # enc.rnn.bias_ih_l1.data.copy_(m["model"]["encoder.rnn.bias_ih_l1"])
        # enc.rnn.bias_hh_l1.data.copy_(m["model"]["encoder.rnn.bias_hh_l1"])
        enc.rnn2.layers[0].i2h.weight.copy_(m['model']['encoder.rnn2.layers.0.i2h.weight'])
        enc.rnn2.layers[0].h2h.weight.copy_(m['model']['encoder.rnn2.layers.0.h2h.weight'])
        enc.rnn2.layers[0].i2h.bias.copy_(m['model']['encoder.rnn2.layers.0.i2h.bias'])
        enc.rnn2.layers[0].h2h.bias.copy_(m['model']['encoder.rnn2.layers.0.i2h.bias'])

        enc.rnn2.layers[1].i2h.weight.copy_(m['model']['encoder.rnn2.layers.1.i2h.weight'])
        enc.rnn2.layers[1].h2h.weight.copy_(m['model']['encoder.rnn2.layers.1.h2h.weight'])
        enc.rnn2.layers[1].i2h.bias.copy_(m['model']['encoder.rnn2.layers.1.i2h.bias'])
        enc.rnn2.layers[1].h2h.bias.copy_(m['model']['encoder.rnn2.layers.1.h2h.bias'])

    dec = decoder.InputFeedRNNDecoder(
        "LSTM", BIDIRECTIONAL, NUM_LAYERS, HIDDEN_SIZE, embeddings=emb_decoder
    )
    with torch.no_grad():
        dec.rnn.layers[0].i2h.weight.copy_(m['model']['decoder.rnn.layers.0.i2h.weight'])
        dec.rnn.layers[0].h2h.weight.copy_(m['model']['decoder.rnn.layers.0.h2h.weight'])
        dec.rnn.layers[0].i2h.bias.copy_(m['model']['decoder.rnn.layers.0.i2h.bias'])
        dec.rnn.layers[0].h2h.bias.copy_(m['model']['decoder.rnn.layers.0.i2h.bias'])

        dec.rnn.layers[1].i2h.weight.copy_(m['model']['decoder.rnn.layers.1.i2h.weight'])
        dec.rnn.layers[1].h2h.weight.copy_(m['model']['decoder.rnn.layers.1.h2h.weight'])
        dec.rnn.layers[1].i2h.bias.copy_(m['model']['decoder.rnn.layers.1.i2h.bias'])
        dec.rnn.layers[1].h2h.bias.copy_(m['model']['decoder.rnn.layers.1.h2h.bias'])

        # dec.attn.linear_in.weight.data.copy_(m['model']['decoder.attn.linear_in.weight'])
        # dec.attn.linear_out.weight.data.copy_(m['model']['decoder.attn.linear_out.weight'])


    final_model = NMTModel(enc, dec)

    # MANUALLY Set all the options for the translator.
    # Using openNMT defaults / additions from paper.

    opt = Namespace()
    opt.random_sampling_topk = 1
    opt.gpu = 0
    opt.fp32 = False
    opt.alpha = 0.7
    opt.beta = 0.0
    opt.n_best = 1
    opt.phrase_table = ""
    opt.report_time = False
    opt.verbose = True
    opt.batch_type = "sents"
    opt.beam_size = beamsearch
    opt.dump_beam = ""
    opt.data_type = "text"
    opt.seed = 11
    opt.stepwise_penalty = False
    opt.replace_unk = False
    opt.report_align = False
    opt.length_penalty = "avg"
    opt.coverage_penalty = "none"
    opt.ignore_when_blocking = []
    opt.max_length = 100
    opt.random_sampling_temp = 1.0
    opt.min_length = 0
    opt.ratio = 0.0
    opt.block_ngram_repeat = 0
    #opt.log_file = "Hi.txt"

    # Default scorer
    scorer = GNMTGlobalScorer(
        alpha=0.0, beta=0.0, length_penalty="avg", coverage_penalty="none"
    )

    # Default generator
    final_model.generator = nn.Sequential(
        nn.Linear(HIDDEN_SIZE, VOC_SIZE_DEC), nn.LogSoftmax(dim=-1)
    )
    final_model.generator[0].weight.data.copy_(m["generator"]["0.weight"])
    final_model.generator[0].bias.data.copy_(m["generator"]["0.bias"])
    # final_model = final_model.to(device)

    # Translator needs an out_file, just set a temporary file.
    with open(".temp.out", "w") as f:
        translator = Translator.from_opt(
            model=final_model,
            fields=m["vocab"],
            opt=opt,
            model_opt=m["opt"],
            global_scorer=scorer,
            out_file=f,
        )
    return translator