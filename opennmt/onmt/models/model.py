""" Onmt NMT Model base class definition """
import torch.nn as nn
import torch


class NMTModel(nn.Module):
    """
    Core trainable object in OpenNMT. Implements a trainable interface
    for a simple, generic encoder + decoder model.

    Args:
      encoder (onmt.encoders.EncoderBase): an encoder object
      decoder (onmt.decoders.DecoderBase): a decoder object
    """

    def __init__(self, encoder, decoder):
        super(NMTModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.srcs = []
        self.tgts = []
        self.dec_outs = []
        self.preds = []
        self.h_enc = []
        self.c_enc = []
        self.i_enc = []
        self.f_enc = []
        self.o_enc = []
        self.h_dec = []
        self.c_dec = []
        self.i_dec = []
        self.f_dec = []
        self.o_dec = []

    def forward(self, src, tgt, lengths, bptt=False, with_align=False):
        """Forward propagate a `src` and `tgt` pair for training.
        Possible initialized with a beginning decoder state.

        Args:
            src (Tensor): A source sequence passed to encoder.
                typically for inputs this will be a padded `LongTensor`
                of size ``(len, batch, features)``. However, may be an
                image or other generic input depending on encoder.
            tgt (LongTensor): A target sequence passed to decoder.
                Size ``(tgt_len, batch, features)``.
            lengths(LongTensor): The src lengths, pre-padding ``(batch,)``.
            bptt (Boolean): A flag indicating if truncated bptt is set.
                If reset then init_state
            with_align (Boolean): A flag indicating whether output alignment,
                Only valid for transformer decoder.

        Returns:
            (FloatTensor, dict[str, FloatTensor]):

            * decoder output ``(tgt_len, batch, hidden)``
            * dictionary attention dists of ``(tgt_len, batch, src_len)``
        """
        dec_in = tgt[:-1]  # exclude last target from inputs

        enc_state, memory_bank, lengths, h_enc, c_enc, i_enc, f_enc, o_enc = self.encoder(src, lengths)

        if bptt is False:
            self.decoder.init_state(src, memory_bank, enc_state)
        dec_out, attns, h_dec, c_dec, i_dec, f_dec, o_dec = self.decoder(
            dec_in, memory_bank, memory_lengths=lengths, with_align=with_align)
        if not self.training:
            self.srcs.append(src)
            self.tgts.append(tgt)
            self.preds.append(torch.argmax(self.generator(dec_out), -1).detach().cpu().numpy())
            self.dec_outs.append(dec_out.detach().cpu().numpy())
            self.h_dec.append(h_dec)
            self.c_dec.append(c_dec)
            self.i_dec.append(i_dec)
            self.f_dec.append(f_dec)
            self.o_dec.append(o_dec)
            self.h_enc.append(h_enc)
            self.c_enc.append(c_enc)
            self.i_enc.append(i_enc)
            self.f_enc.append(f_enc)
            self.o_enc.append(o_enc)
        return dec_out, attns

    def update_dropout(self, dropout):
        self.encoder.update_dropout(dropout)
        self.decoder.update_dropout(dropout)
