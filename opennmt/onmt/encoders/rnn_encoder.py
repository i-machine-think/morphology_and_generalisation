"""Define RNN-based encoders."""
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack

from onmt.encoders.encoder import EncoderBase
from onmt.utils.rnn_factory import rnn_factory
from onmt.models.stacked_rnn import StackedLSTM
import logging
import torch


class RNNEncoder(EncoderBase):
    """ A generic recurrent neural network encoder.

    Args:
       rnn_type (str):
          style of recurrent unit to use, one of [RNN, LSTM, GRU, SRU]
       bidirectional (bool) : use a bidirectional RNN
       num_layers (int) : number of stacked layers
       hidden_size (int) : hidden size of each layer
       dropout (float) : dropout value for :class:`torch.nn.Dropout`
       embeddings (onmt.modules.Embeddings): embedding module to use
    """

    def __init__(self, rnn_type, bidirectional, num_layers,
                 hidden_size, dropout=0.0, embeddings=None,
                 use_bridge=False):
        super(RNNEncoder, self).__init__()
        assert embeddings is not None

        num_directions = 2 if bidirectional else 1
        assert hidden_size % num_directions == 0
        hidden_size = hidden_size // num_directions
        self.hidden_size = hidden_size
        self.embeddings = embeddings
        self.rnn, self.no_pack_padded_seq = \
            rnn_factory(rnn_type,
                        input_size=embeddings.embedding_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        bidirectional=bidirectional)

        self.rnn2 = StackedLSTM(num_layers=num_layers, input_size=embeddings.embedding_size,
                               rnn_size=hidden_size, dropout=dropout)


        # Initialize the bridge layer
        self.use_bridge = use_bridge
        if self.use_bridge:
            self._initialize_bridge(rnn_type,
                                    hidden_size,
                                    num_layers)

    @classmethod
    def from_opt(cls, opt, embeddings):
        """Alternate constructor."""
        return cls(
            opt.rnn_type,
            opt.brnn,
            opt.enc_layers,
            opt.enc_rnn_size,
            opt.dropout[0] if type(opt.dropout) is list else opt.dropout,
            embeddings,
            opt.bridge)

    def forward(self, src, lengths=None):
        """See :func:`EncoderBase.forward()`"""
        self._check_args(src, lengths)

        emb = self.embeddings(src)
        s_len, batch, emb_dim = emb.size()

        # Use the "unpacked" custom LSTM to get gates, etc.
        if True: # not self.training:
            # Step 2: run the LSTM
            all_h, all_c, all_i, all_f, all_o = [], [], [], [], []
            dec_state = None
            for emb_t in emb.split(1):
                rnn_output, dec_state, i_, f_, o_ = self.rnn2(emb_t, dec_state)
                h_, c_ = dec_state
                all_h.append(h_)
                all_c.append(c_)
                all_i.append(i_)
                all_f.append(f_)
                all_o.append(o_)

            all_h = torch.stack(all_h, dim=0)
            all_c = torch.stack(all_c, dim=0)
            all_i = torch.stack(all_i, dim=0)
            all_f = torch.stack(all_f, dim=0)
            all_o = torch.stack(all_o, dim=0)

            # Step 3: get the last timestep before padding using the lengths
            memory_bank = all_h[:, -1, :, :]
            hs, cs = [], []
            for i in range(batch):
                hs.append(all_h[lengths[i] - 1, :, i, :])
                cs.append(all_c[lengths[i] - 1, :, i, :])
            encoder_final = (torch.stack(hs, dim=1), torch.stack(cs, dim=1))
        else:
            packed_emb = emb
            if lengths is not None and not self.no_pack_padded_seq:
                # Lengths data is wrapped inside a Tensor.
                lengths_list = lengths.view(-1).tolist()
                packed_emb = pack(emb, lengths_list)

            memory_bank, encoder_final = self.rnn(packed_emb)
            if lengths is not None and not self.no_pack_padded_seq:
                memory_bank = unpack(memory_bank)[0]

        #if self.use_bridge:
        #    encoder_final = self._bridge(encoder_final)
        return encoder_final, memory_bank, lengths, all_h.detach().cpu().numpy(), all_c.detach().cpu().numpy(), all_i.detach().cpu().numpy(), all_f.detach().cpu().numpy(), all_o.detach().cpu().numpy()

    def _initialize_bridge(self, rnn_type,
                           hidden_size,
                           num_layers):

        # LSTM has hidden and cell state, other only one
        number_of_states = 2 if rnn_type == "LSTM" else 1
        # Total number of states
        self.total_hidden_dim = hidden_size * num_layers

        # Build a linear layer for each
        self.bridge = nn.ModuleList([nn.Linear(self.total_hidden_dim,
                                               self.total_hidden_dim,
                                               bias=True)
                                     for _ in range(number_of_states)])

    def _bridge(self, hidden):
        """Forward hidden state through bridge."""
        def bottle_hidden(linear, states):
            """
            Transform from 3D to 2D, apply linear and return initial size
            """
            size = states.size()
            result = linear(states.view(-1, self.total_hidden_dim))
            return F.relu(result).view(size)

        if isinstance(hidden, tuple):  # LSTM
            outs = tuple([bottle_hidden(layer, hidden[ix])
                          for ix, layer in enumerate(self.bridge)])
        else:
            outs = bottle_hidden(self.bridge[0], hidden)
        return outs

    def update_dropout(self, dropout):
        self.rnn.dropout = dropout
