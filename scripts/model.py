import torch
import torch.nn as nn
import torch.nn.functional as F

from indrnn import IndRNN


class DiacModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_vocab_size,
                 n_layers=1, bidirectional=True, dropout=0.0, n=11,
                 input2id=None, output2id=None,
                 model_type='gru', out_dropout=0.0, finish_type='flatten'):
        super(DiacModel, self).__init__()

        self.n = n
        self.input2id = input2id
        self.output2id = output2id

        self.embed = nn.Embedding(vocab_size, embed_size)
        model = {'gru': nn.GRU,
                 'lstm': nn.LSTM,
                 'indrnn': IndRNN,
                 'rnn': nn.RNN}
        model_instance = model[model_type]
        self.gru = model_instance(embed_size, hidden_size, num_layers=n_layers,
                                  dropout=dropout, bidirectional=bidirectional)

        self.hidden_size = hidden_size
        self.bidirectional_multiplier = 2 if bidirectional else 1
        self.finish_type = finish_type

        h2o_multiplier_dict = {
            'attention': 2,
            'central_only': 1,
            'flatten': n
        }

        h2o_multiplier = h2o_multiplier_dict[finish_type]

        h_size = hidden_size * self.bidirectional_multiplier * h2o_multiplier

        self.h2o = nn.Linear(h_size, output_vocab_size)

        self.attn = nn.Linear(hidden_size * self.bidirectional_multiplier,
                              self.n)

        self.attn_lin = nn.Linear(self.n, self.n)

        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(out_dropout)

    def forward(self, inp):
        emb = self.embed(inp)
        emb = self.dropout(emb)

        input = emb
        # input = emb.view(len(inp), 1, self.hidden_size)

        gru_out, hidden = self.gru(input)
        if hasattr(self, 'out_dropout'):
            gru_out = self.out_dropout(gru_out)

        final_output = gru_out.view(gru_out.size(0),
                                    gru_out.size(1) * gru_out.size(2))
        if hasattr(self, 'finish_type') and self.finish_type == 'central_only':
            final_output = gru_out[:, self.n//2, :]

        if hasattr(self, 'finish_type') and self.finish_type == 'attention':
            central = gru_out[:, self.n//2, :]
            attn_weights = F.softmax(self.attn_lin(F.tanh(self.attn(central))),
                                     dim=1)

            attended_output = torch.bmm(attn_weights.unsqueeze(1), gru_out)

            final_output = torch.cat((central.squeeze(1),
                                      attended_output.squeeze(1)), 1)

        out = F.tanh(final_output)
        out = self.h2o(final_output)
        return out


class BiGRU(DiacModel):
    def __init__(self, *args, **kwargs):
        kwargs['model_type'] = 'gru'
        super(BiGRU, self).__init__(*args, **kwargs)
