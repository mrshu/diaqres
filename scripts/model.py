import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGRU(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, output_vocab_size,
                 n_layers=1, bidirectional=True, dropout=0.0, n=11,
                 input2id=None, output2id=None):
        super(BiGRU, self).__init__()

        self.n = n
        self.input2id = input2id
        self.output2id = output2id

        self.embed = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, num_layers=n_layers,
                          dropout=dropout, bidirectional=bidirectional)

        self.hidden_size = hidden_size
        self.bidirectional_multiplier = 2 if bidirectional else 1

        self.h2o = nn.Linear(hidden_size * self.bidirectional_multiplier * n,
                             output_vocab_size)

        self.softmax = nn.LogSoftmax(dim=0)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp):
        emb = self.embed(inp)
        emb = self.dropout(emb)

        input = emb
        # input = emb.view(len(inp), 1, self.hidden_size)

        gru_out, hidden = self.gru(input)
        gru_out = gru_out.view(gru_out.size(0),
                               gru_out.size(1) * gru_out.size(2))
        gru_out = F.tanh(gru_out)

        out = self.h2o(gru_out)
        out = self.softmax(out)
        return out
