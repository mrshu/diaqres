from model import BiGRU
from process import parse_train_data, generate_xy
from sacred import Experiment
import numpy as np
from tensorboardX import SummaryWriter
import datetime

import torch
from torch.autograd import Variable

ex = Experiment('some_config')


@ex.config
def my_config():
    embed_size = 32
    hidden_size = 32
    n_layers = 1
    dropout = 0.0
    filename = '../text/workable'
    n = 21
    runs = 3


@ex.automain
def main(embed_size, hidden_size, n_layers, dropout, filename, n, runs):
    train_data, input2id, output2id = parse_train_data(filename)
    id2input = {v: k for k, v in input2id.items()}
    id2output = {v: k for k, v in output2id.items()}

    writer = SummaryWriter()

    m = BiGRU(len(input2id), embed_size, hidden_size, len(output2id),
              n_layers=n_layers, dropout=dropout, n=n,
              input2id=input2id, output2id=output2id)

    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=0.02)

    for _ in range(runs):
        for i, (x, y) in enumerate(generate_xy(train_data, input2id, output2id,
                                               n=n)):
            optimizer.zero_grad()

            words = Variable(torch.LongTensor(x))
            output = m(words)

            out_y = Variable(torch.LongTensor([y]))

            loss = criterion(output, out_y)

            writer.add_scalar('loss', loss.data[0], i)

            if i % 500 == 0:
                print('{} loss: {}'.format(i, loss.data[0]))

                print('text: {}'.format(''.join(list(map(lambda x: id2input[x],
                                                         x)))))
                print('output:\t\t\t{}'.format(id2output[y]))
                _, output_id = torch.max(output, 1)

                output_num = output_id.data[0]
                print('predicted output:\t{}'.format(id2output[output_num]))
                print()

            if i % 10000 == 0 and i > 0:
                dt = datetime.datetime.now().isoformat()
                torch.save(m, 'model_{}'.format(dt))

            loss.backward()
            optimizer.step()
