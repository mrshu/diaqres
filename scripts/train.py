from model import DiacModel
from process import parse_train_data, generate_xy
from sacred import Experiment
import numpy as np
from tensorboardX import SummaryWriter
import datetime
from sklearn.utils import shuffle

import torch
from torch.autograd import Variable

ex = Experiment('some_config')


@ex.config
def my_config():
    embed_size = 32
    hidden_size = 50
    n_layers = 1
    dropout = 0.0
    out_dropout = 0.0
    filename = '../text/workable'
    n = 21
    runs = 3
    save_each_epochs = 20000
    print_each_epochs = 500
    loss_avg_n_epochs = 1000
    minibatch_len = 100
    cuda = True
    clip = 0.25
    lr = 0.002
    weight_decay = 1.2e-6
    teacher_forcing = False
    minibatch_shuffle = True
    model_type = 'gru'


@ex.automain
def main(embed_size, hidden_size, n_layers, dropout, out_dropout, filename,
         n, runs, save_each_epochs, print_each_epochs, loss_avg_n_epochs,
         minibatch_len, cuda, clip, lr, weight_decay, teacher_forcing,
         minibatch_shuffle, model_type):

    train_data, input2id, output2id = parse_train_data(filename)
    if teacher_forcing:
        input2id = output2id.copy()

    print(input2id)
    print(output2id)
    id2input = {v: k for k, v in input2id.items()}
    id2output = {v: k for k, v in output2id.items()}

    writer = SummaryWriter(comment='_{}'.format(ex.current_run._id))
    writer.add_text('config', str(ex.current_run.config))

    m = DiacModel(len(input2id), embed_size, hidden_size, len(output2id),
                  n_layers=n_layers, dropout=dropout, n=n,
                  input2id=input2id, output2id=output2id,
                  model_type=model_type, out_dropout=out_dropout)

    if cuda:
        m = m.cuda()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(m.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    losses = []

    minibatch_x = []
    minibatch_y = []

    tchr_forcing = teacher_forcing
    m.teacher_forcing = teacher_forcing
    for r in range(runs):
        for i, (x, y) in enumerate(generate_xy(train_data, input2id, output2id,
                                               n=n,
                                               teacher_forcing=tchr_forcing)):

            optimizer.zero_grad()
            m.zero_grad()

            if tchr_forcing:
                tf, x = x
                minibatch_x.append(x)
                minibatch_y.append(y)
                minibatch_x.append(tf)
                minibatch_y.append(y)
            else:
                minibatch_x.append(x)
                minibatch_y.append(y)

            if i % minibatch_len != 0:
                continue

            p = np.random.permutation(len(minibatch_x))
            if minibatch_shuffle:
                minibatch_x = np.array(minibatch_x)[p]
                minibatch_y = np.array(minibatch_y)[p]

            words = Variable(torch.LongTensor(minibatch_x))
            if cuda:
                words = words.cuda()

            output = m(words)

            out_y = Variable(torch.LongTensor(minibatch_y))
            if cuda:
                out_y = out_y.cuda()

            loss = criterion(output, out_y)

            writer.add_scalar('loss', loss.data[0], i)

            losses.append(loss.data[0])
            if len(losses) > loss_avg_n_epochs:
                losses.pop(0)

            writer.add_scalar('avg_loss', np.mean(losses), i)
            txt = ''.join(list(map(lambda x: id2input[x], x)))
            # print('text: {} m: {} y: {}'.format(txt,
            #                                     txt[len(txt)//2],
            #                                     id2output[y]))

            if i % print_each_epochs == 0:
                # invert the minibatch back
                ip = np.argsort(p)
                if minibatch_shuffle:
                    minibatch_x = np.array(minibatch_x)[ip]
                    minibatch_y = np.array(minibatch_y)[ip]

                print('{} ({}) loss: {} avg_loss: {}'.format(i, r,
                                                             loss.data[0],
                                                             np.mean(losses)))

                print('text: {}'.format(''.join(list(map(lambda x: id2input[x],
                                                         x)))))
                print('output:\t\t\t{}'.format(id2output[y]))
                _, output_id = torch.max(output, 1)
                output_id_data = output_id.data.cpu().numpy()
                if minibatch_shuffle:
                    output_id_data = output_id_data[ip]

                output_num = output_id_data[-1]
                print('predicted output:\t{}'.format(id2output[output_num]))
                print()

                y_true = ''.join([id2output[x] for x in minibatch_y])
                y_pred = ''.join([id2output[x] for x in output_id_data])
                print("y_pred: {}".format(y_pred))
                print("y_true: {}".format(y_true))
                print()

            if i % save_each_epochs == 0 and i > 0:
                dt = datetime.datetime.now().isoformat()
                torch.save(m, 'model_{}_{}'.format(ex.current_run._id, dt))

            loss.backward()

            torch.nn.utils.clip_grad_norm(m.parameters(), clip)
            optimizer.step()

            minibatch_x = []
            minibatch_y = []
