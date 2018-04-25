from process import parse_train_data, generate_xy
import sys

import torch
from torch.autograd import Variable
from sklearn.metrics import classification_report

import pandas as pd
pd.set_option('display.width', 1000)  # noqa
pd.set_option('display.max_columns', 500)  # noqa
pd.set_option('display.max_rows', 500)  # noqa
pd.set_option('display.height', 1000)  # noqa
pd.set_option('display.expand_frame_repr', False)  # noqa
from pandas_ml import ConfusionMatrix

if __name__ == "__main__":
    minibatch_len = int(sys.argv[3])

    test_data, input2id, output2id = parse_train_data(sys.argv[1])
    m = torch.load(sys.argv[2])
    print(m)
    print('n:', m.n)
    id2input = {v: k for k, v in m.input2id.items()}
    id2output = {v: k for k, v in m.output2id.items()}

    predictions = []
    truths = []

    minibatch_x = []
    minibatch_y = []
    errors = []

    teacher_forcing = False
    if hasattr(m, 'teacher_forcing'):
        teacher_forcing = m.teacher_forcing

    for i, (x, y) in enumerate(generate_xy(test_data, m.input2id, m.output2id,
                                           n=m.n)):

        if len(minibatch_x) < minibatch_len and not teacher_forcing:
            minibatch_x.append(x)
            minibatch_y.append(y)
            continue

        if teacher_forcing:
            if len(predictions) > m.n // 2:
                for i in range(m.n//2):
                    x[i] = m.input2id[predictions[-m.n//2+i]]

            minibatch_x = [x]
            minibatch_y = [y]

        # print(minibatch_x)
        # print(''.join(list(map(lambda x: id2input[x], minibatch_x[0]))))
        words = Variable(torch.LongTensor(minibatch_x))

        output = m(words)
        _, output_ids = torch.max(output, 1)

        predicted = [id2output[num] for num in output_ids.data]
        truth = [id2output[y] for y in minibatch_y]

        for (k, p, t) in zip(range(len(predicted)), predicted, truth):
            if p != t:
                text = ''.join([id2input[num] for num in minibatch_x[k]])
                errors.append((k, text, p, t))

        predictions.extend(predicted)
        truths.extend(truth)

        minibatch_x = []
        minibatch_y = []

        if i % 500 == 0 and i > 0:
            print(i)

    print(classification_report(truths, predictions, digits=5))
    import pprint
    pprint.pprint(errors)

    confusion_matrix = ConfusionMatrix(truths, predictions)
    print(confusion_matrix)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    confusion_matrix.plot(normalized=True)
    plt.savefig('{}_confusion_matrix.png'.format(sys.argv[2]))
