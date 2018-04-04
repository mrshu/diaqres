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
    test_data, input2id, output2id = parse_train_data(sys.argv[1])
    m = torch.load(sys.argv[2])
    id2input = {v: k for k, v in m.input2id.items()}
    id2output = {v: k for k, v in m.output2id.items()}

    predictions = []
    truths = []
    for i, (x, y) in enumerate(generate_xy(test_data, m.input2id, m.output2id,
                                           m.n=n)):

        words = Variable(torch.LongTensor([x]))

        output = m(words)
        _, output_id = torch.max(output, 1)
        output_num = output_id.data[0]

        predictions.append(id2output[output_num])
        truths.append(id2output[y])

    print(classification_report(truths, predictions, digits=5))

    confusion_matrix = ConfusionMatrix(truths, predictions)
    print(confusion_matrix)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    confusion_matrix.plot(normalized=True)
    plt.savefig('{}_confusion_matrix.png'.format(sys.argv[2]))
