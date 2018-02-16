from model import BiGRU
from process import parse_train_data, generate_xy
import sys

import torch
from torch.autograd import Variable
from sklearn.metrics import classification_report

if __name__ == "__main__":
    embed_size = 32
    hidden_size = 32
    n_layers = 1
    dropout = 0.0
    n = 21

    test_data, input2id, output2id = parse_train_data(sys.argv[1])
    m = torch.load(sys.argv[2])
    id2input = {v: k for k, v in m.input2id.items()}
    id2output = {v: k for k, v in m.output2id.items()}

    predictions = []
    truths = []
    for i, (x, y) in enumerate(generate_xy(test_data, m.input2id, m.output2id,
                                           n=n)):

        words = Variable(torch.LongTensor(x))

        output = m(words)
        _, output_id = torch.max(output, 1)
        output_num = output_id.data[0]

        predictions.append(id2output[output_num])
        truths.append(id2output[y])

    print(classification_report(truths, predictions))