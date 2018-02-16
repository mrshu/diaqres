from process import parse_train_data, generate_xy, strip_accents
import sys

from sklearn.metrics import classification_report
import pandas as pd
pd.set_option('display.width', 1000)  # noqa
pd.set_option('display.max_columns', 500)  # noqa
pd.set_option('display.max_rows', 500)  # noqa
pd.set_option('display.height', 1000)  # noqa
pd.set_option('display.expand_frame_repr', False)  # noqa
from pandas_ml import ConfusionMatrix

if __name__ == "__main__":
    n = 21
    test_data, input2id, output2id = parse_train_data(sys.argv[1])
    id2input = {v: k for k, v in input2id.items()}
    id2output = {v: k for k, v in output2id.items()}

    predictions = []
    truths = []
    for i, (x, y) in enumerate(generate_xy(test_data, input2id, output2id,
                                           n=n)):

        predictions.append(strip_accents(id2output[y]))
        truths.append(id2output[y])

    print(classification_report(truths, predictions))

    confusion_matrix = ConfusionMatrix(truths, predictions)
    print(confusion_matrix)
