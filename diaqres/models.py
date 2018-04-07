import click
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from diaqres.utils import string_to_grapheme_corpus
from sklearn.pipeline import Pipeline


class DiacriticsRestorationModel:

    def train(self, corpus):
        raise NotImplementedError

    def restore(self, string):
        raise NotImplementedError


class GraphemeBasedModel(DiacriticsRestorationModel):
    def __init__(self, window=5, input_classes=None):
        self.window = window
        self.input_classes = input_classes

    def train(self, corpus, classes=None, chunk_size=100000):
        self.vectorizer = FeatureHasher(non_negative=True,
                                        n_features=len(classes)*2*self.window,
                                        input_type='pair')
        self.clf = MultinomialNB()
        i = 0
        j = 0
        X = []
        Y = []
        for x, y in corpus:
            if x[self.window][1] in self.input_classes:
                X.append(x)
                Y.append(y)
                i += 1
            if i < chunk_size:
                continue

            j += 1
            click.echo("Running iteration {}".format(j))

            X = self.vectorizer.transform(X)
            self.clf.partial_fit(X, Y, classes)
            X = []
            Y = []
            i = 0

    def restore(self, string):
        corpus = []
        out = ''
        for x, y in string_to_grapheme_corpus(string, self.window):
            if x[self.window][1] in self.input_classes:
                x = self.vectorizer.transform([x])
                out += self.clf.predict(x)[0]
            else:
                out += y
        return out


class NoLearningBaselineModel(DiacriticsRestorationModel):
    def __init__(self, input_classes):
        pass

    def train(self, corpus, classes=None, chunk_size=100000):
        pass

    def restore(self, string):
        return string


class LSTMModel(DiacriticsRestorationModel):
    def __init__(self, window=5, input_classes=None, output_classes=None):
        from keras.models import Sequential
        from keras.layers.core import (TimeDistributedDense, Dense, Dropout,
                                       Activation)
        from keras.layers.embeddings import Embedding
        from keras.layers.recurrent import GRU, LSTM
        from keras.preprocessing import sequence

        self.window = window
        self.input_classes = input_classes
        self.output_classes = output_classes

        self.classes = self.input_classes + self.output_classes

        self.model = Sequential()
        self.model.add(Embedding(128, 10, input_length=2*window+1))
        self.model.add(LSTM(10))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(len(self.classes)))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='sparse_categorical_crossentropy',
                           optimizer='adam')

        self.model.summary()

    def _discretize(self, w):
        def numerize(l):
            if ord(l) > 128:
                return ord(' ')
            return ord(l)
        return list(map(lambda x: numerize(x), map(lambda x: x[1], w)))

    def train(self, corpus, classes=None, chunk_size=100000):
        i = 0
        j = 0
        X = []
        Y = []
        print('in here')
        for x, y in corpus:
            if x[self.window][1] in self.input_classes and \
               y in self.output_classes:
                X.append(self._discretize(x))
                Y.append(self.classes.index(y))
                i += 1

                if i % 10000 == 0:
                    click.echo('Iteration {}'.format(i))

        print(X, Y)
        self.model.fit(X, Y, nb_epoch=5, batch_size=32)

    def restore(self, string):
        import numpy as np
        out = ''
        for x, y in string_to_grapheme_corpus(string, self.window):
            if x[self.window][1] in self.input_classes:
                x = np.array([self._discretize(x)])
                out += self.classes[(self.model.predict_classes(x,
                                                                batch_size=1,
                                                                verbose=0))]
            else:
                out += y
        return out
