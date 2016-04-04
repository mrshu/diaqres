import click
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction import DictVectorizer, FeatureHasher
from diaqres.utils import string_to_grapheme_corpus
from sklearn.pipeline import Pipeline


class DiacriticsRestorationModel(object):

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
                                        n_features=len(classes)*20*self.window,
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
