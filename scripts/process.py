import sys
import unicodedata
from collections import defaultdict


def strip_accents(text):
    """
    Strip accents from input String.

    :param text: The input string.
    :type text: String.

    :returns: The processed String.
    :rtype: String.

    Note: from https://stackoverflow.com/a/31607735
    """
    try:
        text = unicode(text, 'utf-8')
    # unicode is a default on python 3
    except (TypeError, NameError):
        pass

    if text == '<unk>':
        return ' '

    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

SLOVAK_ALPHABET = ' 0123456789aäbcdefghijklmnopqrstuvwxyzáčďéíĺľňóôŕšťúýž'


def generate_xy(input, input2id, output2id, n=11,
                alphabet=SLOVAK_ALPHABET, teacher_forcing=False):
    N = len(input)
    i = 0
    while i+n < N:
        subset = input[i:i+n]
        y = subset[n//2]
        y = output2id[y]
        if teacher_forcing:
            p = n//2+1
            x = subset[:p] + list(map(strip_accents, subset[p:]))
        else:
            x = list(map(strip_accents, subset))
        yield list(map(lambda x: input2id[x], x)), y
        i += 1


def parse_train_data(filename):
    with open(filename) as f:
        contents = list(f.read())

    contents = [x if x in SLOVAK_ALPHABET else '<unk>' for x in contents]
    unique_words = set(contents)
    clean_words = map(strip_accents, unique_words)
    input2id = defaultdict(lambda: len(input2id))
    output2id = defaultdict(lambda: len(output2id))

    unk_id = output2id['<unk>']
    for w in contents:
        id = output2id[w]
    output2id = dict(output2id)

    for w in clean_words:
        id = input2id[w]
    input2id = dict(input2id)

    return list(contents), input2id, output2id

if __name__ == '__main__':

    contents, input2id, output2id = parse_train_data(sys.argv[1])
    input_vocab_size = len(input2id)
    output_vocab_size = len(output2id)

    for x, y in generate_xy(contents, input2id, output2id):
        print(x, y)
