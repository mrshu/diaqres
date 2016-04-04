import re
import sys
import requests
import click
import codecs

TAGS = re.compile(r'<[^>]+>', re.IGNORECASE)
DEBREE = re.compile(r'\%.+>', re.IGNORECASE)
TABLE_DEBREE = re.compile(r'^\![^\|]+\|', re.IGNORECASE)
PX_DEBREE = re.compile(r'px[^>]+>', re.IGNORECASE)
WHITESPACE = re.compile(r'\s+', re.IGNORECASE)
NON_LATIN_RE = r'[^\x00-\x7F\x80-\xFF\u0100-\u017F\u0180-\u024F\u1E00-\u1EFF]'

WORD = re.compile(r'\b\w\w+\b', re.IGNORECASE | re.UNICODE)

WIKI_DUMPS_URL = 'https://dumps.wikimedia.org'
DUMP_DIR = '/{0}wiki/latest/'
DUMP_FILE = '{0}wiki-latest-pages-articles.xml.bz2'


def clean_dump(filename, ignore_character_group=NON_LATIN_RE, chunk_size=1024):
    outfile = '{}.clean'.format(filename)
    ignore_characters = re.compile(ignore_character_group,
                                   re.IGNORECASE | re.UNICODE)

    with codecs.open(filename, 'r', 'utf-8') as f:
        with codecs.open(outfile, 'w', 'utf-8') as out:
            while True:
                line = f.read(chunk_size)
                if not line:
                    break

                line = TAGS.sub('', line)
                line = DEBREE.sub('', line)
                line = TABLE_DEBREE.sub('', line)
                line = PX_DEBREE.sub('', line)
                line = ignore_characters.sub('', unicode(line))

                line = WHITESPACE.sub(' ', line)
                line = WHITESPACE.sub(' ', line)
                out.write(line)


def get_dump(lang, filename=None):
    # Taken from http://stackoverflow.com/a/1094933
    def sizeof_fmt(num, suffix='B'):
        for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)

    url = WIKI_DUMPS_URL + DUMP_DIR + DUMP_FILE
    url = url.format(lang)

    if filename is None:
        filename = DUMP_FILE.format(lang)

    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        total_length = int(r.headers.get('content-length'))
        label = 'Downloading {} ({})'.format(filename,
                                             sizeof_fmt(total_length))
        with click.progressbar(r.iter_content(chunk_size=1024),
                               length=int(total_length / 1024) + 1,
                               label=label) as bar:
            for chunk in bar:
                if chunk:
                    f.write(chunk)
                    f.flush()


def stats(filename, chunk_size=1024):
    from os.path import getsize

    count = 0
    accent_count = 0

    unique_chars = set()
    unique_words = set()
    words = {}

    file_size = getsize(filename)
    with codecs.open(filename, 'r', 'utf-8') as f:
        with click.progressbar(label=filename,
                               length=file_size) as bar:
            while True:
                line = f.read(chunk_size)
                if not line:
                    break

                bar.update(len(line.encode('utf-8')) + 1)
                line = line.lower()
                found_words = WORD.findall(line)

                clean_words = [remove_accents(word) for word in found_words]

                for word, clean_word in zip(found_words, clean_words):
                    if word in unique_words:
                        continue

                    if clean_word not in words:
                        words[clean_word] = 0

                    if clean_word != word:
                        words[clean_word] += 1

                    unique_words.add(word)

                for c in line:
                    unique_chars.add(c)
                    clean_c = remove_accents(c)
                    if not clean_c.isalpha():
                        continue

                    count += 1
                    if c != clean_c:
                        accent_count += 1

    percent = (accent_count/float(count))*100
    click.echo("Accent characters: {}/{} ({:.3}%)".format(accent_count,
                                                          count,
                                                          percent))
    click.echo()

    from collections import Counter
    c = Counter(words.values())

    n_all_words = 0
    n_clean_words = len(words)
    n_unique_words = len(unique_words)

    click.echo("       # words | # alternations")
    click.echo("---------------|---------------")
    for k, v in dict(c.items()).iteritems():
        click.echo("{:14} | {}".format(v, k))
        n_all_words += v * (k + 1)

    click.echo()
    click.echo("Number of all words: {}".format(n_all_words))
    click.echo("Number of all 'clean' words: {}".format(n_clean_words))
    click.echo("Number of unique words: {}".format(n_unique_words))

    lexdiff = float(n_unique_words)/n_clean_words
    click.echo("LexDiff score: {}/{} = {:.3}".format(n_unique_words,
                                                     n_clean_words,
                                                     lexdiff))

    click.echo("Top 5 most ambiguous words:")
    for w in sorted(words, key=words.get, reverse=True)[:5]:
        click.echo("{}\t{}".format(words[w], w))

    unique_chars = list(unique_chars)
    n_characters = len(unique_chars)
    click.echo("All unique characters: {}".format(n_characters))

    characters_line = unicode("".join(unique_chars))
    clean_characters_line = remove_accents(characters_line)
    accented_characters = []
    clean_characters = []
    for x, y in zip(list(characters_line),
                    list(clean_characters_line)):
        if x != y and y.isalpha() and x.isalpha():
            accented_characters.append(x)
            clean_characters.append(y)

    click.echo("Accented characters: {}".format(len(accented_characters)))
    click.echo("\t" + "".join(accented_characters))
    click.echo("\t" + "".join(clean_characters))

    class_list_file = "{}.classes".format(filename)
    save_object(unique_chars, class_list_file)
    click.echo("List of classes was saved into {}".format(class_list_file))


def remove_accents(line):
    import unicodedata
    ndfk = unicodedata.normalize('NFKD', line)
    return "".join([c for c in ndfk if not unicodedata.combining(c)])


def file_to_grapheme_corpus(filename, window, chunk_size=100):
    corpus = []
    f = codecs.open(filename, 'r', 'utf-8')
    return readable_to_grapheme_corpus(f, window, chunk_size)


def string_to_grapheme_corpus(string, window, chunk_size=100):
    from io import StringIO
    out = StringIO()
    out.write(unicode(string))
    # Start at the beginning of the file/string
    out.seek(0)
    return readable_to_grapheme_corpus(out, window, chunk_size)


def readable_to_grapheme_corpus(readable, window, chunk_size=100):
    prev_data = ' ' * window
    while True:
        data = readable.read(chunk_size + window)
        if not data:
            break

        if len(data) < chunk_size + window:
            data = data + ' ' * window

        line = prev_data[-2*window:] + data
        prev_data = data
        line = line.lower()
        clean_line = remove_accents(line)
        for i in range(window, len(line)-window):
            yield (zip(range(2*window + 1),
                       list(clean_line[i-window:i+1+window])),
                   line[i])


def save_object(object, file):
    import pickle
    with open(file, 'w') as f:
        pickle.dump(object, f)


def load_object(file):
    import pickle
    with open(file, 'r') as f:
        return pickle.load(f)


def test_model(model, filename, characters=None):
    from sklearn.metrics import classification_report
    import string
    f = codecs.open(filename, 'r', 'utf-8')
    line = f.read().lower()
    if characters:
        removed_characters = remove_accents(characters)
        d = dict(zip(characters, removed_characters))
        out_line = ''
        for c in line:
            if c in d:
                out_line += d[c]
            else:
                out_line += c
        restored_line = model.restore(out_line)
    else:
        restored_line = model.restore(remove_accents(line))
    print(line[:50])
    print(restored_line[:50])
    click.echo(classification_report(list(line),
                                     list(restored_line)))
