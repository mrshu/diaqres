import re
import sys
import requests
import click
TAGS = re.compile(r'<[^>]+>', re.IGNORECASE)
DEBREE = re.compile(r'\%.+>', re.IGNORECASE)
TABLE_DEBREE = re.compile(r'^\![^\|]+\|', re.IGNORECASE)
PX_DEBREE = re.compile(r'px[^>]+>', re.IGNORECASE)

WORD = re.compile(r'\b\w\w+\b', re.IGNORECASE | re.UNICODE)

WIKI_DUMPS_URL = 'https://dumps.wikimedia.org'
DUMP_DIR = '/{0}wiki/latest/'
DUMP_FILE = '{0}wiki-latest-pages-articles.xml.bz2'


def clean_dump(filename):
    outfile = '{}.clean'.format(filename)
    with open(filename, 'r') as f:
        with open(outfile, 'w') as out:
            for line in f:
                line = TAGS.sub('', line)
                line = DEBREE.sub('', line)
                line = TABLE_DEBREE.sub('', line)
                line = PX_DEBREE.sub('', line)
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


def stats(filename):
    import codecs
    from os.path import getsize

    def remove_accents(line):
        import unicodedata
        ndfk = unicodedata.normalize('NFKD', line)
        return "".join([c for c in ndfk if not unicodedata.combining(c)])

    count = 0
    accent_count = 0

    unique_words = set()
    words = {}

    file_size = getsize(filename)
    with codecs.open(filename, 'r', 'utf-8') as f:
        with click.progressbar(label=filename,
                               length=file_size) as bar:
            for line in f:
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
                    clean_c = remove_accents(c)
                    if not clean_c.isalpha():
                        continue

                    count += 1
                    if c != clean_c:
                        accent_count += 1

    percent = (accent_count/float(count))*100
    output = "Accent characters: {}/{} ({:.3}%)".format(accent_count,
                                                        count,
                                                        percent)
    click.echo(output)
    click.echo()

    from collections import Counter
    c = Counter(words.values())

    n_all_words = 0

    click.echo("       # words + # alternations")
    click.echo("---------------|---------------")
    for k, v in dict(c.items()).iteritems():
        click.echo("{:14} | {}".format(v, k))
        n_all_words += v * (k + 1)

    click.echo()
    click.echo("Numer of all words: {}".format(n_all_words))
    click.echo("Numer of unique words: {}".format(len(unique_words)))
    click.echo("Top 5 most ambiguous words:")
    for w in sorted(words, key=words.get, reverse=True)[:5]:
        click.echo("{}\t{}".format(words[w], w))
