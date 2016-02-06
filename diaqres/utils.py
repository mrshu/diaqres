import re
import sys
import requests
import click
TAGS = re.compile(r'<[^>]+>', re.IGNORECASE)
DEBREE = re.compile(r'\%.+>', re.IGNORECASE)
TABLE_DEBREE = re.compile(r'^\![^\|]+\|', re.IGNORECASE)
PX_DEBREE = re.compile(r'px[^>]+>', re.IGNORECASE)

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
    url = WIKI_DUMPS_URL + DUMP_DIR + DUMP_FILE
    url = url.format(lang)

    if filename is None:
        filename = DUMP_FILE.format(lang)

    r = requests.get(url, stream=True)
    with open(filename, 'wb') as f:
        total_length = int(r.headers.get('content-length')) / 1024 + 1
        label = 'Downloading {}'.format(filename)
        with click.progressbar(r.iter_content(chunk_size=1024),
                               length=total_length,
                               label=label) as bar:
            for chunk in bar:
                if chunk:
                    f.write(chunk)
                    f.flush()