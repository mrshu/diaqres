import click


@click.group()
def main():
    pass


@main.command('clean-dump')
@click.option('--filename',
              type=click.Path(exists=True),
              required=True)
def clean_dump(filename):
    """Clean processed text from Wikipedia dump."""
    from .utils import clean_dump as clean
    click.echo('Starting to cleaning {}'.format(filename))
    clean(filename)
    click.echo('Finished cleaning {}'.format(filename))


@main.command('get-dump')
@click.argument('lang')
@click.option('--filename')
def get_dump(lang, filename):
    """Download a Wikipedia dump for a language."""
    from .utils import get_dump as download
    click.echo('Starting to download Wikipedia dump for lang {}.'.format(lang))
    download(lang, filename=filename)
    click.echo('Download finished')


@main.command('stats')
@click.option('--filename',
              type=click.Path(exists=True),
              required=True)
def stats(filename):
    """Print (diacritics) statistics on a given file"""
    from .utils import stats as print_stats
    click.echo('Starting to gather statistics on file {}'.format(filename))
    print_stats(filename)
    click.echo('Statistics printing finished')


if __name__ == '__main__':
    main()
