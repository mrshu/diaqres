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


@main.command('graphemize')
@click.option('--filename',
              type=click.Path(exists=True),
              required=True)
def graphemize(filename):
    from .utils import file_to_grapheme_corpus, load_object, save_object
    from .models import GraphemeBasedModel
    corpus = file_to_grapheme_corpus(filename, 5)
    classes = load_object("{}.classes".format(filename))
    model = GraphemeBasedModel()
    model.train(corpus, classes)
    click.echo(model.restore('Fantasticky program      '))
    save_object(model, "{}.model".format(filename))


@main.command('model-test')
@click.option('--modelpath',
              type=click.Path(exists=True),
              required=True)
@click.option('--filepath',
              type=click.Path(exists=True),
              required=True)
def model_test(modelpath, filepath):
    from .utils import file_to_grapheme_corpus, load_object, test_model
    model = load_object(modelpath)
    test_model(model, filepath)

if __name__ == '__main__':
    main()
