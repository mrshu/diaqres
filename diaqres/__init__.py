import click


@click.group()
def main():
    pass


@main.command('clean-dump')
@click.option('--filename',
              type=click.Path(exists=True),
              required=True)
def clean_dump(filename):
    from .utils import clean_dump as clean
    click.echo('Starting to cleaning {}'.format(filename))
    clean(filename)
    click.echo('Finished cleaning {}'.format(filename))


if __name__ == '__main__':
    main()
