from setuptools import setup

setup(
    name='diaqres',

    description='Diacritics Restoration toolkit',
    author='Marek Suppa',
    author_email='marek@suppa.sk',
    version='0.0.1',

    url='https://github.com/mrshu/diaqres',

    include_package_data=True,
    install_requires=['Click', 'requests', 'scipy', 'numpy', 'scikit-learn'],
    packages=['diaqres'],

    license="GPL 3.0",
    keywords=['nlp', 'diacritics restoration'],
    entry_points={
        'console_scripts': [
            'diaqres = diaqres:main'
        ]
    }
)
